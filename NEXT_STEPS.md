# Próximos pasos

Estado actual: `server.py` + `index.html` corren GPT-2 / Qwen2.5-0.5B-Instruct / Pythia-410m en local con
PyTorch + **TransformerLens 4.2** (`TransformerBridge`), enganchando la activación **real por neurona**
de la MLP de cada capa con el hook estándar `blocks.{i}.mlp.hook_post` y enviándola por WebSocket al heatmap.

Este documento cubre tres ampliaciones independientes:

1. [Dockerizar](#1-dockerizar)
2. [Migrar a TransformerLens](#2-migrar-a-transformerlens) — **implementado**
3. [Migrar a nnsight](#3-migrar-a-nnsight)

Al final hay una sección de [mejoras pendientes](#4-mejoras-pendientes).

---

## 1. Dockerizar

**Por qué:** aislar Python 3.14 + torch + transformers, evitar tocar el sistema y poder mover el proyecto
a otra máquina con un comando. El único punto delicado es el modelo: GPT-2 son ~526 MB y Qwen ~1 GB, así
que se deben cachear en un **volumen** (no dentro de la imagen) para no reconstruirla ni re-descargar.

### `Dockerfile`

```dockerfile
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# Solo CPU. Para GPU ver la variante más abajo.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install "transformer-lens>=4.2" fastapi "uvicorn[standard]" websockets

COPY server.py index.html ./

EXPOSE 8000
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/models')" || exit 1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

> El orden importa: instalar `torch` desde el índice **CPU** reduce la imagen de ~3 GB (con CUDA) a ~1.5 GB.
> `PIP_NO_CACHE_DIR=1` evita que la imagen engorde con las ruedas descargadas.

### `.dockerignore`

```
.venv
__pycache__
*.pyc
```

### `docker-compose.yml`

```yaml
services:
  visualizador:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - hf-models:/models          # el modelo se descarga aquí la primera vez
    # límite de RAM (el server ya descarga el modelo anterior al cambiar)
    deploy:
      resources:
        limits:
          memory: 6g
    restart: unless-stopped

volumes:
  hf-models:
```

### Comandos

```bash
docker compose up --build        # primera vez: descarga el modelo dentro del volumen
# abrir http://localhost:8000
docker compose down              # los modelos quedan en el volumen hf-models
docker compose down -v           # borra también los modelos (se re-descargarán)
```

### Variante GPU (opcional)

Cambiar la base y la rueda de torch, y activar la GPU en compose:

```dockerfile
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
# torch con CUDA (índice por defecto ya trae la build cu)
RUN pip install --break-system-packages torch transformers fastapi "uvicorn[standard]" websockets
```

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Notas:
- Si se usa GPU conviene cargar el modelo en `float16`/`bfloat16` (`AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16)`)
  para aprovechar la VRAM; en CPU dejar `float32`.
- Para muchos usuarios concurrentes, los hooks globales (`activations` compartido) se pisan entre conexiones.
  Habría que usar un `LanguageModel`/hooks **por conexión** o serializar las generaciones.
- Solo hace falta exponer el puerto 8000; la UI se sirve desde el mismo proceso (`GET /`).

---

## 2. Migrar a TransformerLens

**Estado:** implementado en `server.py` con **TransformerLens 4.2**. En la v4 se eliminó
`HookedTransformer`; el reemplazo es `TransformerBridge.boot_transformers(...)`.

**Qué es:** librería de interpretabilidad mecanicista. En lugar de calcular a mano las rutas internas
(`model.transformer.h[i].mlp.act`, `model.model.layers[i].mlp.act_fn`), expone un **caché de activaciones
con nombres estándar** por capa y por hook point, y trae utilidades de intervención.

**Qué gana este proyecto:**
- Un nombre estable para "las neuronas de la MLP": `blocks.{i}.mlp.hook_post` (dimensión `d_mlp`).
- En MLPs con compuerta (Qwen/LLaMA) ese hook ya es `act(gate_proj(x)) * up_proj(x)`, es decir, la
  **neurona completa**; antes solo se capturaba `act_fn(gate_proj(x))`.
- Ablación / activation patching en una línea, para comprobar *qué hace* una neurona (no solo verla).
- Multi-modelo consistente: `gpt2`, `pythia-*`, `llama`, `gemma`, `qwen`, etc., con los mismos nombres.

**Instalación:** `pip install "transformer-lens>=4.2"`

**Implementación actual (reemplazo del hook manual):**

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(hf_id, device="cpu", tokenizer=tokenizer)

# Cachear solo las 12/24 activaciones MLP, no los ~637 hook points del modelo.
logits, cache = bridge.run_with_cache(
    input_ids,
    names_filter=lambda name: name.endswith(".mlp.hook_post"),
    return_cache_object=False,
)
acts = cache["blocks.0.mlp.hook_post"]            # [1, n_tokens, d_mlp]
neurons_layer0 = acts[0, -1, :].tolist()          # último token, todas las neuronas
```

Decisiones de la migración:
- **Sin `enable_compatibility_mode()`:** ese modo pliega LayerNorm y centra pesos (otra semántica y
  ~1,5 GB extra de RAM en GPT-2). Los pesos HF crudos dan las mismas activaciones que antes, y el alias
  `blocks.{i}.mlp.hook_post` ya está registrado por el bridge.
- **`names_filter`:** el caché por defecto guarda todos los hook points (637 en GPT-2). Filtrar a
  `.mlp.hook_post` baja el pico de ~2,7 GB a ~1,0 GB en GPT-2.
- `run_with_cache` devuelve `(logits, cache)`, donde `logits` es un `Tensor`, no un objeto de Hugging Face.

Nombres útiles del caché (por capa `i`):

| Hook point | Dimensión | Qué es |
|---|---|---|
| `blocks.{i}.mlp.hook_post` | `d_mlp` | **las neuronas** (post-activación, antes de `down_proj`) |
| `blocks.{i}.hook_mlp_out` | `d_model` | salida de la MLP |
| `blocks.{i}.attn.hook_z` | `n_heads × d_head` | atención por cabeza |
| `blocks.{i}.hook_resid_pre` / `.hook_resid_post` | `d_model` | residual stream |

**Intervenciones (lo que hoy no puedes hacer):**

```python
# apagar la neurona 2910 de la capa 11
def ablate(act, hook):
    act[:, :, 2910] = 0.0
    return act

logits = bridge.run_with_hooks(input_ids, fwd_hooks=[("blocks.11.mlp.hook_post", ablate)])

# activation patching: inyectar activaciones de un prompt A en un prompt B
_, cache_a = bridge.run_with_cache(input_ids_a, names_filter="blocks.5.mlp.hook_post")
logits_b = bridge.run_with_hooks(
    input_ids_b,
    fwd_hooks=[("blocks.5.mlp.hook_post", lambda act, hook: cache_a[hook.name])],
)
```

**A tener en cuenta:** el bridge soporta un conjunto de arquitecturas verificadas (no cualquiera de
Hugging Face); `TransformerBridge.check_model_support(hf_id)` lo consulta sin cargar pesos. Si necesitas
un modelo exótico, nnsight es más flexible.

---

## 3. Migrar a nnsight

**Qué es:** librería que envuelve un modelo Hugging Face **sin convertirlo** y permite leer/modificar
activaciones con sintaxis Python dentro de un contexto `trace()`. Su ventaja única: puede ejecutarse
**contra modelos remotos** (NDIF), p. ej. LLaMA-70B, sin GPU local.

**Instalación:** `pip install nnsight`

**Lectura de neuronas (mismo ejemplo que arriba):**

```python
from nnsight import LanguageModel

model = LanguageModel("Qwen/Qwen2.5-0.5B-Instruct", device_map="cpu", dispatch=True)

with model.trace(prompt):
    neurons = model.model.layers[3].mlp.act_fn.output.save()   # [1, seq, d_mlp]

print(neurons.value[0, -1, :])   # últimas activaciones por neurona de la capa 3
```

**Intervención:**

```python
with model.trace(prompt):
    acts = model.model.layers[11].mlp.act_fn.output
    acts[:, :, 2910] = 0.0        # apagar la neurona 2910
    out = model.lm_head.output.save()
```

**Modelos grandes sin GPU (NDIF):**

```python
big = LanguageModel("meta-llama/Meta-Llama-3.1-70B-Instruct", remote=True)
with big.trace(prompt):
    neurons = big.model.layers[40].mlp.act_fn.output.save()
```

**Cómo encaja aquí:** igual que TransformerLens, reemplaza los hooks manuales dentro de `server.py`, pero
además abre la puerta a modelos enormes (con `remote=True`) manteniendo *algún* acceso a activaciones, algo
que Groq/Ollama no permiten. La limitación es que las trazas remotas tienen latencia y cuota.

**Comparativa rápida para decidir:**

| Necesidad | Herramienta |
|---|---|
| Ver neuronas con el mínimo cambio | `transformers` (actual) |
| Experimentar: ablar, parchear, buscar circuitos | TransformerLens |
| Modelos arbitrarios de HF o gigantes remotos | nnsight |

---

## 4. Mejoras pendientes

- **MLP con compuerta (Qwen/LLaMA/Gemma):** resuelto con `blocks.{i}.mlp.hook_post`, que es
  `act_fn(gate_proj(x)) * up_proj(x)` (la neurona completa, antes de `down_proj`).
- **Atención:** añadir un heatmap de cabezas de atención (`attn.hook_z` / `attn.hook_pattern`).
- **Comparar prompts:** guardar dos pasadas y mostrar el *diferencial* de activación (qué neuronas cambian).
- **Ablación interactiva:** clic en una neurona → apagarla y ver cómo cambia el texto generado.
- **Exportar:** volcar las activaciones de una generación a CSV/Parquet para analizarlas fuera.
- **Rendimiento:** el servidor mantiene un solo modelo en RAM; si agregas concurrencia, aísla `activations`
  por conexión.
