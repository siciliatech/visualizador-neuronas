# Próximos pasos

Estado actual: `server.py` + `index.html` corren GPT-2 / Qwen2.5-0.5B-Instruct / Pythia-410m en local con
PyTorch + **TransformerLens 4.2** (`TransformerBridge`), enganchando la activación **real por neurona**
de la MLP de cada capa con el hook estándar `blocks.{i}.mlp.hook_post` y enviándola por WebSocket al heatmap.

Este documento resume dos cambios principales, ambos implementados:

1. [Dockerizar](#1-dockerizar) — **implementado**
2. [Migrar a TransformerLens](#2-migrar-a-transformerlens) — **implementado**

Al final hay una sección de [mejoras pendientes](#4-mejoras-pendientes).

---

## 1. Dockerizar

**Estado:** implementado en `Dockerfile`, `.dockerignore` y `docker-compose.yml`.

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
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install -r requirements.txt

COPY server.py index.html ./

EXPOSE 8000
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

> Las dependencias se fijan en `requirements.txt`. El orden importa: instalar `torch==2.14.1` desde el
> índice **CPU** reduce la imagen de ~3 GB (con CUDA) a ~1.5 GB. El build verificado instala
> `torch 2.14.1+cpu` con `torch.version.cuda is None`; `PIP_NO_CACHE_DIR=1` evita que la imagen engorde
> con las ruedas descargadas.

**RAM medida en el proceso servidor (macOS, CPU):** GPT-2 ~1,2 GB tras cargar y ~1,4 GB de pico con
512 tokens; Qwen2.5-0.5B ~2,3 GB y ~2,7 GB respectivamente. El límite de 6 GB del compose deja margen
para el modelo y el sistema.

### `.dockerignore`

```
.venv/
venv/
__pycache__/
*.pyc
.git/
models/
.cache/
*.log
.DS_Store
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

Notas:
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
Hugging Face); `TransformerBridge.check_model_support(hf_id)` lo consulta sin cargar pesos.

---

## 4. Mejoras pendientes

- **Rendimiento base:** **implementado** — `pos_slice=-1` limita el caché de activaciones/atención a la
  última query, y el forward de calentamiento se reutiliza para muestrear el primer token (sin una
  segunda pasada idéntica por el prompt).
- **KV cache:** **implementado** — el prefill procesa el prompt completo y cada token siguiente pasa solo
  el token nuevo con `past_key_values`; las etiquetas de atención conservan el contexto completo. Si el
  modelo no devuelve caché, se vuelve al forward completo. `VISUALIZER_KV_CACHE=0` permite comparar o
  revertir sin tocar código. Con GPT-2 y 512 tokens de contexto el coste por token bajó de ~0,158 s a
  ~0,012 s (~13×).
- **Docker y RAM:** **implementado** — el `Dockerfile` fija `torch==2.14.1` desde el índice CPU
  (build verificado con `torch 2.14.1+cpu`, sin CUDA) y el README documenta el RSS medido para GPT-2 y
  Qwen. El compose mantiene el límite de 6 GB.
- **Payload por capa (protocolo 2):** **implementado** — el cliente pide `select_layer`; el servidor
  guarda la capa por conexión, responde con `layer` usando el último estado completo y envía en cada
  `step` solo `activations["layer_N"]` más `stats` globales para el informe. El overview (`top`) y la
  atención siguen completos. Medido con GPT-2: ~32–36 KiB por step (antes ~258 KiB); Qwen ~67 KiB
  (antes ~0,8 MB). El cambio de capa es inmediato y no vacía el heatmap.
- **Muestreo y semántica del informe:** **implementado** — la UI expone `top_p` (0–1) y
  `no_repeat_ngram` (0–10) además de los controles existentes; ambos viajan en `params` y se muestran
  en el informe. El informe define “activación más intensa” como máximo valor absoluto (muestra el
  signo), etiqueta la media como `|activación|`, aclara que el récord de atención sale del top-8
  enviado y ya no describe cada token como una “pasada completa” (el prompt se procesa una vez y los
  tokens siguientes reutilizan el KV cache cuando está disponible).
- **Concurrencia y locks:** **implementado** — `load_lock` evita cargas duplicadas del mismo modelo y
  `generation_lock` serializa las generaciones. El segundo cliente recibe `queued` y espera; las
  activaciones, la atención y la decodificación son locales a cada generación, así que no se mezclan.
- **Decodificación incremental:** **implementado** — cada `step` envía `delta` (con `token` como alias),
  decodificado desde el prefijo completo para no partir caracteres UTF-8 multibyte; los tokens parciales
  de atención quedan vacíos, los especiales se etiquetan con su nombre y `done.generated_text` permite
  reconciliar el stream. El smoke test verifica que no haya `�` ni diferencias entre el stream y el texto
  final.
- **Errores y validación del WebSocket:** **implementado** — Pydantic valida `generate` antes de tocar el
  modelo; los errores de entrada (`invalid_request`, `invalid_json`) no cierran el socket y los fallos
  internos se reportan con `recoverable: false` antes de cerrar con código 1011.
- **MLP con compuerta (Qwen/LLaMA/Gemma):** resuelto con `blocks.{i}.mlp.hook_post`, que es
  `act_fn(gate_proj(x)) * up_proj(x)` (la neurona completa, antes de `down_proj`).
- **Atención:** **implementado** — se captura `blocks.{i}.attn.hook_pattern`, se toma la última query
  y se envían las top-8 conexiones por cabeza; la UI dibuja el fan y el mapa cabezas × tokens.
- **Atención explicada:** **implementado** — introducción en lenguaje llano, frase «Lectura» generada
  con los datos reales de la cabeza seleccionada, etiquetas sin jerga («última palabra», «palabras en
  las que se fija») e iconos de ayuda en los dos gráficos.
- **Atención honesta:** **implementado** — la leyenda y el tooltip aclaran que solo se dibuja el top-8
  por cabeza; las celdas restantes llevan un punto tenue para no confundirlas con atención cero. Los
  tokens especiales de los modelos chat se etiquetan con su nombre real en vez de quedar en blanco.
- **Forma del mapa:** **implementado** — el selector **Forma** dibuja las neuronas como cuadrícula
  (por defecto), cerebro o chip. Es un adorno: cada neurona se asigna a una celda de la máscara y
  hover, clic, brillo y anillo de selección usan esa posición. No cambia valores, rankings ni exportaciones.
- **Ayuda contextual:** **implementado** — cada control de la barra superior tiene un icono **?** que abre
  un popover accesible (teclado, `aria-expanded`, cierre con Esc o clic fuera) con qué es, para qué sirve
  y un consejo. Los textos están en el objeto `HELP` de `index.html`.
- **Escala del heatmap:** **implementado** — la casilla **escala por capa** ajusta los colores al máximo
  absoluto de la capa visible para distinguir activaciones pequeñas frente a valores atípicos. La
  escala global sigue siendo el valor por defecto y permite comparar magnitudes entre capas; el cambio
  es solo visual y no altera valores, rankings, informes ni exportaciones. En el modo diff se mantiene
  la escala propia de `máx |Δ|`.
- **Canvas y retina:** **implementado** — el heatmap y la animación se limitan a ~30 fps, la atención
  solo se redibuja cuando cambia y los canvas escalan su backing store por `devicePixelRatio` (máximo
  2×) manteniendo el tamaño CSS.
- **Tests, lint y CI:** **implementado** — `pyproject.toml` concentra la configuración de Ruff y pytest;
  `ruff check`/`ruff format --check` pasan, la suite unitaria excluye los tests `slow` y `ui`, y
  `tests/test_integration_slow.py` valida un forward real de GPT-2 con KV cache. El workflow de GitHub
  Actions usa Python 3.12, caché de Hugging Face, Ruff y `pytest -q -m "not slow and not ui"`; además
  instala Chromium y ejecuta `pytest -q -m ui`, que prueba la ablación con semilla sobre `index.html`
  usando un WebSocket simulado, sin modelos ni FastAPI.
- **Logging y salud:** **implementado** — se sustituyó `print` por `logging` con tiempos de carga,
  forward (contexto, entrada y modo de caché), tamaño de cada payload y errores con traza. `/health`
  informa `model_key` y `ready` sin cargar ni descargar el modelo. `VISUALIZER_LOG_LEVEL` permite ajustar
  el nivel.
- **Informe automático:** **implementado** — resume la generación en lenguaje claro (capa y neurona
  con mayor |activación|, capa con mayor media |activación|, atención más fuerte del top-8, todos los
  parámetros de muestreo y motivo de parada) y explica los términos. Incluye botón para copiar el
  informe.
- **Detalles de robustez:** **implementado** — `index.html` se sirve con una ruta resuelta desde
  `server.py` (no depende del directorio de trabajo), EOS se compara contra un `set` para admitir
  `eos_token_id` escalar o lista, y las etiquetas del overview/detalle usan el `top_k` del servidor. El
  panel ordena por activación con signo y lo explicita para no confundirlo con la magnitud.
- **Comparar prompts:** **implementado** — con **todas las capas** activado, **Guardar A** conserva la
  primera pasada y **Comparar** enfrenta la generación actual con A en la capa visible. El selector
  **Vista** permite comparar la **media de toda la generación**, un **paso concreto por posición** o
  un **token coincidente por contenido**. La alineación por contenido usa la subsecuencia común más
  larga sobre texto normalizado, puede emparejar tokens en posiciones distintas y respeta el orden de
  generación; los repetidos se emparejan en orden. No es una alineación semántica ni por subpalabras.
  El panel muestra las 12 neuronas con mayor `|Δ|`, el signo de `B − A` y los tokens generados en cada
  pasada. **Ver diff en heatmap** reemplaza el mapa, ranking, tooltip, detalle y sparkline por el
  diferencial, con escala propia de `máx |Δ|`; **Ver heatmap actual** lo restaura. Cambiar de capa
  recalcula todo. Exige el mismo modelo.
- **Ablación interactiva:** **implementado** — tras una pasada normal, el botón **Ablar y regenerar**
  repite el mismo modelo, prompt y parámetros con una neurona MLP forzada a `0`. El cliente envía
  `ablation: {layer, neuron}`; el servidor valida el rango y añade un hook temporal a
  `blocks.{layer}.mlp.hook_post` que clona la activación y apaga solo esa neurona. El hook se retira en
  `finally` tras cada forward, también con KV cache, y la captura muestra el `0` real. El panel compara
  el texto base con el ablacionado. Cada generación registra una semilla (`meta.seed`) y la ablación
  reutiliza la de la base, así que con muestreo la comparación es reproducible. No es activation
  patching y el texto puede no cambiar; para una lectura simple conviene `greedy`.
- **Exportar activaciones:** **implementado** — la casilla **todas las capas** activa `capture_all`
  en el mensaje `generate`; el servidor envía entonces todas las capas en cada `step` y el cliente las
  guarda como `Float32Array`. Al terminar, **CSV capa** descarga la capa visible, **CSV todas** descarga
  todas las capas en un único archivo con formato `step,token,layer,neuron,activation` (una fila por
  neurona, capa y token, ordenado por capa y paso) y **Parquet** descarga todas las capas en formato
  columnar. El Parquet se escribe en el navegador sin dependencias: un *row group*, columnas
  `required`, `PLAIN` sin compresión y *Data Page v1*, con esquema `step INT32`,
  `token BYTE_ARRAY (UTF8)`, `layer INT32`, `neuron INT32` y `activation FLOAT`. El modo normal
  mantiene el payload de una sola capa. La preparación es en memoria y pide confirmación si el archivo
  estimado supera los 2 millones de filas.
- **Concurrencia:** el servidor mantiene un solo modelo en RAM y serializa las generaciones; el segundo
  cliente espera en cola. Este comportamiento es deliberado para el uso local y evita duplicar el
  consumo de memoria.
