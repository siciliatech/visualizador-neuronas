# Visualizador de activación neuronal

Herramienta de **interpretabilidad mecanicista**: mientras un transformer genera texto token a token,
captura la activación **real de las neuronas de la MLP** de cada capa y la pinta en vivo en un heatmap
que se envía al navegador por WebSocket.

No entrena ni modifica el modelo: solo *mira hacia dentro* con **TransformerLens** y muestra lo que ve.

## Cómo funciona

```
prompt ──▶ server.py (FastAPI)
             │  1. carga el modelo con TransformerBridge.boot_transformers
             │  2. por cada token: run_with_cache(mlp.hook_post + attn.hook_pattern)
             │  3. guarda la activación del ÚLTIMO token (d_mlp floats)
             │  4. atención de la última query: top-8 conexiones por cabeza
             ▼  WebSocket /ws  →  {token, activations, top, attention}
        index.html (canvas)
             │  overview: Top-16 neuronas por capa (clic para seleccionar capa)
             │  heatmap:  todas las neuronas de la capa seleccionada
             │  atención: conexiones reales por cabeza + mapa de cabezas
             └  top-k / detalle de neurona / stream de texto
```

La neurona capturada es la **post-activación de la MLP** (antes de `down_proj`). En arquitecturas con
compuerta (Qwen/LLaMA/Gemma) ese hook ya es `act(gate_proj(x)) * up_proj(x)`, es decir la **neurona
completa**, no solo la compuerta.

## Modelos incluidos

| Clave | Modelo | Uso |
|---|---|---|
| `qwen2.5-0.5b-instruct` (por defecto) | Qwen2.5-0.5B-Instruct | responde preguntas (chat template) |
| `gpt2` | GPT-2 base (124M) | continúa texto |
| `pythia-410m` | EleutherAI Pythia-410m | continúa texto |

Se carga **un solo modelo en memoria** a la vez: al cambiar de modelo se descarga el anterior para no
agotar la RAM. La primera vez que se usa un modelo, Hugging Face lo descarga a la caché local.

## Requisitos

- Python 3.10+ (el `Dockerfile` usa 3.14).
- ~2 GB de RAM libres (el modelo por defecto, Qwen2.5-0.5B, ocupa ~2 GB en `float32`; GPT-2 ~0,5 GB).
- ~1 GB de disco por modelo descargado (Qwen ~1 GB, GPT-2 ~550 MB).
- No requiere GPU (corre en CPU con `float32`).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `torch` desde el índice CPU reduce mucho el tamaño de la instalación (evita las ruedas con CUDA):
> `pip install --index-url https://download.pytorch.org/whl/cpu torch`

## Arranque

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Abre <http://localhost:8000>. La primera generación tardará lo que tarde en descargar el modelo.

### Con Docker

```bash
docker compose up --build      # la primera vez descarga el modelo dentro del volumen
# abrir http://localhost:8000
docker compose down            # los modelos quedan en el volumen hf-models
docker compose down -v         # borra también los modelos
```

## Uso

1. Elige el **modelo** en el desplegable.
2. Escribe un **prompt** y el número de **tokens** a generar (1–100).
3. Ajusta el muestreo: `mode` (muestreo/greedy), `temperature`, `top_k` y `repetition_penalty`.
4. Pulsa **Generar**.
5. En el heatmap, **clic en una fila** del overview (izquierda) para seleccionar la capa mostrada;
   pasa el ratón por el heatmap para ver capa/neurona/valor exactos.
6. Haz **clic en una neurona** del heatmap o del ranking para ver su valor, su posición y su
   **historial de activación token a token**. El panel «¿Qué estoy viendo?» resume la interpretación.
7. En el panel de **atención**, elige una cabeza (`c0`…`c13`) o haz clic en su fila del mapa para ver
   sus conexiones más fuertes con los tokens anteriores.
8. Al terminar, el **informe** explica en lenguaje claro qué pasó: capa y neurona récord, atención más
   fuerte, muestreo usado y por qué terminó la generación (límite de tokens o EOS).

Junto al botón **Generar** hay un badge con el estado del modelo: `sin cargar`, `cargando …`
(descarga/carga los pesos la primera vez) y `listo`. El mensaje bajo los controles da el detalle.

La escala de color es **global al paso** (0 → máximo absoluto del paso, mostrado bajo el heatmap), con
los valores negativos en azul. La leyenda indica el valor `0` y el máximo.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | sirve `index.html` |
| `GET` | `/models` | catálogo de modelos disponibles (JSON) |
| `GET` | `/health` | sonda de liveness para Docker (`{"status": "ok"}`) |
| `WS` | `/ws` | canal de generación: recibe `{prompt, tokens, model, params}` y emite `meta`, `step` (activaciones, top y atención) y `done` (texto y motivo de parada) |

## Estructura

| Archivo | Rol |
|---|---|
| `server.py` | Backend FastAPI: carga del modelo, generación, captura de activaciones y WebSocket |
| `index.html` | Frontend autocontenido (HTML+CSS+JS, canvas puro, sin dependencias) |
| `requirements.txt` | Dependencias fijadas |
| `Dockerfile` / `docker-compose.yml` | Empaquetado con volumen para la caché de modelos |
| `NEXT_STEPS.md` | Roadmap (Docker, TransformerLens, nnsight) y mejoras pendientes |

## Limitaciones conocidas

- **Un cliente a la vez:** las activaciones viven en estado global; varias conexiones se pisan. Para
  concurrencia habría que aislar hooks por conexión.
- **Escala global:** un valor atípico en una capa puede aplanar el resto (no hay normalización por capa).
- **Atención:** se envían las **8 conexiones más fuertes** por cabeza (no la matriz completa) para
  mantener ligero el payload. La ablación interactiva sigue en el roadmap.
- **Prompts largos:** el prompt se recorta a sus **últimos 512 tokens** para que los patrones de
  atención no consuman demasiada RAM.
