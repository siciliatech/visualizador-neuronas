# Visualizador de activación neuronal

Herramienta de **interpretabilidad mecanicista**: mientras un transformer genera texto token a token,
captura la activación **real de las neuronas de la MLP** de cada capa y la pinta en vivo en un heatmap
que se envía al navegador por WebSocket.

No entrena ni modifica el modelo: solo *mira hacia dentro* con **TransformerLens** y muestra lo que ve.

## Cómo funciona

```
prompt ──▶ server.py (FastAPI)
             │  1. carga el modelo con TransformerBridge.boot_transformers
             │  2. por cada token: run_with_cache(blocks.{i}.mlp.hook_post)
             │  3. guarda la activación del ÚLTIMO token (d_mlp floats)
             ▼  WebSocket /ws  →  {token, activations, top}
        index.html (canvas)
             │  overview: Top-16 neuronas por capa (clic para seleccionar capa)
             │  heatmap:  todas las neuronas de la capa seleccionada
             └  top-k:   ranking de la capa seleccionada + stream de texto
```

La neurona capturada es la **post-activación de la MLP** (antes de `down_proj`). En arquitecturas con
compuerta (Qwen/LLaMA/Gemma) ese hook ya es `act(gate_proj(x)) * up_proj(x)`, es decir la **neurona
completa**, no solo la compuerta.

## Modelos incluidos

| Clave | Modelo | Uso |
|---|---|---|
| `gpt2` (por defecto) | GPT-2 base (124M) | continúa texto |
| `qwen2.5-0.5b-instruct` | Qwen2.5-0.5B-Instruct | responde preguntas (chat template) |
| `pythia-410m` | EleutherAI Pythia-410m | continúa texto |

Se carga **un solo modelo en memoria** a la vez: al cambiar de modelo se descarga el anterior para no
agotar la RAM. La primera vez que se usa un modelo, Hugging Face lo descarga a la caché local.

## Requisitos

- Python 3.10+ (el `Dockerfile` usa 3.14).
- ~2 GB de RAM libres para GPT-2, ~1 GB de disco por modelo descargado.
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

La escala de color es **global al paso** (0 → máximo absoluto del paso, mostrado bajo el heatmap), con
los valores negativos en azul. La leyenda indica el valor `0` y el máximo.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | sirve `index.html` |
| `GET` | `/models` | catálogo de modelos disponibles (JSON) |
| `WS` | `/ws` | canal de generación: recibe `{prompt, tokens, model, params}` y emite `meta`, `step`, `done` |

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
- Solo se visualiza la **MLP**; la atención y la ablación interactiva están en el roadmap
  (ver `NEXT_STEPS.md`).
