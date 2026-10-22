# Visualizador de activación neuronal

Herramienta de **interpretabilidad mecanicista**: mientras un transformer genera texto token a token,
captura la activación **real de las neuronas de la MLP** de cada capa y la pinta en vivo en un heatmap
que se envía al navegador por WebSocket.

No entrena ni modifica el modelo: solo *mira hacia dentro* con **TransformerLens** y muestra lo que ve.

## Cómo funciona

```
prompt ──▶ server.py (FastAPI)
             │  1. carga el modelo con TransformerBridge.boot_transformers
             │  2. prefill con run_with_cache; cada token nuevo reutiliza el KV cache
             │  3. guarda la activación del ÚLTIMO token (d_mlp floats)
             │  4. atención de la última query: top-8 conexiones por cabeza
             ▼  WebSocket /ws  →  {delta, activations, top, attention}
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

## Desarrollo

Instala las dependencias de desarrollo (incluye `pytest`) después de torch CPU:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements-dev.txt
pytest -q
```

Para comprobar el protocolo WebSocket contra un servidor en marcha:

```bash
# en otra terminal: uvicorn server:app --port 8000
python scripts/smoke_ws.py --model gpt2 --tokens 3
```

El smoke test valida que llegan `meta`, `step` (con todas las capas/neuronas) y `done`, que ningún
delta contiene `�` y que el stream acumulado coincide con `done.generated_text`.

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
| `WS` | `/ws` | canal de generación: recibe `{prompt, tokens, model, params}` y emite `meta`, `step` (delta de texto, activaciones, top y atención), `done` (texto completo, texto generado y motivo de parada), `queued` (otro cliente está generando) y `error` |

### Errores del WebSocket

El mensaje se valida con **Pydantic** antes de tocar el modelo. Si la petición no es válida, el servidor
responde y **mantiene la conexión abierta** para que se pueda corregir y reintentar:

```json
{"type": "error", "code": "invalid_request", "message": "Petición inválida — params.temperature: ...", "recoverable": true}
```

Un fallo interno (carga del modelo, OOM) envía `recoverable: false` y cierra con código **1011**. La UI
muestra el mensaje, deja el botón **Generar** habilitado y, tras la reconexión automática, permite
reintentar sin recargar la página.

### Texto incremental

Cada `step` trae `delta` con el texto nuevo y `token` como alias temporal del mismo valor. El servidor
decodifica el prefijo generado completo y recorta los caracteres de reemplazo de bytes UTF-8 incompletos,
así que un carácter multibyte partido entre dos tokens (`¿`, `¡`, acentos, emojis) no muestra `�`:

```json
{"type": "step", "delta": "¿", "token": "¿", "activations": {...}, "top": {...}, "attention": {...}}
```

Si `delta` es `""`, el token fue especial o solo aportó bytes que se completarán en el siguiente paso: la
UI cuenta el paso pero no añade texto al stream. Para el informe y las etiquetas de atención, el token
que completa un carácter recibe el texto y el token parcial queda vacío (la UI lo muestra como `␣`).

Al cerrar, `done.text` es la decodificación segura de **prompt + generación** y `done.generated_text` es
solo lo generado. La UI usa este último para reconciliar el stream si hiciera falta.

### Concurrencia

El modelo se comparte entre conexiones, pero las generaciones se serializan: si otro cliente está
generando, el nuevo recibe un mensaje `queued` y espera su turno en vez de recibir datos mezclados.
La carga también tiene su propio lock, así que dos clientes que piden el mismo modelo no lo cargan dos
veces. Las activaciones, la atención y el estado de decodificación son locales a cada generación; solo
el modelo cargado (`bridge`/`tokenizer`) es compartido.

### Rendimiento

- El forward de calentamiento que calcula `meta` también produce los logits del primer token; se
  reutilizan y no se repite la pasada completa por el prompt.
- `run_with_cache(..., pos_slice=-1)` guarda activaciones y atención solo de la última query. Con GPT-2
  y un prompt de 401 tokens, el pico de RSS baja aproximadamente 50 MiB respecto de cachear todas las
  posiciones.
- **KV cache:** el prefill procesa el prompt completo y cada token siguiente reutiliza las claves/valores
  (`past_key_values`), así que el modelo ya no recalcula todo el contexto. En GPT-2 con 512 tokens de
  contexto medimos ~0,158 s por token con forward completo y ~0,012 s con KV cache (~13×). El flag
  `VISUALIZER_KV_CACHE=0` desactiva la ruta incremental; si un modelo no devuelve caché, el servidor
  vuelve automáticamente al forward completo.

## Estructura

| Archivo | Rol |
|---|---|
| `server.py` | Backend FastAPI: carga del modelo, generación, captura de activaciones y WebSocket |
| `index.html` | Frontend autocontenido (HTML+CSS+JS, canvas puro, sin dependencias) |
| `requirements.txt` | Dependencias fijadas |
| `requirements-dev.txt` | Dependencias de desarrollo (pytest) |
| `scripts/smoke_ws.py` | Prueba de humo del WebSocket (`meta`/`step`/`done`) |
| `tests/` | Tests unitarios de muestreo, truncado, decodificación incremental, validación de mensajes y bucle de generación con KV cache |
| `plan-mejoras.md` | Plan incremental de mejoras y prioridades |
| `Dockerfile` / `docker-compose.yml` | Empaquetado con volumen para la caché de modelos |
| `NEXT_STEPS.md` | Roadmap (Docker, TransformerLens, nnsight) y mejoras pendientes |

## Limitaciones conocidas

- **Generaciones en serie:** el modelo es compartido y solo una generación corre a la vez; un segundo
  cliente recibe `queued` y espera. Las activaciones y la atención ya son locales y no se mezclan; para
  paralelismo real habría que aislar también el bridge/caché por conexión.
- **Escala global:** un valor atípico en una capa puede aplanar el resto (no hay normalización por capa).
- **Atención:** se envían las **8 conexiones más fuertes** por cabeza (no la matriz completa) para
  mantener ligero el payload. La ablación interactiva sigue en el roadmap.
- **Prompts largos:** el prompt se recorta a sus **últimos 512 tokens** para que los patrones de
  atención no consuman demasiada RAM.
