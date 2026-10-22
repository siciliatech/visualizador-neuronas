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
             ▼  WebSocket /ws  →  {delta, layer, activations (capa visible), stats, top, attention}
        index.html (canvas)
             │  overview: Top-16 neuronas por activación con signo (clic para seleccionar capa)
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
agotar la RAM. La primera vez que se usa un modelo, Hugging Face lo descarga a la caché local; según la
conexión puede tardar varios minutos. El servidor avisa con un mensaje `loading` y la UI muestra
`cargando …` mientras tanto.

## Requisitos

- Python 3.10+ (el `Dockerfile` usa 3.14).
- **RAM:** ~2 GB libres para GPT-2 y ~4 GB para Qwen2.5-0.5B. Medido en el proceso servidor:
  - GPT-2: ~1,2 GB de RSS tras cargar y ~1,4 GB de pico con un prompt de 512 tokens.
  - Qwen2.5-0.5B: ~2,3 GB tras cargar y ~2,7 GB de pico con 512 tokens.
  - El `docker-compose.yml` limita el contenedor a **6 GB** para dejar margen al modelo y al resto del proceso.
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

Instala las dependencias de desarrollo (incluye `pytest` y `ruff`) después de torch CPU:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements-dev.txt

ruff check .
ruff format --check .
pytest -q                 # unitarios; excluye los marcados como slow
pytest -q -m slow         # integración real con GPT-2 (descarga/carga el modelo)
```

La configuración de Ruff y pytest vive en `pyproject.toml`. El test `slow` carga GPT-2 de verdad y
comprueba el forward y la reutilización del KV cache; no se ejecuta en cada push ni en la suite normal.

Para comprobar el protocolo WebSocket contra un servidor en marcha:

```bash
# en otra terminal: uvicorn server:app --port 8000
python scripts/smoke_ws.py --model gpt2 --tokens 3
```

El smoke test valida el protocolo 2: `meta` con `protocol`/`selected_layer`, `step` con una sola capa
(la seleccionada) o todas con `--capture-all`, `stats` globales, `top` completo y `done`; también
comprueba que ningún delta contiene `�` y que el stream acumulado coincide con `done.generated_text`. Con
`--layer N` prueba además el cambio de capa en vivo:

```bash
python scripts/smoke_ws.py --model gpt2 --tokens 3 --layer 5
```

Con `--capture-all` el script comprueba además que cada `step` incluya **todas las capas** (el modo que
usa la exportación CSV):

```bash
python scripts/smoke_ws.py --model gpt2 --tokens 2 --capture-all
```

## Uso

1. Elige el **modelo** en el desplegable.
2. Escribe un **prompt** y el número de **tokens** a generar (1–100).
3. Ajusta el muestreo: `mode` (muestreo/greedy), `temperature`, `top_k`, `top_p`,
   `repetition_penalty` y `no_repeat_ngram`. Si vas a exportar el CSV de varias capas, marca
   **todas las capas** en «Exportar» antes de generar.
4. Pulsa **Generar**.
5. En el heatmap, **clic en una fila** del overview (izquierda) para seleccionar la capa mostrada;
   pasa el ratón por el heatmap para ver capa/neurona/valor exactos.
6. Haz **clic en una neurona** del heatmap o del ranking para ver su valor, su posición y su
   **historial de activación token a token**. El panel «¿Qué estoy viendo?» resume la interpretación.
7. En el panel de **atención**, elige una cabeza (`c0`…`c13`) o haz clic en su fila del mapa para ver
   sus conexiones más fuertes con los tokens anteriores.
8. Al terminar, el **informe** explica en lenguaje claro qué pasó: capa y neurona con mayor
   |activación| (magnitud y signo), capa con mayor media de |activación|, atención más fuerte dentro
   del top-8 enviado, **todos** los parámetros de muestreo y por qué terminó la generación.
9. Si activaste **todas las capas**, selecciona una capa y pulsa **CSV capa** para descargar sus
   activaciones de toda la generación.

Los valores iniciales del muestreo son `temperature=0.8`, `top_k=40`, `top_p=0.95`,
`repetition_penalty=1.15` y `no_repeat_ngram=0`. `top_p=1` desactiva el filtro nucleus;
`no_repeat_ngram=0` (o `1`) desactiva la prohibición de repetir n-gramas y `N≥2` la activa.

Junto al botón **Generar** hay un badge con el estado del modelo: `sin cargar`, `cargando …`
(descarga/carga los pesos la primera vez) y `listo`. El mensaje bajo los controles da el detalle.

La escala de color es **global al paso** (0 → máximo absoluto del paso, mostrado bajo el heatmap), con
los valores negativos en azul. La leyenda indica el valor `0` y el máximo.

El overview y el ranking de neuronas ordenan por el **valor de activación con signo** (de mayor a
menor); el ancho de las barras usa `|v|` solo como escala visual. El informe, en cambio, busca la mayor
**magnitud** `|activación|`, así que puede destacar una neurona muy negativa que no encabeza el ranking.

En el mapa de atención solo se dibujan las **8 conexiones más fuertes por cabeza**. Las celdas que
quedan fuera de ese top-8 llevan un punto tenue: no son atención cero y el tooltip lo indica como
«fuera del top-8». Los tokens especiales de los modelos chat (`<|im_start|>`, `<|endoftext|>`, etc.) se
etiquetan con su nombre en lugar de mostrarse como un espacio.

Los canvas se dibujan a la resolución de `devicePixelRatio` (limitada a **2×** para no disparar el coste
en pantallas 3×) y la animación se limita a **~30 fps**. Así la atención no se redibuja en cada frame y
la vista se mantiene nítida en pantallas retina sin subir el uso de CPU.

### Exportar activaciones a CSV

La generación normal envía solo la capa visible para mantener el payload pequeño. Para exportar cualquier
capa, marca **todas las capas** antes de generar: el servidor incluirá todas las activaciones en cada
`step`, el navegador las guarda en memoria y, al terminar, el botón **CSV capa** descarga la capa que
estés viendo. Para otra capa, selecciónala en el overview y vuelve a descargar.

El CSV tiene una fila por neurona y token generado:

```csv
step,token,layer,neuron,activation
1," la",0,0,0.123
1," la",0,1,-0.045
```

El nombre incluye modelo, capa y fecha. La captura vive solo en la pestaña hasta que generes de nuevo o
la cierres; no se escribe nada en disco en el servidor. Activar todas las capas aumenta el payload de
cada `step` (y la memoria del navegador), así que conviene usarlo solo cuando se vaya a exportar.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | sirve `index.html` |
| `GET` | `/models` | catálogo de modelos disponibles (JSON) |
| `GET` | `/health` | estado del proceso sin cargar ni descargar modelos: `{"status": "ok", "model_key": "gpt2", "ready": true}`; si no hay modelo, `model_key` es `null` y `ready` es `false` |
| `WS` | `/ws` | canal de generación: recibe `{prompt, tokens, model, params, capture_all}` y `{type: "select_layer", layer: N}`; emite `meta` (protocolo 2, capa seleccionada y `capture_all`), `step` (delta, capa visible o todas las capas si se pidió captura, `stats` globales, `top` y atención), `layer` (respuesta al cambio de capa), `done` (texto completo, texto generado y motivo de parada), `queued` y `error` |

### Logs

El servidor usa `logging` (no `print`) y emite, con nivel `INFO` por defecto:

- tiempo de carga del modelo y número de capas/neuronas;
- cada forward con contexto, tokens de entrada, modo de caché (`prefill`, `reuse` u `off`) y duración;
- cada `step` con capa, longitud del contexto, delta y **tamaño real del payload** en bytes;
- el cierre de la generación con motivo y duración;
- los errores internos con traza completa (`logger.exception`).

`VISUALIZER_LOG_LEVEL=DEBUG` (o `WARNING`) ajusta el nivel. Los logs de descarga de Hugging Face se
mantienen en `WARNING` para no tapar los de la generación.

### Errores del WebSocket

El mensaje se valida con **Pydantic** antes de tocar el modelo. Si la petición no es válida, el servidor
responde y **mantiene la conexión abierta** para que se pueda corregir y reintentar:

```json
{"type": "error", "code": "invalid_request", "message": "Petición inválida — params.temperature: ...", "recoverable": true}
```

Un fallo interno (carga del modelo, OOM) envía `recoverable: false` y cierra con código **1011**. La UI
muestra el mensaje, deja el botón **Generar** habilitado y, tras la reconexión automática, permite
reintentar sin recargar la página.

### Capa visible y payload (protocolo 2)

El heatmap muestra **una sola capa**, así que el `step` ya no necesita enviar todas. `meta` declara
`"protocol": 2` y la capa inicial (`selected_layer`). El cliente pide otra capa con:

```json
{"type": "select_layer", "layer": 5}
```

El servidor guarda la selección **por conexión** y responde con el último estado completo que conserva
en memoria, sin repetir el forward:

```json
{"type": "layer", "layer": 5, "activations": [0.12, -0.03, "..."]}
```

Los `step` siguientes traen solo `activations["layer_N"]`, el campo `"layer": N` y `stats` agregados de
**todas** las capas para que el informe conserve los récords globales:

```json
"stats": {
  "max_abs": {"value": 12.3, "layer": 5, "neuron": 912, "token": "..."},
  "max_signed": {"value": -12.3, "layer": 5, "neuron": 912, "token": "..."},
  "layer_means": {"layer_0": 0.42, "layer_5": 1.08}
}
```

Si el cliente envía `"capture_all": true`, `activations` incluye todas las capas para permitir la
exportación CSV; `meta` repite `capture_all` para confirmar el modo.

`top` y `attention` siguen siendo completos: el overview y la vista de atención no cambian al cambiar de
capa. El clic en una fila del overview envía `select_layer`; mientras no llega la respuesta, el heatmap
mantiene la capa anterior en vez de quedar vacío. El frontend también acepta `step` del protocolo
anterior (todas las capas y sin `stats`) como red de seguridad durante la transición.

### Texto incremental

Cada `step` trae `delta` con el texto nuevo y `token` como alias temporal del mismo valor. El servidor
decodifica el prefijo generado completo y recorta los caracteres de reemplazo de bytes UTF-8 incompletos,
así que un carácter multibyte partido entre dos tokens (`¿`, `¡`, acentos, emojis) no muestra `�`:

```json
{"type": "step", "layer": 5, "delta": "¿", "token": "¿", "activations": {"layer_5": [...]}, "stats": {...}, "top": {...}, "attention": {...}}
```

Si `delta` es `""`, el token fue especial o solo aportó bytes que se completarán en el siguiente paso: la
UI cuenta el paso pero no añade texto al stream. Para el informe y las etiquetas de atención, el token
que completa un carácter recibe el texto y el token parcial queda vacío (la UI lo muestra como `␣`).
Los tokens especiales se etiquetan con su nombre real (`<|im_start|>`, `<|endoftext|>`, etc.).

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
- **Payload por capa:** cada `step` bajó de tamaño al enviar solo la capa visible. Medido con el smoke
  test (incluyendo `top` y `attention` completos): GPT-2 ~32–36 KiB por token (antes ~258 KiB, ~8× menos)
  y Qwen2.5-0.5B ~67 KiB (antes ~0,8 MB, ~12× menos). El cambio de capa no reenvía activaciones: usa el
  último estado que ya está en el servidor.
- **Canvas:** el heatmap y la animación se redibujan como máximo a ~30 fps; la atención solo se redibuja
  cuando cambia (token, capa o cabeza), no en cada frame. Los canvas usan `devicePixelRatio` (máximo 2×)
  manteniendo el mismo tamaño CSS, así que se ven nítidos en retina sin multiplicar el coste por 3.

## Estructura

| Archivo | Rol |
|---|---|
| `server.py` | Backend FastAPI: carga del modelo, generación, captura de activaciones y WebSocket |
| `index.html` | Frontend autocontenido (HTML+CSS+JS, canvas puro, sin dependencias) |
| `requirements.txt` | Dependencias fijadas |
| `requirements-dev.txt` | Dependencias de desarrollo (pytest y ruff) |
| `pyproject.toml` | Configuración de Ruff y pytest (incluye el marcador `slow`) |
| `scripts/smoke_ws.py` | Prueba de humo del WebSocket (`meta`/`layer`/`step`/`done`) |
| `tests/` | Tests unitarios de muestreo, truncado, decodificación incremental y tokens especiales, validación de mensajes, bucle de generación con KV cache, cambio de capa por WebSocket, `/health` y tamaño de payload; `test_integration_slow.py` carga GPT-2 real |
| `.github/workflows/ci.yml` | CI con Python 3.12, caché de Hugging Face, Ruff y `pytest -m "not slow"` |
| `plan-mejoras.md` | Plan incremental de mejoras y prioridades |
| `Dockerfile` / `docker-compose.yml` | Empaquetado con volumen para la caché de modelos |
| `NEXT_STEPS.md` | Roadmap (Docker, TransformerLens, nnsight) y mejoras pendientes |

## Limitaciones conocidas

- **Generaciones en serie:** el modelo es compartido y solo una generación corre a la vez; un segundo
  cliente recibe `queued` y espera. Las activaciones y la atención ya son locales y no se mezclan; para
  paralelismo real habría que aislar también el bridge/caché por conexión.
- **Escala global:** un valor atípico en una capa puede aplanar el resto (no hay normalización por capa).
- **Atención:** se envían las **8 conexiones más fuertes** por cabeza (no la matriz completa) para
  mantener ligero el payload; el resto se marca con un punto y no significa atención cero. La ablación
  interactiva sigue en el roadmap.
- **Prompts largos:** el prompt se recorta a sus **últimos 512 tokens** para que los patrones de
  atención no consuman demasiada RAM.
- **Exportación completa:** activar **todas las capas** aumenta el payload de cada `step` y la memoria
  del navegador; el CSV se descarga de una capa a la vez. En modo normal no se paga ese coste.
