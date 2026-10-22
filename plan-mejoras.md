# Plan de mejoras

> Estado de referencia: commit `867ef2f`. Los números de línea pueden moverse; usa los nombres de función como ancla.
> Objetivo: resolver los hallazgos del análisis en **incrementos pequeños, verificables y reversibles**, sin dejar el proyecto roto entre pasos.

## Cómo usar este plan

- **Un PR = un incremento.** No mezclar arreglos de fiabilidad con cambios de protocolo o refactors.
- Cada incremento debe dejar `main` funcionando y la UI usable.
- Si el cambio es de rendimiento, **medir antes y después** con la tabla de métricas base.
- Si cambia el protocolo WebSocket o el comportamiento visible, actualizar `README.md` y `NEXT_STEPS.md` en el mismo PR.
- Los tamaños son orientativos: `S` < 1 h, `M` 1–3 h, `L` 3–6 h.

## Hallazgos a cubrir

| ID | Hallazgo | Severidad | Dónde |
|---|---|---|---|
| F01 | Excepción en `/ws` cierra el socket sin mensaje | P0 | `server.py` → `websocket_endpoint` |
| F02 | Decodificación token a token corrompe UTF-8 (`�`) | P0 | `server.py` → `forward_once` / bucle de generación |
| F03 | Estado global y carga de modelo sin lock | P0 | `server.py` → `_current`, `load_model` |
| F14 | El badge queda en “generando…” al terminar | P0 | `index.html` → handler `done` |
| F17 | Mensajes WS sin validación de esquema | P1 | `server.py` → `websocket_endpoint` |
| F05 | Caché de atención O(T²) innecesario | P1 | `server.py` → `forward_once` |
| F04 | No se usa KV cache: se reprocesa todo el contexto | P1 | `server.py` → bucle de generación |
| F06 | Payload de ~0,8 MB/token en Qwen | P1 | `server.py` → `step` |
| F07 | Docker instala torch sin fijar versión | P1 | `Dockerfile` |
| F08 | RAM documentada por debajo de la real | P1 | `README.md` |
| F12 | Canvas redibuja a ~60 fps durante la generación | P2 | `index.html` → `frame` / `drawHeatmap` |
| F13 | Canvas sin `devicePixelRatio` | P2 | `index.html` → `layout*` |
| F09 | `top_p` y `no_repeat_ngram` ocultos | P2 | `server.py` / `index.html` |
| F10 | Semántica ambigua del informe | P2 | `index.html` → `applyStep` / `renderReport` |
| F11 | La atención top-8 se muestra como si el resto fuera cero | P2 | `index.html` → `drawAttnMap` |
| F15 | `FileResponse("index.html")` depende del CWD | P2 | `server.py` → `index` |
| F16 | EOS como lista no contemplado | P2 | `server.py` → bucle de generación |
| F18 | Sin tests ni CI | P2 | repo |

## Métricas base (para comparar)

Medidas en un venv aislado con **GPT-2** (modelo cacheado), commit `867ef2f`:

| Métrica | Valor actual |
|---|---|
| Payload de un `step` (GPT-2, 12×3072) | **258,6 KiB** |
| Payload estimado de un `step` (Qwen, 24×4864) | **~0,8 MB** |
| Forward completo con 512 tokens (GPT-2) | **0,20 s** |
| Decode incremental con KV cache (GPT-2, contexto corto) | **0,01 s** |
| Pico de RSS con prompt de 512 tokens (GPT-2) | **~1,39 GB** |
| RSS tras liberar caché (GPT-2) | **~747 MB** |
| Carga en caliente de GPT-2 | **~6 s** |
| Primera carga (red HF lenta) | **~239 s** (una vez) |

> Qwen/Pythia no se ejecutaron en la validación; sus valores son calculados a partir de `config.json`.

---

# Fase 0 · Red de seguridad (recomendada, 30–45 min)

No cambia el producto, pero hace mucho más seguros los cambios de las fases siguientes.

### 0.1 · Script de humo por WebSocket  `S`

**Archivos:** `scripts/smoke_ws.py` (nuevo).

**Cambios:**
- Cliente mínimo que conecta a `/ws`, envía un `generate` con `gpt2`, 3 tokens y `mode=greedy`.
- Imprime `meta`, el tamaño de cada `step`, los tokens recibidos y `done`.
- Código de salida ≠ 0 si no llega `done` o si un `step` no trae todas las capas.
  - Nota: cuando se implemente 2.3 (payload por capa), ajustar el criterio a “trae la capa seleccionada y el `top` completo”.

**Aceptación:**
- `python scripts/smoke_ws.py` funciona contra un servidor local.
- Sirve como prueba manual repetible después de cada PR.

### 0.2 · Tests de las funciones puras  `S`

**Archivos:** `tests/test_sampling.py`, `tests/test_inputs.py`, `pytest.ini` o `pyproject.toml`.

**Cambios:**
- Tests de `ban_repeated_ngrams` y `sample_next_token` con logits sintéticos: greedy, temperatura 0, top-k, top-p, penalización por repetición, n-gramas.
- Test de truncado a `MAX_INPUT_TOKENS` con un tokenizer falso (sin descargar modelos).
- Añadir `pytest` a las dependencias de desarrollo (no a `requirements.txt` de producción).

**Aceptación:**
- `pytest -q` pasa sin descargar modelos.
- Los tests cubren los casos límite que hoy no tienen red: todos los logits en `-inf`, `top_k=0`, `repetition_penalty=1`.

---

# Fase 1 · Fiabilidad (P0)

### 1.1 · Reset de estado y badge en el cliente  `S`

**Archivos:** `index.html`.

**Cambios:**
- En `done`: `setBadge('ready', 'listo')` (hoy solo se actualiza `status`).
- En `onclose` y `onerror`: `running = false`, ocultar `caret`, deshabilitar el botón y mostrar “desconectado”.
- Al reconectar (`onopen`): no asumir que el modelo está descargado; mostrar `sin cargar` solo si no hubo `meta` en la sesión, o mantener un estado `lastModelReady`.

**Aceptación:**
- Al terminar una generación el badge dice “listo”, no “generando…”.
- Si se mata el servidor a mitad de generación y vuelve, el botón **Generar** funciona en el primer clic.

**Prueba:** generar, matar `uvicorn`, esperar la reconexión, generar de nuevo.

**Riesgo:** bajo.

---

### 1.2 · Mensaje de error del servidor  `M`

**Archivos:** `server.py` (`websocket_endpoint`), `index.html` (handler nuevo).

**Cambios:**
- Envolver el procesamiento de cada mensaje en `try/except Exception`.
- Enviar `{"type": "error", "code": "...", "message": "..."}` antes de cerrar.
- Distinguir dos casos:
  - **Error de entrada** (parámetros inválidos): mantener la conexión abierta para que el usuario corrija.
  - **Error interno** (fallo del modelo, OOM): enviar el error y cerrar con código 1011.
- Loguear la traza completa solo en el servidor; nunca enviar el stack al cliente.
- En el frontend: mostrar `message` en `status`, poner el badge en `error`, `running = false`, ocultar el caret y habilitar el botón.

**Aceptación:**
- Un `temperature: "abc"` produce un mensaje de error legible, sin traza ASGI en la consola del cliente.
- Tras el error, el usuario puede volver a pulsar **Generar** sin recargar la página.
- Un fallo al cargar un modelo inexistente también muestra un error claro.

**Prueba:** enviar el payload inválido que hoy cierra el socket; comprobar que la UI se recupera.

**Riesgo:** bajo.

---

### 1.3 · Validación del mensaje con Pydantic  `M`

**Depende de:** 1.2.

**Archivos:** `server.py`.

**Cambios:**
- Definir modelos:
  - `SamplingParams`: `mode` (`sample`/`greedy`), `temperature` (0–5), `top_k` (0–vocab), `top_p` (0–1), `repetition_penalty` (0.1–5), `no_repeat_ngram` (0–10).
  - `GenerateRequest`: `prompt` (str, longitud máxima razonable), `tokens` (int, 1–100), `model` (clave válida), `params`.
- Validar con `GenerateRequest.model_validate(msg)` y responder `error` con el detalle del campo.
- Aplicar los valores ya validados en el bucle; eliminar los `float(...)` sueltos que pueden lanzar `ValueError`.

**Aceptación:**
- Mensajes válidos se comportan exactamente igual que antes.
- Mensajes inválidos devuelven un error de validación claro y no llegan a tocar el modelo.

**Riesgo:** bajo; mantener los valores por defecto actuales para no cambiar el comportamiento.

---

### 1.4 · Decodificación incremental (arreglo de `�`)  `M`

**Archivos:** `server.py`, `index.html` (solo si hace falta).

**Cambios:**
- En lugar de `tokenizer.decode(next_token_id, ...)`, mantener el texto generado y calcular el delta:
  ```python
  # antes del bucle: prev_text = tokenizer.decode(seen_ids, skip_special_tokens=True)
  full_text = tokenizer.decode(seen_ids, skip_special_tokens=True)
  delta = full_text[len(prev_text):]
  prev_text = full_text
  ```
- Enviar `delta` en el `step` y mantener `token` como alias durante la transición:
  ```json
  { "type": "step", "delta": "ción", "token": "ción", ... }
  ```
  El frontend usa `data.delta ?? data.token`.
- Si `delta == ""` (token especial), no añadir nada al stream, pero sí contar el paso.
- Para las etiquetas de atención: construir `attention.tokens` con un helper que asigne el delta al token que completa el carácter y deje vacío el token parcial; la UI ya convierte vacío en `␣`.
- En `done`, comparar `stream.textContent` con `done.text` (para modelos base, descontando el prompt) y corregir si difieren.

**Aceptación:**
- Un prompt/generación con `¿`, `¡`, acentos partidos y un emoji no muestra `�` en el stream.
- El texto del stream coincide con `done.text` (salvo el eco del prompt).
- El informe sigue mostrando el token que corresponda.

**Prueba:** usar el tokenizer de GPT-2 para verificar `['ĠÂ', '¿']`; generar una respuesta en español y revisar el stream.

**Riesgo:** medio-bajo; el cambio es local a la decodificación.

---

### 1.5 · Lock de carga y generación  `M`

**Archivos:** `server.py`.

**Cambios:**
- Añadir `load_lock = asyncio.Lock()` y `generation_lock = asyncio.Lock()`.
- Dentro de `load_lock`: volver a comprobar `_current["key"]` antes de cargar; así dos clientes no cargan el mismo modelo dos veces.
- Serializar la generación completa con `generation_lock` (o usar una cola con un mensaje `{type:"queued"}` si hay otro cliente).
- Mantener el modelo compartido (no hace falta un bridge por conexión) pero **no** compartir `activations`/`attention`: pasarlas como estado local de la generación y enviarlas directamente, sin escribirlas en `_current`.
- Si se prefiere el cambio mínimo: mantener `_current` pero serializar; documentar que sigue siendo un cliente a la vez.

**Aceptación:**
- Dos clientes simultáneos no comparten activaciones ni atención.
- El modelo se carga una sola vez (verificar el log).
- Un segundo cliente recibe una respuesta clara (espera o “ocupado”), no datos mezclados.

**Riesgo:** medio; toca el corazón del servidor. Hacerlo después de 1.2/1.4 y con el smoke test.

---

# Fase 2 · Rendimiento (P1)

### 2.1 · `pos_slice=-1` y eliminar el forward duplicado  `S`

**Archivos:** `server.py` (`forward_once`, `websocket_endpoint`).

**Cambios:**
- Añadir `pos_slice=-1` a `bridge.run_with_cache(...)`. Devuelve `[1, 1, d_mlp]` y `[1, heads, 1, seq]`, compatibles con el slicing actual.
- Reutilizar el forward de calentamiento: hoy se hace uno antes de `meta` y otro para el primer token. Guardar esos `logits` y usarlos para muestrear el primer token.
- Mantener `meta` calculado a partir del calentamiento.

**Aceptación:**
- El primer token no requiere un segundo forward (contar llamadas en el log).
- Los valores del primer `step` son idénticos a los de antes.
- El pico de RSS baja (medir con la tabla base).

**Riesgo:** bajo; una línea + una reorganización pequeña.

---

### 2.2 · KV cache para decodificación incremental  `L`

**Depende de:** 2.1.

**Archivos:** `server.py`.

**Cambios:**
- En el prefill:
  ```python
  (logits, past), cache = bridge.run_with_cache(
      ids, pos_slice=-1, return_type="logits_and_cache", use_cache=True, ...
  )
  ```
- En cada token siguiente:
  ```python
  (logits, past), cache = bridge.run_with_cache(
      torch.tensor([[next_id]]), past_key_values=past, pos_slice=-1, ...
  )
  ```
- Guardar `past` por generación y resetearlo al empezar un prompt o cambiar de modelo.
- Fallback: si `past is None` o el modelo no soporta `use_cache`, volver al forward completo (flag `USE_KV_CACHE`).
- Mantener `curr_ids`/`seen_ids` para `done.text` y para las etiquetas de atención; el forward ya no necesita el contexto completo.
- Cuidado con el offset de atención: en el prefill el patrón es `[1, heads, 1, T]` y en cada decode `[1, heads, 1, T+1]`. La lista `attention.tokens` debe incluir el token recién añadido y el top-k se indexa sobre el contexto completo.

**Aceptación:**
- Salida greedy idéntica a la implementación anterior para un prompt fijo (guardar un golden test).
- Activaciones del último token equivalentes dentro de `1e-5`.
- Mejora medible de tiempo por token con contexto de 512 (GPT-2: 0,20 s → ~0,01–0,02 s).
- Sin errores ni fugas tras 20 tokens.

**Riesgo:** alto; es el cambio más delicado. Hacerlo con el golden test de 2.1 y un flag para revertir.

---

### 2.3 · Reducir el payload del `step`  `M`

**Idea:** el heatmap solo muestra **una** capa; hoy se envían todas.

**Opción A (recomendada): capa seleccionada.**
- Cliente → servidor: `{"type": "select_layer", "layer": N}`.
- Servidor: guarda la capa seleccionada por conexión y envía en `step.activations` solo `layer_N`; añade `"layer": N`.
- Al cambiar de capa, el servidor responde con `{"type": "layer", "layer": N, "activations": [...]}` usando el último estado completo que conserva en memoria.
- **Importante para el informe:** hoy `applyStep` calcula los récords recorriendo todas las capas. Al enviar una sola, el servidor debe incluir en `step.stats` los agregados de todas las capas:
  ```json
  "stats": {
    "max_abs": { "value": 12.3, "layer": 5, "neuron": 912, "token": "..." },
    "max_signed": { "value": 12.3, "layer": 5, "neuron": 912, "token": "..." },
    "layer_means": { "layer_0": 0.42, "...": 0.0 }
  }
  ```
  El frontend usa `stats` para el informe y `activations` solo para el heatmap.
- El overview sigue usando `top` (ya viene completo).
- `index.html`: el clic del overview envía `select_layer`; `applyStep` usa `data.layer` para el heatmap.

**Opción B (si no se quiere tocar la selección):** cuantizar a int8 + escala y enviar base64. Reduce ~8×, pero sigue enviando todas las capas.

**Aceptación:**
- GPT-2: `step` de ~258 KiB → ~22 KiB (una capa).
- Qwen: ~0,8 MB → ~35 KiB.
- Cambiar de capa se siente inmediato y el heatmap no queda vacío.
- El informe sigue mostrando el récord global (capa/neurona) y la capa más activa, usando `stats` y no solo la capa enviada.
- La vista de overview y la atención no cambian.

**Riesgo:** medio; cambio de protocolo. Añadir `"protocol": 2` en `meta` y aceptar el formato antiguo durante la transición.

---

### 2.4 · Throttling del canvas y `devicePixelRatio`  `M`

**Archivos:** `index.html`.

**Cambios:**
- `frame()` solo redibuja si `needsDraw`; durante la animación, limitar a ~30 fps (o dibujar el heatmap base en un canvas offscreen y animar solo el glow).
- Separar “redibujar heatmap” de “redibujar animación”: el heatmap solo cambia en cada `step`, no en cada frame.
- Escalar los canvas por `window.devicePixelRatio` manteniendo el tamaño CSS.

**Aceptación:**
- El uso de CPU del navegador baja de forma visible durante una generación.
- No hay regresión visual; el canvas se ve nítido en retina.

**Riesgo:** bajo.

---

# Fase 3 · Empaquetado y documentación (P1)

### 3.1 · Fijar torch en Docker y corregir la RAM documentada  `S`

**Archivos:** `Dockerfile`, `README.md`, `requirements.txt` (comentario).

**Cambios:**
- `Dockerfile`: `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1`.
- `README.md`: actualizar los requisitos de RAM con las cifras medidas:
  - GPT-2: ~1,2 GB de RSS tras cargar; pico ~1,4 GB con 512 tokens.
  - Qwen: ~3 GB o más; recomendar 4–6 GB libres.
- Aclarar que el primer arranque puede tardar minutos por la descarga/comprobación de Hugging Face.
- Verificar que `docker compose build` no descarga ruedas CUDA.

**Aceptación:**
- El build usa la rueda CPU aunque exista una versión más nueva de torch.
- Las cifras del README coinciden con una medición reproducible.

**Riesgo:** bajo.

---

# Fase 4 · Consistencia de producto (P2)

### 4.1 · Exponer `top_p` y `no_repeat_ngram`  `S`

**Archivos:** `index.html`, `server.py`, `README.md`.

**Cambios:**
- Añadir controles para `top_p` (0–1) y `no_repeat_ngram` (0–10), o eliminar `no_repeat_ngram` del backend si no se va a usar.
- Enviar los valores en `params` y mostrarlos en el informe.
- Decidir el valor por defecto de `top_p` (hoy 0.95 aunque la UI no lo diga).

**Aceptación:**
- El informe refleja **todos** los parámetros que afectan al muestreo.
- Repetir una generación con los mismos parámetros produce el mismo comportamiento descrito.

---

### 4.2 · Unificar la semántica del informe  `S`

**Archivos:** `index.html` (`applyStep`, `renderReport`).

**Cambios:**
- Definir “activación más intensa” como **máximo valor absoluto** y mostrar el signo.
- Etiquetar la media como “media de |activación|” si se sigue calculando así.
- Aclarar que el récord de atención se calcula solo sobre el top-8 enviado.

**Aceptación:**
- Los números del informe coinciden con lo que muestra el heatmap y la escala.
- El texto no sugiere una media con signo cuando es un promedio de magnitudes.

---

### 4.3 · Hacer honesta la visualización de atención  `M`

**Archivos:** `index.html` (`drawAttnMap`, leyenda), `README.md`.

**Cambios:**
- Indicar explícitamente “top-8 por cabeza; el resto no se dibuja” en la leyenda.
- Opcional: dibujar una marca tenue en las columnas no incluidas o permitir ver la distribución completa de una cabeza seleccionada.
- Para modelos chat, etiquetar los tokens especiales (`<|im_start|>`, etc.) en lugar de dejarlos en blanco.

**Aceptación:**
- Un usuario no interpreta las celdas oscuras como “atención cero”.
- El tooltip distingue “fuera del top-8” de “0”.

---

### 4.4 · Detalles pequeños  `S`

- `dRank` usa `META.top_k` en lugar del 16 hardcodeado.
- Comparar EOS con un `set` (por si `eos_token_id` es lista).
- `FileResponse(Path(__file__).parent / "index.html")`.
- Revisar el texto “fuera del Top-16” y la noción de “neurona más activa” (signo vs. magnitud).
- Eliminar código muerto (`no_repeat_ngram` si no se expone).

---

# Fase 5 · Calidad continua (P2)

### 5.1 · Tests, lint y CI  `M`

**Archivos:** `tests/`, `.github/workflows/ci.yml`, `pyproject.toml` (ruff/pytest).

**Cambios:**
- Tests unitarios de sampling, truncado y decodificación incremental.
- Test de integración marcado como `slow` con un modelo pequeño soportado por TransformerLens; no ejecutarlo en cada push si tarda demasiado.
- `ruff check` y `ruff format --check`.
- CI con Python 3.12, caché de Hugging Face y `pytest -q -m "not slow"`.

**Aceptación:**
- CI verde en cada PR.
- Los tests detectan una regresión en `sample_next_token` y en el delta de decodificación.

---

### 5.2 · Logging y observabilidad  `S`

- Sustituir `print` por `logging`.
- Loguear: tiempo de carga del modelo, tiempo por forward, longitud del contexto, tamaño del payload y errores con traza.
- Añadir un `/health` que opcionalmente informe `model_key` y `ready` (sin descargar nada).

---

# Apéndice A · Protocolo WebSocket objetivo

Mantener compatibilidad durante la transición con un campo `"protocol": 2` en `meta`.

**Cliente → servidor**

```json
{ "type": "generate", "model": "gpt2", "prompt": "...", "tokens": 40,
  "params": { "mode": "sample", "temperature": 0.8, "top_k": 40,
              "top_p": 0.95, "repetition_penalty": 1.15, "no_repeat_ngram": 0 } }
```

```json
{ "type": "select_layer", "layer": 3 }
```

**Servidor → cliente**

| Tipo | Cuándo | Campos nuevos |
|---|---|---|
| `loading` | antes de cargar un modelo | igual que hoy |
| `meta` | modelo listo | `protocol`, `selected_layer` |
| `step` | cada token | `layer`, `delta` (texto), `token` (compatibilidad), `stats` |
| `layer` | al cambiar de capa | `layer`, `activations` |
| `error` | error de entrada o interno | `code`, `message`, `recoverable` |
| `done` | fin | igual que hoy |

---

# Apéndice B · Orden sugerido de PRs

| # | PR | Incrementos | Riesgo | Depende de |
|---|---|---|---|---|
| 0 | `chore/verification` | 0.1 + 0.2 | Bajo | — |
| 1 | `fix/ui-reset` | 1.1 | Bajo | — |
| 2 | `fix/ws-errors` | 1.2 + 1.3 | Bajo | 1 |
| 3 | `fix/stream-decode` | 1.4 | Medio-bajo | 2 |
| 4 | `fix/session-lock` | 1.5 | Medio | 2, 3 |
| 5 | `perf/pos-slice` | 2.1 | Bajo | 4 |
| 6 | `perf/kv-cache` | 2.2 | Alto | 5 |
| 7 | `perf/payload` | 2.3 | Medio | 6 |
| 8 | `chore/docker-memory` | 3.1 | Bajo | — |
| 9 | `ui/sampling-report` | 4.1 + 4.2 | Bajo | 3 |
| 10 | `ui/attention-canvas` | 4.3 + 2.4 | Medio | 7 |
| 11 | `chore/ci-logging` | 5.1 + 5.2 | Bajo | — |

**Regla:** si un PR de rendimiento no muestra una mejora medible, se revierte sin culpa. La corrección y la claridad van antes que la optimización.

---

# Apéndice C · Checklist por PR

- [ ] `python -m py_compile server.py` (o `ruff check`).
- [ ] `pytest -q` (cuando exista la fase 0).
- [ ] `python scripts/smoke_ws.py` contra un servidor local.
- [ ] Prueba manual en navegador: generar, cambiar capa, seleccionar neurona, ver atención, copiar informe.
- [ ] Si es rendimiento: anotar payload, tiempo por token y RSS antes/después.
- [ ] Si cambia el protocolo o la UI: actualizar `README.md` y `NEXT_STEPS.md`.
- [ ] No commitear modelos, caché, `.venv` ni logs.

---

# Apéndice D · Criterio de “terminado”

Un incremento está terminado cuando:

1. El caso que motivaba el hallazgo ya no se reproduce.
2. Hay una forma reproducible de comprobarlo (test, script o pasos manuales).
3. El sistema sigue funcionando con GPT-2 y, cuando aplique, con Qwen.
4. La documentación refleja el comportamiento nuevo.
5. El cambio se puede revertir sin tocar otros incrementos.

> Prioridad recomendada: **Fase 1 completa → 2.1 → 2.2 → 3.1 → 2.3 → resto**. Así se eliminan primero los fallos que el usuario nota, y luego se gana velocidad con una base estable.
