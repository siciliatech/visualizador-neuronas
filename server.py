import asyncio
import gc
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer

LOG_LEVEL = getattr(
    logging, os.getenv("VISUALIZER_LOG_LEVEL", "INFO").strip().upper(), logging.INFO
)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("visualizador")
logger.setLevel(LOG_LEVEL)
# Las descargas de Hugging Face son muy verbosas en INFO y no aportan al
# diagnóstico de la generación; sus warnings/errores sí se conservan.
for _noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

DEFAULT_PROMPT = "¿Qué cosas te puedo preguntar?"
DEFAULT_MODEL = "qwen2.5-0.5b-instruct"
DEFAULT_TOKENS = 40
TOKEN_INTERVAL = 0.35
TOP_K = 16
TOP_ATTN = 8
# Los patrones de atención son seq×seq por capa y cabeza; recortamos prompts largos
# para que el caché temporal no consuma RAM de más.
MAX_INPUT_TOKENS = 512
# Límite de caracteres del prompt (no de tokens): evita mensajes enormes antes
# de tokenizar. El recorte real a MAX_INPUT_TOKENS lo hace build_input_ids.
MAX_PROMPT_CHARS = 10_000
# KV cache: se puede desactivar sin tocar código para comparar o revertir.
#   VISUALIZER_KV_CACHE=0 uvicorn server:app
USE_KV_CACHE = os.getenv("VISUALIZER_KV_CACHE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Cada modelo se carga bajo demanda y se descarga el anterior para no agotar la RAM.
MODEL_REGISTRY = {
    "gpt2": {
        "hf_id": "gpt2",
        "label": "GPT-2 base (124M, continúa texto)",
        "chat": False,
    },
    "qwen2.5-0.5b-instruct": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5-0.5B-Instruct (responder preguntas)",
        "chat": True,
    },
    "pythia-410m": {
        "hf_id": "EleutherAI/pythia-410m",
        "label": "Pythia-410m base (EleutherAI)",
        "chat": False,
    },
}


class SamplingParams(BaseModel):
    """Parámetros de muestreo validados antes de tocar el modelo."""

    model_config = ConfigDict(extra="ignore")

    mode: Literal["sample", "greedy"] = "sample"
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int = Field(default=40, ge=0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5.0)
    no_repeat_ngram: int = Field(default=0, ge=0, le=10)


class AblationSpec(BaseModel):
    """Neurona MLP que se fuerza a cero durante toda la generación."""

    model_config = ConfigDict(extra="forbid")

    layer: int = Field(ge=0)
    neuron: int = Field(ge=0)


class GenerateRequest(BaseModel):
    """Mensaje `generate` del WebSocket, con los mismos defaults de antes."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["generate"] = "generate"
    model: str = DEFAULT_MODEL
    prompt: str = Field(default=DEFAULT_PROMPT, max_length=MAX_PROMPT_CHARS)
    tokens: int = Field(default=DEFAULT_TOKENS, ge=1, le=100)
    params: SamplingParams = Field(default_factory=SamplingParams)
    # Semilla opcional para que una comparación (por ejemplo, una ablación)
    # reuse exactamente la misma secuencia de números aleatorios. Si no se
    # envía, el servidor elige una y la informa en `meta`.
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    # La UI lo activa solo cuando el usuario quiere exportar todas las capas.
    # En modo normal el step sigue enviando únicamente la capa visible.
    capture_all: bool = False
    # Ablación opcional de una neurona MLP. Se valida contra el modelo cargado
    # dentro de process_generate, cuando ya se conocen capas y d_mlp.
    ablation: AblationSpec | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value not in MODEL_REGISTRY:
            options = ", ".join(MODEL_REGISTRY)
            raise ValueError(f"modelo desconocido {value!r}; opciones válidas: {options}")
        return value

    @field_validator("prompt", mode="before")
    @classmethod
    def normalize_prompt(cls, value):
        if value is None:
            return DEFAULT_PROMPT
        if isinstance(value, str):
            return value.strip() or DEFAULT_PROMPT
        return value

    @field_validator("params", mode="before")
    @classmethod
    def normalize_params(cls, value):
        # `params: null` equivale a no enviar parámetros.
        return {} if value is None else value


class SelectLayerRequest(BaseModel):
    """Mensaje `select_layer`: cambia la capa que se envía en cada `step`."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["select_layer"]
    layer: int = Field(ge=0)


class ModelLoadError(RuntimeError):
    """Fallo al cargar el modelo; se reporta como error no recuperable."""


app = FastAPI(title="Visualizador de Activación Neuronal")

# El modelo se comparte entre conexiones, pero la carga y la generación se
# serializan: así nadie cambia el modelo mientras otro está haciendo forwards.
load_lock = asyncio.Lock()
generation_lock = asyncio.Lock()

# Solo estado del modelo. Las activaciones, la atención y la decodificación son
# locales a cada generación (process_generate) para no mezclar clientes.
_current = {
    "key": None,
    "bridge": None,
    "tokenizer": None,
    "labels": [],
    "neurons_per_layer": 0,
    "module_path": "",
}

# Hook estándar de TransformerLens para la activación post-MLP (dimensión d_mlp).
# En MLPs con compuerta (Qwen/LLaMA) es act(gate_proj(x)) * up_proj(x), es decir,
# la neurona completa, no solo la compuerta.
MLP_NEURON_HOOK = "blocks.{i}.mlp.hook_post"

# Atención post-softmax: [batch, n_heads, q, k]. Solo usamos la última query.
ATTN_HOOK = "blocks.{i}.attn.hook_pattern"

# No depender del CWD: el servidor puede arrancarse desde otro directorio.
INDEX_PATH = Path(__file__).resolve().parent / "index.html"


def load_model(key: str):
    if _current["key"] == key:
        return _current

    old_bridge = _current.get("bridge")
    _current.update(
        key=None,
        bridge=None,
        tokenizer=None,
        labels=[],
        neurons_per_layer=0,
        module_path="",
    )
    del old_bridge
    gc.collect()

    spec = MODEL_REGISTRY[key]
    started = time.perf_counter()
    logger.info("cargando modelo %s (%s)…", key, spec["hf_id"])
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"])
    # Sin enable_compatibility_mode(): usamos los pesos HF crudos (las mismas
    # activaciones que antes) y evitamos ~1,5 GB extra de procesado de pesos.
    # El alias blocks.{i}.mlp.hook_post ya viene registrado por el bridge.
    bridge = TransformerBridge.boot_transformers(spec["hf_id"], device="cpu", tokenizer=tokenizer)
    bridge.eval()

    n_layers = bridge.cfg.n_layers
    neurons = getattr(bridge.cfg, "d_mlp", None) or getattr(bridge.cfg, "d_ff", None) or 0

    _current.update(
        key=key,
        bridge=bridge,
        tokenizer=tokenizer,
        labels=[f"layer_{i}" for i in range(n_layers)],
        neurons_per_layer=neurons,
        module_path=MLP_NEURON_HOOK,
    )
    logger.info(
        "modelo listo %s: %d capas x %d neuronas (%s) en %.2fs",
        key,
        n_layers,
        neurons,
        MLP_NEURON_HOOK,
        time.perf_counter() - started,
    )
    return _current


def build_input_ids(tokenizer, prompt: str, chat: bool):
    if chat and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, return_tensors="pt").input_ids
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    # Conservamos el final (incluye el turno del asistente en modo chat).
    if ids.shape[-1] > MAX_INPUT_TOKENS:
        ids = ids[:, -MAX_INPUT_TOKENS:]
    return ids


def safe_decode(tokenizer, ids: list[int]) -> str:
    """Decodifica ids sin los `�` de bytes UTF-8 incompletos en los extremos.

    Los tokenizers byte-level (GPT-2) parten a veces un carácter multibyte en
    dos tokens; decodificar cada token por separado produce `�`. Decodificar el
    prefijo completo y recortar los reemplazos de los extremos deja el texto
    seguro.
    """
    return tokenizer.decode(ids, skip_special_tokens=True).strip("\ufffd")


def decode_delta(tokenizer, ids: list[int], previous_text: str) -> tuple[str, str]:
    """Devuelve (texto nuevo, texto acumulado) para un prefijo de tokens."""
    full_text = safe_decode(tokenizer, ids)
    if full_text.startswith(previous_text):
        delta = full_text[len(previous_text) :]
    else:
        delta = full_text
    return delta, full_text


def special_token_label(tokenizer, token_id: int) -> str | None:
    """Devuelve una etiqueta legible para un token especial, o None.

    `decode(..., skip_special_tokens=True)` convierte los tokens de control de
    los modelos chat (por ejemplo `<|im_start|>`) en cadena vacía. Para la
    atención conviene mostrarlos con su nombre real en vez de dejar la columna
    en blanco.
    """
    special_ids = getattr(tokenizer, "all_special_ids", None) or []
    if token_id not in special_ids:
        return None
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is not None:
        try:
            label = convert(token_id)
        except (KeyError, TypeError, ValueError):
            label = None
        if isinstance(label, str) and label:
            return label
    return f"<|special_{token_id}|>"


def token_labels(tokenizer, ids: list[int]) -> list[str]:
    """Etiqueta cada token con el texto que completa.

    Un token que solo aporta bytes incompletos queda como ""; el token que
    completa el carácter recibe el texto entero. Los tokens especiales se
    etiquetan con su nombre (`<|im_start|>`, `<|endoftext|>`, …) para que la
    atención no los muestre como si fueran espacios. Así la atención no
    muestra `�` y el stream puede enviar deltas seguros.
    """
    labels = []
    emitted = ""
    for i in range(1, len(ids) + 1):
        delta, emitted = decode_delta(tokenizer, ids[:i], emitted)
        labels.append(special_token_label(tokenizer, ids[i - 1]) or delta)
    return labels


def update_decode_state(state: dict | None, tokenizer, ids: list[int]) -> dict:
    """Mantiene las etiquetas del contexto creciendo de uno en uno.

    El contexto normal solo añade un token por paso; en ese caso basta con
    decodificar el prefijo nuevo. Si el contexto cambia (otro prompt, otro
    modelo), se recalculan todas las etiquetas.
    """
    if state is not None and ids == state["ids"]:
        return state

    if state is not None and len(ids) == len(state["ids"]) + 1 and ids[:-1] == state["ids"]:
        delta, full_text = decode_delta(tokenizer, ids, state["text"])
        if full_text.startswith(state["text"]):
            state["labels"].append(special_token_label(tokenizer, ids[-1]) or delta)
            state["ids"] = ids
            state["text"] = full_text
            return state

    return {
        "ids": ids,
        "labels": token_labels(tokenizer, ids),
        "text": safe_decode(tokenizer, ids),
    }


def ban_repeated_ngrams(logits, ids: list[int], n: int):
    if n <= 1 or len(ids) < n:
        return
    suffix = tuple(ids[-(n - 1) :])
    for i in range(len(ids) - n + 1):
        if tuple(ids[i : i + n - 1]) == suffix:
            logits[0, ids[i + n - 1]] = float("-inf")


def sample_next_token(logits, seen_ids: list[int], params: SamplingParams):
    logits = logits.clone()
    rp = params.repetition_penalty
    if rp != 1.0:
        uniq = torch.tensor(sorted(set(seen_ids)), dtype=torch.long)
        scores = logits[0, uniq]
        logits[0, uniq] = torch.where(scores < 0, scores * rp, scores / rp)

    ban_repeated_ngrams(logits, seen_ids, params.no_repeat_ngram)

    temperature = params.temperature
    if params.mode == "greedy" or temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    top_k = params.top_k
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # top_p=0 se mantiene como "sin filtro nucleus" por compatibilidad.
    top_p = params.top_p or 1.0
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) == 0:
        return int(torch.argmax(logits, dim=-1).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def eos_token_ids(tokenizer) -> set[int]:
    """Normaliza `eos_token_id`, que algunos tokenizers exponen como lista."""
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(token_id) for token_id in eos if token_id is not None}
    return {int(eos)}


def payload_size(payload: dict) -> int:
    """Tamaño UTF-8 del JSON compacto que se envía por WebSocket."""
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def make_ablation_hook(neuron: int):
    """Devuelve un hook de TransformerLens que apaga una neurona MLP.

    `hook_post` tiene forma [batch, posición, d_mlp]. Se clona la activación
    para no modificar tensores que el modelo pueda reutilizar y se deja el
    resto de neuronas intacto.
    """

    def ablate(tensor: torch.Tensor, hook) -> torch.Tensor:
        output = tensor.clone()
        output[..., neuron] = 0.0
        return output

    return ablate


def forward_once(
    ids: torch.Tensor,
    model: dict,
    decode_state: dict | None,
    *,
    context_ids: list[int] | None = None,
    past: object | None = None,
    use_cache: bool = False,
    ablation: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, dict, dict, dict, object | None]:
    bridge = model["bridge"]
    tokenizer = model["tokenizer"]
    n_layers = len(model["labels"])
    # En modo KV cache `ids` puede ser solo el token nuevo; las etiquetas de
    # atención y el texto deben seguir el contexto completo.
    context_ids = ids[0].tolist() if context_ids is None else context_ids
    decode_state = update_decode_state(decode_state, tokenizer, context_ids)
    cache_kwargs = {
        "names_filter": lambda name: (
            name.endswith(".mlp.hook_post") or name.endswith(".attn.hook_pattern")
        ),
        "return_cache_object": False,
        # Solo nos interesa la última posición: evita guardar el caché de todas
        # las queries y baja el pico de memoria en prompts largos.
        "pos_slice": -1,
    }
    forward_started = time.perf_counter()
    ablation_hook_point = None
    ablation_level = None
    if ablation is not None:
        layer_idx, neuron_idx = ablation
        hook_name = MLP_NEURON_HOOK.format(i=layer_idx)
        ablation_hook_point = bridge.get_hook_point(hook_name)
        if ablation_hook_point is None:
            raise RuntimeError(f"No se encontró el hook de ablación {hook_name!r}.")
        # Nivel propio para retirar solo este hook; run_with_cache usa un nivel
        # interno distinto para sus hooks de caché.
        ablation_level = bridge.context_level
        ablation_hook_point.add_hook(make_ablation_hook(neuron_idx), level=ablation_level)
    try:
        with torch.no_grad():
            if use_cache:
                output, cache = bridge.run_with_cache(
                    ids,
                    return_type="logits_and_cache",
                    use_cache=True,
                    past_key_values=past,
                    **cache_kwargs,
                )
                if isinstance(output, tuple) and len(output) == 2:
                    logits, new_past = output
                else:
                    # El modelo ignoró `use_cache`: el siguiente paso vuelve al
                    # forward completo.
                    logits, new_past = output, None
            else:
                logits, cache = bridge.run_with_cache(
                    ids,
                    use_cache=False,
                    **cache_kwargs,
                )
                new_past = None
    finally:
        if ablation_hook_point is not None:
            ablation_hook_point.remove_hooks(level=ablation_level)
    cache_mode = "off" if not use_cache else ("reuse" if past is not None else "prefill")
    logger.info(
        "forward modelo=%s contexto=%d entrada=%d cache=%s tiempo=%.3fs",
        model.get("key") or "?",
        len(context_ids),
        ids.shape[-1],
        cache_mode,
        time.perf_counter() - forward_started,
    )

    # Solo guardamos el último token y una copia pequeña (lista de floats), para
    # que el caché completo de TransformerLens se libere al salir de la función.
    activations = {}
    for layer_idx in range(n_layers):
        tensor = cache.get(MLP_NEURON_HOOK.format(i=layer_idx))
        if tensor is None:
            continue
        acts = tensor[0, -1, :].detach().float().cpu()
        activations[f"layer_{layer_idx}"] = [round(v, 3) for v in acts.tolist()]

    # Atención real: de la última query a todas las claves. Guardamos solo el
    # top-K por cabeza para que el payload sea ligero.
    attn_layers = {}
    for layer_idx in range(n_layers):
        pattern = cache.get(ATTN_HOOK.format(i=layer_idx))
        if pattern is None:
            continue
        last = pattern[0, :, -1, :].detach().float().cpu()  # [heads, seq]
        k = min(TOP_ATTN, last.shape[-1])
        vals, idxs = torch.topk(last, k, dim=-1)
        heads = []
        for h in range(last.shape[0]):
            heads.append(
                [
                    [int(pos), round(float(w), 4)]
                    for pos, w in zip(idxs[h].tolist(), vals[h].tolist())
                ]
            )
        attn_layers[f"layer_{layer_idx}"] = heads

    attention = {
        "tokens": list(decode_state["labels"]),
        "layers": attn_layers,
    }

    return logits, activations, attention, decode_state, new_past


def activation_stats(activations: dict[str, list[float]]) -> dict:
    """Agregados de todas las capas para el informe del frontend.

    El `step` solo lleva la capa visible en el heatmap, pero el informe
    necesita seguir mostrando el récord global (capa/neurona) y la capa con
    mayor media de magnitud. Calculamos esos agregados aquí, recorriendo el
    estado completo que el servidor ya tiene en memoria.
    """
    max_abs = {"value": 0.0, "layer": 0, "neuron": 0}
    max_signed = {"value": 0.0, "layer": 0, "neuron": 0}
    layer_means: dict[str, float] = {}

    for key, values in activations.items():
        layer_idx = int(key.rsplit("_", 1)[-1])
        total = 0.0
        for neuron, value in enumerate(values):
            magnitude = abs(value)
            total += magnitude
            if magnitude > max_abs["value"]:
                max_abs = {"value": magnitude, "layer": layer_idx, "neuron": neuron}
                max_signed = {"value": value, "layer": layer_idx, "neuron": neuron}
        layer_means[key] = round(total / len(values), 3) if values else 0.0

    return {
        "max_abs": max_abs,
        "max_signed": max_signed,
        "layer_means": layer_means,
    }


async def send_layer_frame(websocket: WebSocket, session: dict, layer: int) -> None:
    """Envía la capa nueva usando el último estado completo conservado."""
    key = f"layer_{layer}"
    values = session["last_activations"].get(key, [])
    await websocket.send_json(
        {
            "type": "layer",
            "layer": layer,
            "activations": values,
        }
    )


@app.get("/")
async def index():
    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/models")
async def models():
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {"key": k, "label": v["label"], "chat": v["chat"]} for k, v in MODEL_REGISTRY.items()
        ],
    }


@app.get("/health")
async def health():
    """Estado del proceso sin tocar el modelo ni descargar nada."""
    key = _current.get("key")
    return {
        "status": "ok",
        "model_key": key,
        "ready": bool(key and _current.get("bridge") is not None),
    }


def validation_message(exc: ValidationError) -> str:
    """Resumen legible de un ValidationError, sin objetos internos de Pydantic."""
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "mensaje"
        message = error["msg"]
        # Pydantic prefija los ValueError de validadores propios.
        if message.startswith("Value error, "):
            message = message[len("Value error, ") :]
        details.append(f"{location}: {message}")
    return "Petición inválida — " + "; ".join(details)


def error_detail(exc: Exception) -> str:
    """Primera línea del error, acotada; la traza completa va solo al log."""
    lines = str(exc).strip().splitlines()
    detail = lines[0] if lines else type(exc).__name__
    return detail[:300]


async def send_error(websocket: WebSocket, code: str, message: str, recoverable: bool) -> None:
    try:
        await websocket.send_json(
            {
                "type": "error",
                "code": code,
                "message": message,
                "recoverable": recoverable,
            }
        )
    except Exception:
        # Si el cliente ya se fue, no hay a quién avisar.
        pass


async def process_generate(
    websocket: WebSocket,
    req: GenerateRequest,
    session: dict | None = None,
) -> None:
    spec = MODEL_REGISTRY[req.model]
    generation_started = time.perf_counter()
    if session is None:
        session = {"selected_layer": 0, "last_activations": {}}
    # Al empezar una generación no hay estado de la anterior que reutilizar.
    session["last_activations"] = {}

    # La carga se protege aparte para que dos clientes que piden el mismo
    # modelo no lo descarguen/carguen a la vez. Se revalida la clave dentro
    # del lock: el primero carga y el segundo reutiliza el modelo.
    async with load_lock:
        if _current["key"] != req.model:
            # Avisar al frontend ANTES de cargar: la primera vez descarga los
            # pesos de Hugging Face y puede tardar varios minutos sin feedback.
            await websocket.send_json(
                {
                    "type": "loading",
                    "model": req.model,
                    "label": spec["label"],
                }
            )
            try:
                state = await asyncio.to_thread(load_model, req.model)
            except Exception as exc:
                raise ModelLoadError(
                    f"No se pudo cargar el modelo {req.model!r}: {error_detail(exc)}"
                ) from exc
        else:
            state = _current

    n_layers = len(state["labels"])
    if n_layers:
        session["selected_layer"] = min(max(int(session.get("selected_layer", 0)), 0), n_layers - 1)
    selected_layer = session["selected_layer"]

    ablation = None
    if req.ablation is not None:
        if req.ablation.layer >= n_layers:
            await send_error(
                websocket,
                "invalid_request",
                f"Capa de ablación {req.ablation.layer} fuera de rango; el modelo tiene "
                f"{n_layers} capas (0..{n_layers - 1}).",
                recoverable=True,
            )
            return
        if req.ablation.neuron >= state["neurons_per_layer"]:
            await send_error(
                websocket,
                "invalid_request",
                f"Neurona de ablación {req.ablation.neuron} fuera de rango; la capa tiene "
                f"{state['neurons_per_layer']} neuronas "
                f"(0..{state['neurons_per_layer'] - 1}).",
                recoverable=True,
            )
            return
        ablation = (req.ablation.layer, req.ablation.neuron)

    generation_seed = req.seed
    if generation_seed is None:
        generation_seed = secrets.randbelow(2**31)
    torch.manual_seed(generation_seed)

    tokenizer = state["tokenizer"]
    eos_ids = eos_token_ids(tokenizer)
    curr_ids = build_input_ids(tokenizer, req.prompt, spec["chat"])
    seen_ids = curr_ids[0].tolist()
    generated_ids: list[int] = []
    generated_text = ""
    decode_state = None
    use_kv = USE_KV_CACHE
    past = None
    logger.info(
        "generación inicio modelo=%s chat=%s contexto=%d tokens=%d capa=%d kv=%s seed=%d ablación=%s",
        req.model,
        spec["chat"],
        len(seen_ids),
        req.tokens,
        selected_layer,
        use_kv,
        generation_seed,
        f"capa {ablation[0]} neurona {ablation[1]}" if ablation else "no",
    )

    # Forward de calentamiento: confirma que los hooks disparan y nos da
    # el número real de neuronas antes de mandar el "meta" al frontend.
    logits, activations, attention, decode_state, past = await asyncio.to_thread(
        forward_once,
        curr_ids,
        state,
        decode_state,
        context_ids=list(seen_ids),
        past=None,
        use_cache=use_kv,
        ablation=ablation,
    )
    # Si el modelo no devolvió caché, seguimos con forwards completos.
    if use_kv and past is None:
        use_kv = False
    neurons = len(next(iter(activations.values()))) if activations else state["neurons_per_layer"]
    attn_heads = len(next(iter(attention["layers"].values()))) if attention["layers"] else 0

    await websocket.send_json(
        {
            "type": "meta",
            "protocol": 2,
            "model": req.model,
            "model_label": spec["label"],
            "chat": spec["chat"],
            "module_path": state["module_path"],
            "prompt": req.prompt,
            "layers": n_layers,
            # `selected_layer` es el nombre del apéndice del plan; `layer`
            # queda como alias para clientes que ya leen el campo del step.
            "selected_layer": selected_layer,
            "layer": selected_layer,
            "neurons_per_layer": neurons,
            "top_k": TOP_K,
            "capture_all": req.capture_all,
            "seed": generation_seed,
            "ablation": req.ablation.model_dump() if req.ablation else None,
            "attention": {
                "available": bool(attention["layers"]),
                "heads": attn_heads,
                "top_k": TOP_ATTN,
            },
        }
    )

    generated = 0
    stop_reason = "limit"
    for step in range(req.tokens):
        # El forward de calentamiento ya produjo los logits del primer token;
        # reutilizarlos evita una pasada completa idéntica por prompt.
        next_token_id = sample_next_token(logits[:, -1, :], seen_ids, req.params)
        if next_token_id in eos_ids:
            stop_reason = "eos"
            break
        curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]])], dim=-1)
        seen_ids.append(next_token_id)
        generated_ids.append(next_token_id)
        # Decodificar el prefijo generado y quedarnos con lo nuevo evita `�`
        # cuando un carácter multibyte se reparte entre dos tokens.
        delta, generated_text = decode_delta(tokenizer, generated_ids, generated_text)
        generated += 1

        top_by_layer = {}
        for key, values in activations.items():
            ranked = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
            top_by_layer[key] = [[i, values[i]] for i in ranked[:TOP_K]]

        # Guardamos el estado completo para responder a `select_layer` sin
        # esperar al siguiente token, pero enviamos solo la capa visible.
        session["last_activations"] = activations
        if n_layers:
            selected_layer = min(max(int(session.get("selected_layer", 0)), 0), n_layers - 1)
            session["selected_layer"] = selected_layer
        layer_key = f"layer_{selected_layer}"
        visible = {layer_key: activations[layer_key]} if layer_key in activations else {}
        stats = activation_stats(activations)
        stats["max_abs"]["token"] = delta
        stats["max_signed"]["token"] = delta

        payload = {
            "type": "step",
            "layer": selected_layer,
            "delta": delta,
            "token": delta,
            # `capture_all` es optativo: el CSV necesita todas las capas, pero
            # la vista normal mantiene el payload reducido.
            "activations": activations if req.capture_all else visible,
            "stats": stats,
            "top": top_by_layer,
            "attention": attention,
        }
        await websocket.send_json(payload)
        logger.info(
            "step %d/%d modelo=%s capa=%d contexto=%d delta=%r payload=%dB",
            generated,
            req.tokens,
            req.model,
            selected_layer,
            len(seen_ids),
            delta,
            payload_size(payload),
        )
        await asyncio.sleep(TOKEN_INTERVAL)

        # Preparar el siguiente paso con el contexto ya ampliado. Después del
        # último token no hace falta otro forward.
        if step + 1 < req.tokens:
            # Con KV cache solo pasamos el token nuevo; sin caché, el contexto
            # completo. `context_ids` siempre lleva el contexto para las
            # etiquetas de atención.
            model_ids = torch.tensor([[next_token_id]]) if use_kv and past is not None else curr_ids
            logits, activations, attention, decode_state, past = await asyncio.to_thread(
                forward_once,
                model_ids,
                state,
                decode_state,
                context_ids=list(seen_ids),
                past=past if use_kv else None,
                use_cache=use_kv,
                ablation=ablation,
            )
            # Si el modelo dejó de devolver caché, el siguiente paso usa el
            # contexto completo.
            if use_kv and past is None:
                use_kv = False

    await websocket.send_json(
        {
            "type": "done",
            "text": safe_decode(tokenizer, seen_ids),
            "generated_text": generated_text,
            "stopped": stop_reason,
            "tokens": generated,
        }
    )
    logger.info(
        "generación fin modelo=%s tokens=%d motivo=%s duración=%.2fs texto=%d chars",
        req.model,
        generated,
        stop_reason,
        time.perf_counter() - generation_started,
        len(generated_text),
    )


async def run_generation(websocket: WebSocket, req: GenerateRequest, session: dict) -> None:
    """Ejecuta una generación en segundo plano.

    Dejar libre el bucle de recepción permite procesar `select_layer` mientras
    el modelo genera: el cliente puede cambiar de capa sin esperar a que
    termine la respuesta.
    """
    try:
        if generation_lock.locked():
            await websocket.send_json(
                {
                    "type": "queued",
                    "message": "Otro cliente está generando; esperando turno…",
                }
            )
        async with generation_lock:
            # Si el cliente se fue mientras esperaba turno, no hace falta
            # generar para un socket cerrado. Si ya estaba generando, la
            # tarea termina en el siguiente envío y libera el lock.
            if session.get("disconnected"):
                return
            await process_generate(websocket, req, session)
    except WebSocketDisconnect:
        logger.debug("cliente desconectado durante la generación de %s", req.model)
        return
    except OSError:
        # El cliente se fue mientras enviábamos un frame; no es un fallo
        # interno del modelo y no hay a quién reportar.
        logger.debug("socket cerrado durante la generación de %s", req.model)
        return
    except asyncio.CancelledError:
        raise
    except ModelLoadError as exc:
        logger.exception("fallo al cargar el modelo %s", req.model)
        await send_error(websocket, "model_load_failed", str(exc), recoverable=False)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    except Exception as exc:
        logger.exception("error interno durante la generación de %s", req.model)
        await send_error(
            websocket,
            "generation_failed",
            f"Error interno durante la generación: {error_detail(exc)}",
            recoverable=False,
        )
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Estado local de la conexión: capa visible y último estado completo para
    # responder a los cambios de capa sin volver a hacer un forward.
    session = {
        "selected_layer": 0,
        "last_activations": {},
        "disconnected": False,
    }
    generation_task: asyncio.Task | None = None

    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            # No cancelamos una generación en curso: `asyncio.to_thread` no se
            # puede interrumpir y liberar el lock dejaría el bridge en uso.
            session["disconnected"] = True
            return

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            await send_error(
                websocket,
                "invalid_json",
                f"El mensaje no es JSON válido: {exc.msg}",
                recoverable=True,
            )
            continue

        if isinstance(msg, dict) and msg.get("type") == "select_layer":
            try:
                selection = SelectLayerRequest.model_validate(msg)
            except ValidationError as exc:
                await send_error(
                    websocket,
                    "invalid_request",
                    validation_message(exc),
                    recoverable=True,
                )
                continue

            labels = _current.get("labels") or []
            if labels and selection.layer >= len(labels):
                await send_error(
                    websocket,
                    "invalid_request",
                    f"Capa {selection.layer} fuera de rango; el modelo tiene "
                    f"{len(labels)} capas (0..{len(labels) - 1}).",
                    recoverable=True,
                )
                continue

            session["selected_layer"] = selection.layer
            try:
                await send_layer_frame(websocket, session, selection.layer)
            except Exception:
                # El cliente pudo cerrarse justo después de enviar el mensaje.
                session["disconnected"] = True
                return
            continue

        try:
            req = GenerateRequest.model_validate(msg)
        except ValidationError as exc:
            # Error de entrada: la conexión sigue abierta para que el usuario
            # corrija el prompt o los parámetros sin recargar la página.
            await send_error(
                websocket,
                "invalid_request",
                validation_message(exc),
                recoverable=True,
            )
            continue

        if generation_task is not None and not generation_task.done():
            await send_error(
                websocket,
                "generation_in_progress",
                "Ya hay una generación en curso en esta conexión; esperá a que termine.",
                recoverable=True,
            )
            continue

        generation_task = asyncio.create_task(run_generation(websocket, req, session))
