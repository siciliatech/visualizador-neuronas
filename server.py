import asyncio
import gc
import json
import os
import traceback
from typing import Literal

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer

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


class GenerateRequest(BaseModel):
    """Mensaje `generate` del WebSocket, con los mismos defaults de antes."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["generate"] = "generate"
    model: str = DEFAULT_MODEL
    prompt: str = Field(default=DEFAULT_PROMPT, max_length=MAX_PROMPT_CHARS)
    tokens: int = Field(default=DEFAULT_TOKENS, ge=1, le=100)
    params: SamplingParams = Field(default_factory=SamplingParams)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value not in MODEL_REGISTRY:
            options = ", ".join(MODEL_REGISTRY)
            raise ValueError(
                f"modelo desconocido {value!r}; opciones válidas: {options}"
            )
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


def load_model(key: str):
    if _current["key"] == key:
        return _current

    old_bridge = _current.get("bridge")
    _current.update(
        key=None, bridge=None, tokenizer=None, labels=[], neurons_per_layer=0,
        module_path="",
    )
    del old_bridge
    gc.collect()

    spec = MODEL_REGISTRY[key]
    print(f"[carga] {spec['hf_id']} ...", flush=True)
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
    print(f"[listo] {spec['hf_id']}: {n_layers} capas x {neurons} neuronas ({MLP_NEURON_HOOK})", flush=True)
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


def decode_delta(
    tokenizer, ids: list[int], previous_text: str
) -> tuple[str, str]:
    """Devuelve (texto nuevo, texto acumulado) para un prefijo de tokens."""
    full_text = safe_decode(tokenizer, ids)
    if full_text.startswith(previous_text):
        delta = full_text[len(previous_text):]
    else:
        delta = full_text
    return delta, full_text


def token_labels(tokenizer, ids: list[int]) -> list[str]:
    """Etiqueta cada token con el texto que completa.

    Un token que solo aporta bytes incompletos queda como ""; el token que
    completa el carácter recibe el texto entero. Así la atención no muestra
    `�` y el stream puede enviar deltas seguros.
    """
    labels = []
    emitted = ""
    for i in range(1, len(ids) + 1):
        delta, emitted = decode_delta(tokenizer, ids[:i], emitted)
        labels.append(delta)
    return labels


def update_decode_state(
    state: dict | None, tokenizer, ids: list[int]
) -> dict:
    """Mantiene las etiquetas del contexto creciendo de uno en uno.

    El contexto normal solo añade un token por paso; en ese caso basta con
    decodificar el prefijo nuevo. Si el contexto cambia (otro prompt, otro
    modelo), se recalculan todas las etiquetas.
    """
    if state is not None and ids == state["ids"]:
        return state

    if (
        state is not None
        and len(ids) == len(state["ids"]) + 1
        and ids[:-1] == state["ids"]
    ):
        delta, full_text = decode_delta(tokenizer, ids, state["text"])
        if full_text.startswith(state["text"]):
            state["labels"].append(delta)
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
    suffix = tuple(ids[-(n - 1):])
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n - 1]) == suffix:
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


def forward_once(
    ids: torch.Tensor,
    model: dict,
    decode_state: dict | None,
    *,
    context_ids: list[int] | None = None,
    past: object | None = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, dict, dict, dict, object | None]:
    bridge = model["bridge"]
    tokenizer = model["tokenizer"]
    n_layers = len(model["labels"])
    # En modo KV cache `ids` puede ser solo el token nuevo; las etiquetas de
    # atención y el texto deben seguir el contexto completo.
    context_ids = ids[0].tolist() if context_ids is None else context_ids
    decode_state = update_decode_state(decode_state, tokenizer, context_ids)
    cache_kwargs = {
        "names_filter": lambda name: name.endswith(".mlp.hook_post")
        or name.endswith(".attn.hook_pattern"),
        "return_cache_object": False,
        # Solo nos interesa la última posición: evita guardar el caché de todas
        # las queries y baja el pico de memoria en prompts largos.
        "pos_slice": -1,
    }
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
        last = pattern[0, :, -1, :].detach().float().cpu()   # [heads, seq]
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


@app.get("/")
async def index():
    return FileResponse("index.html", media_type="text/html")


@app.get("/models")
async def models():
    return {
        "default": DEFAULT_MODEL,
        "models": [{"key": k, "label": v["label"], "chat": v["chat"]} for k, v in MODEL_REGISTRY.items()],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


def validation_message(exc: ValidationError) -> str:
    """Resumen legible de un ValidationError, sin objetos internos de Pydantic."""
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "mensaje"
        message = error["msg"]
        # Pydantic prefija los ValueError de validadores propios.
        if message.startswith("Value error, "):
            message = message[len("Value error, "):]
        details.append(f"{location}: {message}")
    return "Petición inválida — " + "; ".join(details)


def error_detail(exc: Exception) -> str:
    """Primera línea del error, acotada; la traza completa va solo al log."""
    lines = str(exc).strip().splitlines()
    detail = lines[0] if lines else type(exc).__name__
    return detail[:300]


async def send_error(
    websocket: WebSocket, code: str, message: str, recoverable: bool
) -> None:
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


async def process_generate(websocket: WebSocket, req: GenerateRequest) -> None:
    spec = MODEL_REGISTRY[req.model]

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

    tokenizer = state["tokenizer"]
    curr_ids = build_input_ids(tokenizer, req.prompt, spec["chat"])
    seen_ids = curr_ids[0].tolist()
    generated_ids: list[int] = []
    generated_text = ""
    decode_state = None
    use_kv = USE_KV_CACHE
    past = None

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
    )
    # Si el modelo no devolvió caché, seguimos con forwards completos.
    if use_kv and past is None:
        use_kv = False
    neurons = len(next(iter(activations.values()))) if activations else state["neurons_per_layer"]
    attn_heads = len(next(iter(attention["layers"].values()))) if attention["layers"] else 0

    await websocket.send_json(
        {
            "type": "meta",
            "model": req.model,
            "model_label": spec["label"],
            "chat": spec["chat"],
            "module_path": state["module_path"],
            "prompt": req.prompt,
            "layers": len(state["labels"]),
            "neurons_per_layer": neurons,
            "top_k": TOP_K,
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
        if next_token_id == tokenizer.eos_token_id:
            stop_reason = "eos"
            break
        curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]])], dim=-1)
        seen_ids.append(next_token_id)
        generated_ids.append(next_token_id)
        # Decodificar el prefijo generado y quedarnos con lo nuevo evita `�`
        # cuando un carácter multibyte se reparte entre dos tokens.
        delta, generated_text = decode_delta(
            tokenizer, generated_ids, generated_text
        )
        generated += 1

        top_by_layer = {}
        for key, values in activations.items():
            ranked = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
            top_by_layer[key] = [[i, values[i]] for i in ranked[:TOP_K]]

        await websocket.send_json(
            {
                "type": "step",
                "delta": delta,
                "token": delta,
                "activations": activations,
                "top": top_by_layer,
                "attention": attention,
            }
        )
        await asyncio.sleep(TOKEN_INTERVAL)

        # Preparar el siguiente paso con el contexto ya ampliado. Después del
        # último token no hace falta otro forward.
        if step + 1 < req.tokens:
            # Con KV cache solo pasamos el token nuevo; sin caché, el contexto
            # completo. `context_ids` siempre lleva el contexto para las
            # etiquetas de atención.
            model_ids = (
                torch.tensor([[next_token_id]])
                if use_kv and past is not None
                else curr_ids
            )
            logits, activations, attention, decode_state, past = await asyncio.to_thread(
                forward_once,
                model_ids,
                state,
                decode_state,
                context_ids=list(seen_ids),
                past=past if use_kv else None,
                use_cache=use_kv,
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
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

        try:
            if generation_lock.locked():
                await websocket.send_json(
                    {
                        "type": "queued",
                        "message": "Otro cliente está generando; esperando turno…",
                    }
                )
            async with generation_lock:
                await process_generate(websocket, req)
        except WebSocketDisconnect:
            return
        except OSError:
            # El cliente se fue mientras enviábamos un frame; no es un fallo
            # interno del modelo y no hay a quién reportar.
            return
        except ModelLoadError as exc:
            traceback.print_exc()
            await send_error(
                websocket, "model_load_failed", str(exc), recoverable=False
            )
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
            return
        except Exception as exc:
            traceback.print_exc()
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
            return
