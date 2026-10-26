import asyncio
import gc
import json
import logging
import math
import os
import re
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
    "mixtral-tiny": {
        "hf_id": "yujiepan/mixtral-8xtiny-random",
        "label": "Mixtral 8xTiny (MoE didáctico, 0,5M)",
        "chat": False,
        "architecture": "moe",
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
    "architecture": "dense",
    "unit": "neuron",
    "n_experts": None,
    "router_top_k": None,
    "layers_meta": [],
}

# Hook estándar de TransformerLens para la activación post-MLP (dimensión d_mlp).
# En MLPs con compuerta (Qwen/LLaMA) es act(gate_proj(x)) * up_proj(x), es decir,
# la neurona completa, no solo la compuerta.
MLP_NEURON_HOOK = "blocks.{i}.mlp.hook_post"

# En capas MoE, `hook_post` devuelve d_model (la salida del bloque), no d_mlp.
# La unidad visualizable es el experto: el peso que el router le asigna al
# último token. Los hooks de ruteo los expone TransformerLens por capa.
EXPERT_WEIGHTS_HOOK = "blocks.{i}.mlp.hook_expert_weights"
EXPERT_INDICES_HOOK = "blocks.{i}.mlp.hook_expert_indices"

# Atención post-softmax: [batch, n_heads, q, k]. Solo usamos la última query.
ATTN_HOOK = "blocks.{i}.attn.hook_pattern"

LAYER_IN_NAME = re.compile(r"layers\.(\d+)\.")

# No depender del CWD: el servidor puede arrancarse desde otro directorio.
INDEX_PATH = Path(__file__).resolve().parent / "index.html"


def config_int(cfg, *names: str) -> int | None:
    """Primer entero positivo de la config; None si no está declarado."""
    for name in names:
        value = getattr(cfg, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def detect_expert_count(bridge, layer: int) -> int | None:
    """Número de expertos de una capa MoE.

    La config no siempre declara `num_experts` (el Mixtral diminuto, por
    ejemplo, no lo expone). El peso del router (`gate.weight`) siempre tiene
    forma [n_experts, d_model], así que sirve de respaldo.
    """
    configured = config_int(
        bridge.cfg,
        "num_experts",
        "n_experts",
        "num_local_experts",
        "moe_num_experts",
    )
    if configured:
        return configured

    marker = f"layers.{layer}."
    for name, parameter in bridge.named_parameters():
        if marker not in name or ".mlp." not in name:
            continue
        low = name.lower()
        if not low.endswith("weight") or parameter.ndim != 2:
            continue
        if "gate" in low or "router" in low:
            return int(parameter.shape[0])
    return None


def detect_layers(bridge) -> list[dict]:
    """Describe qué se visualiza en cada capa: neuronas densas o expertos."""
    cfg = bridge.cfg
    d_mlp = getattr(cfg, "d_mlp", None) or getattr(cfg, "d_ff", None) or 0
    configured_top_k = config_int(
        cfg,
        "num_experts_per_tok",
        "n_experts_per_tok",
        "moe_top_k",
        "top_k",
    )
    hooks = set(getattr(bridge, "hook_dict", {}) or {})
    layers = []
    for layer in range(int(cfg.n_layers)):
        is_moe = (
            EXPERT_WEIGHTS_HOOK.format(i=layer) in hooks
            and EXPERT_INDICES_HOOK.format(i=layer) in hooks
        )
        n_experts = detect_expert_count(bridge, layer) if is_moe else None
        units = (n_experts or d_mlp) if is_moe else d_mlp
        layers.append(
            {
                "layer": layer,
                "architecture": "moe" if is_moe else "dense",
                "unit": "expert" if is_moe else "neuron",
                "units": units,
                "d_mlp": d_mlp or None,
                "n_experts": n_experts,
                "top_k": configured_top_k if is_moe else None,
            }
        )
    return layers


def architecture_summary(layers: list[dict]) -> dict:
    """Resumen del modelo a partir de sus capas, para `meta` y la UI."""
    moe_layers = [layer for layer in layers if layer["architecture"] == "moe"]
    if not moe_layers:
        return {
            "architecture": "dense",
            "unit": "neuron",
            "n_experts": None,
            "router_top_k": None,
            "neurons_per_layer": max((layer["units"] for layer in layers), default=0),
        }
    architecture = "moe" if len(moe_layers) == len(layers) else "hybrid"
    n_experts = max((layer["n_experts"] or layer["units"] for layer in moe_layers), default=0)
    top_k = max((layer["top_k"] for layer in moe_layers if layer["top_k"]), default=None)
    return {
        "architecture": architecture,
        "unit": "expert" if architecture == "moe" else "mixed",
        "n_experts": n_experts,
        "router_top_k": top_k,
        "neurons_per_layer": max((layer["units"] for layer in layers), default=0),
    }


def layer_meta(model: dict, layer: int) -> dict:
    """Metadatos de una capa; los modelos de tests pueden no tenerlos."""
    for info in model.get("layers_meta") or []:
        if info.get("layer") == layer:
            return info
    return {}


def expert_param_counts(bridge, n_layers: int) -> dict[int, int]:
    """Parámetros de expertos por capa, detectados por nombre.

    Funciona con expertos como ModuleList (`.experts.0.w1.weight`) y con
    parámetros batched (`.experts.gate_up_proj`). Excluye expertos compartidos
    porque están siempre activos.
    """
    counts = {layer: 0 for layer in range(n_layers)}
    for name, parameter in bridge.named_parameters():
        if ".experts." not in name or "shared_expert" in name:
            continue
        match = LAYER_IN_NAME.search(name)
        if match:
            layer = int(match.group(1))
            if layer in counts:
                counts[layer] += parameter.numel()
    return counts


def active_params_summary(bridge, layers: list[dict], routing: dict) -> dict | None:
    """Parámetros totales vs activos por token (estimación por ruteo)."""
    parameters = getattr(bridge, "parameters", None)
    total = sum(parameter.numel() for parameter in parameters()) if callable(parameters) else None
    if total is None:
        return None
    if not routing:
        return {
            "total_params": total,
            "active_params": total,
            "active_fraction": 1.0 if total else None,
            "expert_params_total": 0,
        }

    expert_params = expert_param_counts(bridge, len(layers))
    inactive = 0
    for key, info in routing.items():
        layer = int(key.rsplit("_", 1)[-1])
        top_k = info.get("top_k") or 0
        n_experts = info.get("n_experts") or 0
        if n_experts > 1 and top_k:
            inactive += int(expert_params.get(layer, 0) * (1 - top_k / n_experts))
    active = max(0, total - inactive)
    return {
        "total_params": total,
        "active_params": active,
        "active_fraction": round(active / total, 4) if total else None,
        "expert_params_total": sum(expert_params.values()),
    }


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
    layers_meta = detect_layers(bridge)
    summary = architecture_summary(layers_meta)
    neurons = summary["neurons_per_layer"]
    module_path = EXPERT_WEIGHTS_HOOK if summary["unit"] in {"expert", "mixed"} else MLP_NEURON_HOOK

    _current.update(
        key=key,
        bridge=bridge,
        tokenizer=tokenizer,
        labels=[f"layer_{i}" for i in range(n_layers)],
        neurons_per_layer=neurons,
        module_path=module_path,
        layers_meta=layers_meta,
        architecture=summary["architecture"],
        unit=summary["unit"],
        n_experts=summary["n_experts"],
        router_top_k=summary["router_top_k"],
    )
    logger.info(
        "modelo listo %s: %d capas x %d %s (%s, arquitectura=%s) en %.2fs",
        key,
        n_layers,
        neurons,
        "expertos" if summary["unit"] in {"expert", "mixed"} else "neuronas",
        module_path,
        summary["architecture"],
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


def make_expert_ablation_hook(expert: int):
    """Apaga un experto poniendo a cero su peso en el router.

    `hook_expert_weights` tiene forma [tokens, n_experts]. TransformerLens
    re-deriva la selección top-k después de una edición, así que el router
    elige al siguiente mejor experto: la ablación quita al experto de la
    decisión, no solo escala su salida.
    """

    def ablate(tensor: torch.Tensor, hook) -> torch.Tensor:
        output = tensor.clone()
        output[..., expert] = 0.0
        return output

    return ablate


def ablation_hook(model: dict, layer: int):
    """Nombre y función del hook de ablación según la arquitectura de la capa."""
    info = layer_meta(model, layer)
    if info.get("architecture") == "moe":
        return EXPERT_WEIGHTS_HOOK.format(i=layer), make_expert_ablation_hook
    return MLP_NEURON_HOOK.format(i=layer), make_ablation_hook


def layer_routing(cache: dict, layer: int, n_experts_hint: int | None = None) -> dict | None:
    """Ruteo del último token para una capa MoE.

    Devuelve el vector completo de pesos del router (lo que dibuja el heatmap)
    y los expertos elegidos con su peso. Algunas arquitecturas entregan los
    pesos ya recortados a top-k; en ese caso se reconstruye el vector disperso.
    """
    indices = cache.get(EXPERT_INDICES_HOOK.format(i=layer))
    weights = cache.get(EXPERT_WEIGHTS_HOOK.format(i=layer))
    if indices is None or weights is None:
        return None

    indices_flat = indices.detach().cpu().reshape(-1, indices.shape[-1])
    selected_indices = [int(value) for value in indices_flat[-1].tolist()]
    top_k = len(selected_indices)

    weight_flat = weights.detach().float().cpu()
    weight_flat = weight_flat.reshape(-1, weight_flat.shape[-1])
    last_weights = weight_flat[-1]
    width = int(last_weights.numel())

    if n_experts_hint and width <= top_k and n_experts_hint > width:
        n_experts = n_experts_hint
    else:
        n_experts = width

    full = [0.0] * n_experts
    if width == n_experts:
        full = [round(float(value), 5) for value in last_weights.tolist()]
    else:
        for index, value in zip(selected_indices, last_weights.tolist()):
            if 0 <= index < n_experts:
                full[index] = round(float(value), 5)

    selected_weights = [
        full[index] if 0 <= index < n_experts else 0.0 for index in selected_indices
    ]
    total = sum(selected_weights)
    if top_k > 1 and total > 0:
        entropy = 0.0
        for value in selected_weights:
            share = value / total
            if share > 0:
                entropy -= share * math.log2(share)
        entropy /= math.log2(top_k)
    else:
        entropy = 0.0

    return {
        "n_experts": n_experts,
        "top_k": top_k,
        "experts": selected_indices,
        "weights": [round(value, 5) for value in selected_weights],
        "entropy": round(entropy, 4),
        "router_activations": full,
    }


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
    is_moe_model = model.get("architecture") in {"moe", "hybrid"}
    cache_kwargs = {
        "names_filter": lambda name: (
            name.endswith(".mlp.hook_post")
            or name.endswith(".attn.hook_pattern")
            or name.endswith(".mlp.hook_expert_weights")
            or name.endswith(".mlp.hook_expert_indices")
        ),
        "return_cache_object": False,
        # Solo nos interesa la última posición: evita guardar el caché de todas
        # las queries y baja el pico de memoria en prompts largos. En MoE no se
        # puede recortar así: el hook de ruteo tiene forma [tokens, expertos] y
        # el recorte confundiría la dimensión de expertos con la de secuencia.
        "pos_slice": None if is_moe_model else -1,
    }
    forward_started = time.perf_counter()
    ablation_hook_point = None
    ablation_level = None
    if ablation is not None:
        layer_idx, neuron_idx = ablation
        hook_name, hook_factory = ablation_hook(model, layer_idx)
        ablation_hook_point = bridge.get_hook_point(hook_name)
        if ablation_hook_point is None:
            raise RuntimeError(f"No se encontró el hook de ablación {hook_name!r}.")
        # Nivel propio para retirar solo este hook; run_with_cache usa un nivel
        # interno distinto para sus hooks de caché.
        ablation_level = bridge.context_level
        ablation_hook_point.add_hook(hook_factory(neuron_idx), level=ablation_level)
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
    routing = {}
    for layer_idx in range(n_layers):
        info = layer_meta(model, layer_idx)
        has_expert_hooks = cache.get(EXPERT_WEIGHTS_HOOK.format(i=layer_idx)) is not None
        if info.get("architecture") == "moe" or (not info and has_expert_hooks):
            route = layer_routing(cache, layer_idx, info.get("n_experts"))
            if route is None:
                continue
            # En MoE la unidad no es la neurona d_mlp del bloque, sino cada
            # experto: el heatmap muestra los pesos del router.
            activations[f"layer_{layer_idx}"] = [
                round(value, 3) for value in route.pop("router_activations")
            ]
            routing[f"layer_{layer_idx}"] = route
            continue
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
        "routing": routing,
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
            {
                "key": k,
                "label": v["label"],
                "chat": v["chat"],
                "architecture": v.get("architecture", "dense"),
            }
            for k, v in MODEL_REGISTRY.items()
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
        info = layer_meta(state, req.ablation.layer)
        is_expert = info.get("unit") == "expert" or (not info and state.get("unit") == "expert")
        limit = info.get("units") or state["neurons_per_layer"]
        if req.ablation.neuron >= limit:
            unit = "Experto" if is_expert else "Neurona"
            plural = "expertos" if is_expert else "neuronas"
            await send_error(
                websocket,
                "invalid_request",
                f"{unit} de ablación {req.ablation.neuron} fuera de rango; la capa tiene "
                f"{limit} {plural} (0..{limit - 1}).",
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
        (
            f"capa {ablation[0]} "
            f"{'experto' if layer_meta(state, ablation[0]).get('unit') == 'expert' else 'neurona'} "
            f"{ablation[1]}"
            if ablation
            else "no"
        ),
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

    # El primer forward confirma el ruteo real (algunos modelos no declaran
    # top-k en la config). Completamos los metadatos antes de mandar `meta`.
    routing = attention.get("routing") or {}
    layers_meta = state.get("layers_meta") or []
    n_experts = state.get("n_experts")
    router_top_k = state.get("router_top_k")
    if routing:
        n_experts = max((info["n_experts"] for info in routing.values()), default=n_experts)
        router_top_k = max((info["top_k"] for info in routing.values()), default=router_top_k)
        for info in layers_meta:
            route = routing.get(f"layer_{info['layer']}")
            if route:
                info["n_experts"] = route["n_experts"]
                info["top_k"] = route["top_k"]
                if info.get("architecture") == "moe":
                    info["units"] = route["n_experts"]
        if state.get("architecture") not in {None, "dense"}:
            state.update(architecture_summary(layers_meta))
    active_params = active_params_summary(state["bridge"], layers_meta, routing)

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
            "architecture": state.get("architecture", "dense"),
            "unit": state.get("unit", "neuron"),
            "n_experts": n_experts,
            "router_top_k": router_top_k,
            "layers_meta": layers_meta,
            "routing_available": bool(routing),
            "active_params": active_params,
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
        routing = attention.get("routing") or {}

        top_by_layer = {}
        for key, values in activations.items():
            ranked = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
            if key in routing:
                # Los expertos no elegidos no se computan: mostrarlos como
                # ceros daría la idea falsa de que están apagados.
                ranked = [i for i in ranked if abs(values[i]) > 0]
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
        # El ruteo viaja en su propio campo: así `attention` conserva el
        # contrato anterior y no se duplica información en cada paso.
        attention_payload = {key: value for key, value in attention.items() if key != "routing"}

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
            "attention": attention_payload,
            "routing": routing,
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
