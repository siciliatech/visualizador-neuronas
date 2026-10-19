import asyncio
import gc

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer

DEFAULT_PROMPT = "¿Qué cosas te puedo preguntar?"
DEFAULT_TOKENS = 40
TOKEN_INTERVAL = 0.35
TOP_K = 16

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

app = FastAPI(title="Visualizador de Activación Neuronal")

_current = {
    "key": None,
    "bridge": None,
    "tokenizer": None,
    "labels": [],
    "neurons_per_layer": 0,
    "activations": {},
    "module_path": "",
}

# Hook estándar de TransformerLens para la activación post-MLP (dimensión d_mlp).
# En MLPs con compuerta (Qwen/LLaMA) es act(gate_proj(x)) * up_proj(x), es decir,
# la neurona completa, no solo la compuerta.
MLP_NEURON_HOOK = "blocks.{i}.mlp.hook_post"


def load_model(key: str):
    if _current["key"] == key:
        return _current

    old_bridge = _current.get("bridge")
    _current.update(key=None, bridge=None, tokenizer=None, activations={}, labels=[])
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
        activations={},
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
        return tokenizer(text, return_tensors="pt").input_ids
    return tokenizer(prompt, return_tensors="pt").input_ids


def ban_repeated_ngrams(logits, ids: list[int], n: int):
    if n <= 1 or len(ids) < n:
        return
    suffix = tuple(ids[-(n - 1):])
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n - 1]) == suffix:
            logits[0, ids[i + n - 1]] = float("-inf")


def sample_next_token(logits, seen_ids: list[int], params: dict):
    logits = logits.clone()
    rp = float(params.get("repetition_penalty", 1.0) or 1.0)
    if rp != 1.0:
        uniq = torch.tensor(sorted(set(seen_ids)), dtype=torch.long)
        scores = logits[0, uniq]
        logits[0, uniq] = torch.where(scores < 0, scores * rp, scores / rp)

    ban_repeated_ngrams(logits, seen_ids, int(params.get("no_repeat_ngram", 0) or 0))

    temperature = float(params.get("temperature", 0.8) or 0)
    if params.get("mode", "sample") == "greedy" or temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    top_k = int(params.get("top_k", 40) or 0)
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    top_p = float(params.get("top_p", 0.95) or 1.0)
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


def forward_once(ids: torch.Tensor):
    bridge = _current["bridge"]
    activations = _current["activations"]
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(
            ids,
            names_filter=lambda name: name.endswith(".mlp.hook_post"),
            return_cache_object=False,
        )

    # Solo guardamos el último token y una copia pequeña (lista de floats), para
    # que el caché completo de TransformerLens se libere al salir de la función.
    for layer_idx in range(len(_current["labels"])):
        tensor = cache.get(MLP_NEURON_HOOK.format(i=layer_idx))
        if tensor is None:
            continue
        acts = tensor[0, -1, :].detach().float().cpu()
        activations[f"layer_{layer_idx}"] = [round(v, 3) for v in acts.tolist()]

    return logits


@app.get("/")
async def index():
    return FileResponse("index.html", media_type="text/html")


@app.get("/models")
async def models():
    return {
        "default": "gpt2",
        "models": [{"key": k, "label": v["label"], "chat": v["chat"]} for k, v in MODEL_REGISTRY.items()],
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            params = msg.get("params", {})
            model_key = msg.get("model") or "gpt2"
            if model_key not in MODEL_REGISTRY:
                model_key = "gpt2"
            prompt = (msg.get("prompt") or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
            n_tokens = max(1, min(int(msg.get("tokens", DEFAULT_TOKENS)), 100))

            state = await asyncio.to_thread(load_model, model_key)
            tokenizer = state["tokenizer"]
            activations = state["activations"]
            spec = MODEL_REGISTRY[model_key]

            curr_ids = build_input_ids(tokenizer, prompt, spec["chat"])
            seen_ids = curr_ids[0].tolist()

            # Forward de calentamiento: confirma que los hooks disparan y nos da
            # el número real de neuronas antes de mandar el "meta" al frontend.
            await asyncio.to_thread(forward_once, curr_ids)
            neurons = len(next(iter(activations.values()))) if activations else state["neurons_per_layer"]

            await websocket.send_json(
                {
                    "type": "meta",
                    "model": model_key,
                    "model_label": spec["label"],
                    "chat": spec["chat"],
                    "module_path": state["module_path"],
                    "prompt": prompt,
                    "layers": len(state["labels"]),
                    "neurons_per_layer": neurons,
                    "top_k": TOP_K,
                }
            )

            for _ in range(n_tokens):
                logits = await asyncio.to_thread(forward_once, curr_ids)
                next_token_id = sample_next_token(logits[:, -1, :], seen_ids, params)
                if next_token_id == tokenizer.eos_token_id:
                    break
                token = tokenizer.decode(next_token_id, skip_special_tokens=True)
                curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]])], dim=-1)
                seen_ids.append(next_token_id)

                top_by_layer = {}
                for key, values in activations.items():
                    ranked = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
                    top_by_layer[key] = [[i, values[i]] for i in ranked[:TOP_K]]

                await websocket.send_json(
                    {
                        "type": "step",
                        "token": token,
                        "activations": activations.copy(),
                        "top": top_by_layer,
                    }
                )
                await asyncio.sleep(TOKEN_INTERVAL)

            await websocket.send_json({"type": "done", "text": tokenizer.decode(seen_ids, skip_special_tokens=True)})
    except WebSocketDisconnect:
        return
