"""Spike de captura para modelos densos y MoE (Mixture of Experts).

No forma parte del servidor: sirve para validar una arquitectura antes de
integrarla en el visualizador. Carga el modelo con TransformerLens, ejecuta un
forward con caché y reporta:

- los hooks MLP disponibles y sus formas;
- si `hook_post` es un vector de neuronas (`d_mlp`) o la salida del bloque
  (`d_model`), que es lo que ocurre en las capas MoE;
- el ruteo por capa: expertos elegidos, pesos, carga y entropía;
- una estimación de parámetros totales y activos por token.

Ejemplos:

    python scripts/spike_moe.py
    python scripts/spike_moe.py --model yujiepan/mixtral-8xtiny-random
    python scripts/spike_moe.py --model allenai/OLMoE-1B-7B-0924 --dtype bfloat16
    python scripts/spike_moe.py --model gpt2 --prompt "La capital de Francia es"

El modelo por defecto es un Mixtral diminuto (~0,5 M de parámetros) para que el
spike sea rápido y no dependa de una GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Any

import torch
from transformer_lens.model_bridge import TransformerBridge

DEFAULT_MODEL = "yujiepan/mixtral-8xtiny-random"
DEFAULT_PROMPT = "La capital de Francia es"
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def pick_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(value: str, device: torch.device) -> torch.dtype:
    if value != "auto":
        return getattr(torch, value)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    # En CPU, float32 es más rápido y más fiel para interpretabilidad.
    return torch.float32


def shape_of(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def hook_names(bridge: TransformerBridge, layer: int = 0) -> list[str]:
    prefix = f"blocks.{layer}.mlp"
    return sorted(name for name in bridge.hook_dict if name.startswith(prefix))


def layer_expert_params(bridge: TransformerBridge, n_layers: int) -> dict[int, int]:
    """Parámetros de expertos por capa, detectados por nombre.

    Funciona con expertos como ModuleList (`.experts.0.w1.weight`) y con
    parámetros batched (`.experts.gate_up_proj`). Los expertos compartidos
    (`shared_expert`) se excluyen porque están siempre activos.
    """
    params = {layer: 0 for layer in range(n_layers)}
    for name, parameter in bridge.named_parameters():
        if ".experts." not in name or "shared_expert" in name:
            continue
        match = LAYER_RE.search(name)
        if match:
            params[int(match.group(1))] += parameter.numel()
    return params


def gather_selected_weights(weights: torch.Tensor, indices: torch.Tensor) -> list[float]:
    """Devuelve el peso de cada experto elegido en el último token.

    Algunas arquitecturas entregan `weights` ya recortado a top-k y otras lo
    entregan disperso sobre todos los expertos. Se normalizan ambos casos.
    """
    last_weights = weights[-1].float().cpu()
    last_indices = indices[-1].cpu().tolist()
    if last_weights.numel() == indices.shape[-1]:
        selected = last_weights.tolist()
    else:
        selected = [float(last_weights[index]) for index in last_indices]
    return [round(value, 5) for value in selected]


def routing_summary(
    cache: dict[str, Any], n_layers: int, n_experts_hint: int | None = None
) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer in range(n_layers):
        indices = cache.get(f"blocks.{layer}.mlp.hook_expert_indices")
        weights = cache.get(f"blocks.{layer}.mlp.hook_expert_weights")
        if indices is None or weights is None:
            continue

        indices_cpu = indices.detach().cpu()
        weights_cpu = weights.detach().float().cpu()
        top_k = int(indices_cpu.shape[-1])
        n_experts = int(weights_cpu.shape[-1])
        if n_experts == top_k and n_experts_hint and n_experts_hint > top_k:
            n_experts = n_experts_hint

        flat = indices_cpu.reshape(-1)
        flat = flat[flat >= 0]
        counts = torch.bincount(flat, minlength=n_experts).float()
        total = counts.sum().clamp_min(1.0)
        load = (counts / total).tolist()
        entropy_bits = 0.0
        for share in load:
            if share > 0:
                entropy_bits -= share * math.log2(share)
        max_entropy = math.log2(n_experts) if n_experts > 1 else 1.0

        layers[f"layer_{layer}"] = {
            "n_experts": n_experts,
            "top_k": top_k,
            "last_token_experts": indices_cpu[-1].tolist(),
            "last_token_weights": gather_selected_weights(weights_cpu, indices_cpu),
            "load_entropy_bits": round(entropy_bits, 3),
            "load_entropy_normalized": round(entropy_bits / max_entropy, 3),
            "max_expert_share": round(max(load), 4),
            "dead_experts": [expert for expert, share in enumerate(load) if share == 0],
        }
    return layers


def active_params_estimate(
    total_params: int, expert_params: dict[int, int], routing: dict[str, Any]
) -> dict[str, Any] | None:
    """Estimación gruesa: descuenta la fracción de expertos no elegidos.

    No incluye ahorro de atención ni caché; sirve para comparar órdenes de
    magnitud, no como medición exacta de FLOPs.
    """
    if not routing:
        return None
    inactive = 0
    for key, info in routing.items():
        layer = int(key.rsplit("_", 1)[-1])
        fraction_inactive = 1.0 - (info["top_k"] / max(1, info["n_experts"]))
        inactive += int(expert_params.get(layer, 0) * fraction_inactive)
    active = total_params - inactive
    return {
        "total_params": total_params,
        "active_params_estimate": active,
        "active_fraction": round(active / total_params, 4) if total_params else None,
        "expert_params_total": sum(expert_params.values()),
    }


def human_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    env = report["environment"]
    lines.append("== Entorno ==")
    lines.append(
        f"torch {env['torch']} · device={env['device']} · dtype={env['dtype']} · "
        f"cuda={env['cuda']} · mps={env['mps']}"
    )
    model = report["model"]
    lines.append("\n== Modelo ==")
    lines.append(
        f"{model['name']} · capas={model['n_layers']} · d_model={model['d_model']} · "
        f"d_mlp={model['d_mlp']} · parámetros={model['total_params']:,}"
    )
    lines.append(f"hooks MLP (capa 0): {', '.join(model['mlp_hooks']) or 'ninguno'}")

    lines.append("\n== Formas por capa ==")
    for layer, info in report["layers"].items():
        lines.append(
            f"{layer}: post={info['post_shape']} · d_mlp={info['d_mlp']} · "
            f"expertos={info.get('n_experts', '—')} · top_k={info.get('top_k', '—')}"
        )

    if report["routing"]:
        lines.append("\n== Ruteo (último token) ==")
        for layer, info in report["routing"].items():
            lines.append(
                f"{layer}: expertos {info['last_token_experts']} "
                f"pesos {info['last_token_weights']} · "
                f"entropía={info['load_entropy_normalized']:.2f} normalizada · "
                f"máx. carga={info['max_expert_share']:.2f}"
            )
        active = report["active_params"]
        if active:
            lines.append("\n== Cómputo activo (estimación) ==")
            lines.append(
                f"parámetros totales={active['total_params']:,} · "
                f"activos≈{active['active_params_estimate']:,} "
                f"({active['active_fraction'] * 100:.1f} %) · "
                f"expertos={active['expert_params_total']:,}"
            )
    else:
        lines.append("\n== Ruteo ==")
        lines.append("No hay hooks de ruteo: el modelo es denso o no los expone.")

    lines.append("\n== Diagnóstico ==")
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    if not report["warnings"]:
        lines.append("- Sin avisos.")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    bridge = TransformerBridge.boot_transformers(args.model, device=device, dtype=dtype)
    bridge.eval()
    cfg = bridge.cfg

    ids = bridge.to_tokens(args.prompt)
    # Solo los hooks de la MLP: suficiente para el spike y evita cachés enormes.
    names_filter = lambda name: ".mlp." in name  # noqa: E731
    with torch.inference_mode():
        _, cache = bridge.run_with_cache(
            ids,
            names_filter=names_filter,
            return_cache_object=False,
            pos_slice=None,
        )

    n_layers = int(cfg.n_layers)
    d_mlp = getattr(cfg, "d_mlp", None) or getattr(cfg, "d_ff", None)
    n_experts_hint = (
        getattr(cfg, "num_experts", None)
        or getattr(cfg, "n_experts", None)
        or getattr(cfg, "num_local_experts", None)
    )
    hooks = hook_names(bridge, 0)

    layers: dict[str, Any] = {}
    warnings: list[str] = []
    for layer in range(n_layers):
        post = cache.get(f"blocks.{layer}.mlp.hook_post")
        post_shape = shape_of(post)
        indices = cache.get(f"blocks.{layer}.mlp.hook_expert_indices")
        weights = cache.get(f"blocks.{layer}.mlp.hook_expert_weights")
        n_experts = n_experts_hint
        if weights is not None and (indices is None or weights.shape[-1] != indices.shape[-1]):
            n_experts = int(weights.shape[-1])
        elif n_experts is None and indices is not None:
            n_experts = int(indices.shape[-1])
        info = {
            "post_shape": post_shape,
            "d_mlp": d_mlp,
            "n_experts": n_experts,
            "top_k": int(indices.shape[-1]) if indices is not None else None,
        }
        layers[f"layer_{layer}"] = info
        if post_shape and d_mlp and post_shape[-1] != d_mlp:
            warnings.append(
                f"{layer}: hook_post termina en {post_shape[-1]}, no en d_mlp={d_mlp}; "
                "en MoE hook_post es la salida del bloque, no las neuronas del experto."
            )

    routing = routing_summary(cache, n_layers, n_experts_hint)
    if routing and not warnings:
        warnings.append(
            "Hay ruteo MoE: usa hook_expert_indices/weights para la vista de expertos; "
            "no interpretes hook_post como d_mlp."
        )
    if not routing:
        warnings.append(
            "No se detectaron hooks de ruteo; el heatmap denso actual sigue siendo válido."
        )

    total_params = sum(parameter.numel() for parameter in bridge.parameters())
    expert_params = layer_expert_params(bridge, n_layers)
    active = active_params_estimate(total_params, expert_params, routing)

    return {
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "cuda": torch.cuda.is_available(),
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "model": {
            "name": args.model,
            "n_layers": n_layers,
            "d_model": getattr(cfg, "d_model", None),
            "d_mlp": d_mlp,
            "total_params": total_params,
            "mlp_hooks": hooks,
            "prompt_tokens": int(ids.shape[-1]),
            "prompt": args.prompt,
        },
        "layers": layers,
        "routing": routing,
        "active_params": active,
        "warnings": warnings,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ID de Hugging Face")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt de prueba")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Dispositivo; auto elige CUDA, luego MPS, luego CPU",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="Precisión; auto usa float32 en CPU y bfloat16/float16 en GPU",
    )
    parser.add_argument("--json", action="store_true", help="Salida JSON completa")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(human_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
