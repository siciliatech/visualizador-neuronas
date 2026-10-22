#!/usr/bin/env python3
"""Prueba de humo del canal WebSocket del visualizador.

Conecta a /ws, genera unos tokens con un modelo y comprueba el protocolo 2:
llega `meta` con `protocol`/`selected_layer`, cada `step` trae solo la capa
seleccionada (o todas con `--capture-all`, con `stats` globales y `top` completo)
y finalmente llega `done`.
Sirve como verificación manual rápida después de cada cambio en `server.py` o
`index.html`.

Uso:
    uvicorn server:app --port 8000        # en otra terminal
    python scripts/smoke_ws.py
    python scripts/smoke_ws.py --model gpt2 --tokens 3 --prompt "Hola"
    python scripts/smoke_ws.py --layer 5   # prueba select_layer
    python scripts/smoke_ws.py --capture-all  # valida el modo de exportación CSV

La primera ejecución puede tardar minutos si el modelo no está descargado.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import websockets
from websockets.exceptions import ConnectionClosed

DEFAULT_URL = "ws://127.0.0.1:8000/ws"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba de humo del WebSocket del visualizador de neuronas."
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"URL del WebSocket (por defecto {DEFAULT_URL})"
    )
    parser.add_argument("--model", default="gpt2", help="Clave del modelo (por defecto gpt2)")
    parser.add_argument("--prompt", default="La capital de Francia es", help="Prompt a enviar")
    parser.add_argument("--tokens", type=int, default=3, help="Tokens a generar (1-100)")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Capa a seleccionar tras `meta` con un mensaje `select_layer`",
    )
    parser.add_argument(
        "--capture-all",
        action="store_true",
        help="Pide todas las capas en cada step (modo exportación CSV)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Segundos de espera por mensaje; la primera carga del modelo puede tardar (por defecto 900)",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"FALLO: {message}", file=sys.stderr)
    return 1


async def run(args: argparse.Namespace) -> int:
    payload = {
        # El servidor actual ignora "type"; incluirlo deja el script listo
        # para el protocolo con mensajes tipados del plan de mejoras.
        "type": "generate",
        "model": args.model,
        "prompt": args.prompt,
        "tokens": args.tokens,
        "capture_all": args.capture_all,
        "params": {
            "mode": "greedy",
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "no_repeat_ngram": 0,
        },
    }

    meta = None
    saw_requested_layer = False
    steps = 0
    stream = ""

    print(f"Conectando a {args.url} …")
    async with websockets.connect(args.url, max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps(payload))

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
            data = json.loads(raw)
            kind = data.get("type")

            if kind == "loading":
                print(f"  cargando {data.get('model')} … (la primera vez puede tardar)")
                continue

            if kind == "queued":
                print(f"  en cola: {data.get('message', 'otro cliente está generando')}")
                continue

            if kind == "meta":
                meta = data
                attention = data.get("attention") or {}
                print(
                    f"  meta: protocolo {data.get('protocol')} · "
                    f"{data.get('layers')} capas x "
                    f"{data.get('neurons_per_layer')} neuronas · "
                    f"capa {data.get('selected_layer')} · "
                    f"atención {'sí' if attention.get('available') else 'no'} "
                    f"({attention.get('heads', 0)} cabezas)"
                )
                if data.get("protocol") != 2:
                    return fail("meta no declara protocol: 2")
                if not data.get("layers") or not data.get("neurons_per_layer"):
                    return fail("meta no trae layers/neurons_per_layer")
                if bool(data.get("capture_all")) != args.capture_all:
                    return fail("meta.capture_all no coincide con --capture-all")

                initial_layer = data.get("selected_layer", data.get("layer"))
                if not isinstance(initial_layer, int) or not (0 <= initial_layer < data["layers"]):
                    return fail(f"meta.selected_layer inválida: {initial_layer!r}")

                if args.layer is not None:
                    if not 0 <= args.layer < data["layers"]:
                        return fail(
                            f"--layer {args.layer} fuera de rango (0..{data['layers'] - 1})"
                        )
                    await ws.send(json.dumps({"type": "select_layer", "layer": args.layer}))
                    print(f"  select_layer → capa {args.layer}")
                continue

            if kind == "layer":
                layer = data.get("layer")
                activations = data.get("activations")
                if not isinstance(layer, int):
                    return fail(f"layer sin índice válido: {layer!r}")
                if not isinstance(activations, list):
                    return fail("layer.activations no es una lista")
                if (
                    meta is not None
                    and activations
                    and (len(activations) != meta["neurons_per_layer"])
                ):
                    return fail(
                        f"layer {layer} tiene {len(activations)} neuronas, "
                        f"se esperaban {meta['neurons_per_layer']}"
                    )
                if args.layer is not None and layer == args.layer:
                    saw_requested_layer = True
                print(f"  layer: capa {layer} · {len(activations)} neuronas")
                continue

            if kind == "step":
                steps += 1
                text = data.get("delta", data.get("token", ""))
                if "\ufffd" in text:
                    return fail(
                        f"step {steps}: el delta contiene un carácter de reemplazo: {text!r}"
                    )
                stream += text
                activations = data.get("activations") or {}

                if meta is not None:
                    layer = data.get("layer")
                    if not isinstance(layer, int) or not (0 <= layer < meta["layers"]):
                        return fail(f"step {steps}: capa inválida {layer!r}")
                    expected_layers = {f"layer_{i}" for i in range(meta["layers"])}
                    expected_keys = expected_layers if args.capture_all else {f"layer_{layer}"}
                    if set(activations) != expected_keys:
                        return fail(
                            f"step {steps}: activations no coincide con "
                            f"{'todas las capas' if args.capture_all else f'la capa {layer}'}, "
                            f"trajo {sorted(activations)}"
                        )
                    for key in expected_keys:
                        if len(activations[key]) != meta["neurons_per_layer"]:
                            return fail(
                                f"step {steps}: {key} tiene "
                                f"{len(activations[key])} neuronas, "
                                f"se esperaban {meta['neurons_per_layer']}"
                            )

                    top = data.get("top") or {}
                    if set(top) != expected_layers:
                        return fail(
                            f"step {steps}: top incompleto; {len(top)} capas de {meta['layers']}"
                        )

                    stats = data.get("stats") or {}
                    max_abs = stats.get("max_abs") or {}
                    max_signed = stats.get("max_signed") or {}
                    layer_means = stats.get("layer_means") or {}
                    if not isinstance(max_abs.get("value"), (int, float)):
                        return fail(f"step {steps}: stats.max_abs.value inválido")
                    if not isinstance(max_signed.get("value"), (int, float)):
                        return fail(f"step {steps}: stats.max_signed.value inválido")
                    if set(layer_means) != expected_layers:
                        return fail(
                            f"step {steps}: stats.layer_means incompleto; "
                            f"{len(layer_means)} capas de {meta['layers']}"
                        )

                    if args.layer is not None and layer == args.layer:
                        saw_requested_layer = True

                print(
                    f"  step {steps}: {text!r} · {len(raw) / 1024:.1f} KiB · "
                    f"capa {data.get('layer')} · {len(activations)} capa(s)"
                )
                continue

            if kind == "error":
                return fail(data.get("message", "error del servidor"))

            if kind == "done":
                print(
                    f"  done: {data.get('tokens')} tokens · {data.get('stopped')} · "
                    f"{data.get('text', '')[-80:]!r}"
                )
                if steps == 0:
                    return fail("no se recibió ningún step")
                if args.layer is not None and not saw_requested_layer:
                    return fail(f"no se recibió ningún step ni frame en la capa {args.layer}")
                if data.get("tokens") != steps:
                    return fail(f"done dice {data.get('tokens')} tokens, se recibieron {steps}")
                if "generated_text" in data and data["generated_text"] != stream:
                    return fail(
                        "el stream acumulado no coincide con done.generated_text: "
                        f"{stream!r} != {data['generated_text']!r}"
                    )
                print(f"OK: {steps} steps · stream={stream!r}")
                return 0

            print(f"  (mensaje ignorado) {kind}")

    return fail("la conexión se cerró antes de recibir done")


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except (OSError, ConnectionClosed, asyncio.TimeoutError) as exc:
        return fail(f"conexión: {exc}")
    except json.JSONDecodeError as exc:
        return fail(f"respuesta no es JSON: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
