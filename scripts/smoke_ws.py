#!/usr/bin/env python3
"""Prueba de humo del canal WebSocket del visualizador.

Conecta a /ws, genera unos tokens con un modelo y comprueba que el protocolo
básico se cumple: llega `meta`, los `step` traen todas las capas/neuronas y
finalmente llega `done`. Sirve como verificación manual rápida después de
cada cambio en `server.py` o `index.html`.

Uso:
    uvicorn server:app --port 8000        # en otra terminal
    python scripts/smoke_ws.py
    python scripts/smoke_ws.py --model gpt2 --tokens 3 --prompt "Hola"

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
    parser.add_argument("--url", default=DEFAULT_URL, help=f"URL del WebSocket (por defecto {DEFAULT_URL})")
    parser.add_argument("--model", default="gpt2", help="Clave del modelo (por defecto gpt2)")
    parser.add_argument("--prompt", default="La capital de Francia es", help="Prompt a enviar")
    parser.add_argument("--tokens", type=int, default=3, help="Tokens a generar (1-100)")
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
                    f"  meta: {data.get('layers')} capas x "
                    f"{data.get('neurons_per_layer')} neuronas · "
                    f"atención {'sí' if attention.get('available') else 'no'} "
                    f"({attention.get('heads', 0)} cabezas)"
                )
                if not data.get("layers") or not data.get("neurons_per_layer"):
                    return fail("meta no trae layers/neurons_per_layer")
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
                    if len(activations) != meta["layers"]:
                        return fail(
                            f"step {steps}: {len(activations)} capas, "
                            f"se esperaban {meta['layers']}"
                        )
                    for name, values in activations.items():
                        if len(values) != meta["neurons_per_layer"]:
                            return fail(
                                f"step {steps}: {name} tiene {len(values)} neuronas, "
                                f"se esperaban {meta['neurons_per_layer']}"
                            )

                print(
                    f"  step {steps}: {text!r} · {len(raw) / 1024:.1f} KiB · "
                    f"{len(activations)} capas"
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
                if data.get("tokens") != steps:
                    return fail(
                        f"done dice {data.get('tokens')} tokens, se recibieron {steps}"
                    )
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
