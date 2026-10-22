"""Test del endpoint WebSocket: `select_layer` mientras se genera."""

import asyncio
import json

from fastapi import WebSocketDisconnect

import server


class FakeWebSocket:
    def __init__(self):
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.outgoing: asyncio.Queue = asyncio.Queue()
        self.accepted = False
        self.closed_code = None

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        item = await self.incoming.get()
        if item is None:
            raise WebSocketDisconnect()
        return item

    async def send_json(self, data):
        await self.outgoing.put(data)

    async def close(self, code=1000):
        self.closed_code = code


async def next_frame(ws: FakeWebSocket) -> dict:
    return await asyncio.wait_for(ws.outgoing.get(), timeout=1)


async def next_of_type(ws: FakeWebSocket, kind: str) -> dict:
    while True:
        frame = await next_frame(ws)
        if frame.get("type") == kind:
            return frame


def test_select_layer_responde_y_afecta_a_los_siguientes_steps(monkeypatch):
    release = asyncio.Event()
    finished = asyncio.Event()

    async def fake_process_generate(websocket, req, session):
        await websocket.send_json(
            {
                "type": "meta",
                "protocol": 2,
                "layers": 4,
                "neurons_per_layer": 2,
                "selected_layer": session["selected_layer"],
            }
        )
        session["last_activations"] = {f"layer_{i}": [float(i), 0.0] for i in range(4)}
        await websocket.send_json(
            {
                "type": "step",
                "layer": session["selected_layer"],
                "activations": {"layer_0": [0.0, 0.0]},
                "delta": "0",
                "token": "0",
            }
        )
        # La prueba cambia de capa mientras esta tarea está esperando.
        await release.wait()
        for n in range(1, 4):
            layer = session["selected_layer"]
            session["last_activations"] = {f"layer_{i}": [float(n + i), 0.0] for i in range(4)}
            await websocket.send_json(
                {
                    "type": "step",
                    "layer": layer,
                    "activations": {f"layer_{layer}": [float(n + layer), 0.0]},
                    "delta": str(n),
                    "token": str(n),
                }
            )
        finished.set()

    monkeypatch.setattr(server, "process_generate", fake_process_generate)
    monkeypatch.setattr(server, "generation_lock", asyncio.Lock())
    monkeypatch.setattr(
        server,
        "_current",
        {
            "key": "gpt2",
            "bridge": object(),
            "tokenizer": object(),
            "labels": [f"layer_{i}" for i in range(4)],
            "neurons_per_layer": 2,
            "module_path": server.MLP_NEURON_HOOK,
        },
    )

    async def scenario():
        ws = FakeWebSocket()
        endpoint = asyncio.create_task(server.websocket_endpoint(ws))
        await ws.incoming.put(
            json.dumps(
                {
                    "type": "generate",
                    "model": "gpt2",
                    "prompt": "hola",
                    "tokens": 4,
                }
            )
        )

        meta = await next_of_type(ws, "meta")
        assert meta["selected_layer"] == 0

        first_step = await next_of_type(ws, "step")
        assert first_step["layer"] == 0

        # Durante la generación, el endpoint sigue leyendo y responde con el
        # último estado completo (sin esperar a que termine la tarea).
        await ws.incoming.put(json.dumps({"type": "select_layer", "layer": 2}))
        layer_frame = await next_of_type(ws, "layer")
        assert layer_frame["layer"] == 2
        assert layer_frame["activations"] == [2.0, 0.0]

        release.set()
        step = await next_of_type(ws, "step")
        assert step["layer"] == 2
        assert set(step["activations"]) == {"layer_2"}
        await asyncio.wait_for(finished.wait(), timeout=1)

        await ws.incoming.put(None)
        await asyncio.wait_for(endpoint, timeout=1)

    asyncio.run(scenario())
