"""Tests del bucle de generación: warmup, KV cache y conteo de forwards."""

import asyncio

import torch

import server


class FakeTokenizer:
    eos_token_id = 99

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + i) for i in ids)


class FakeWebSocket:
    def __init__(self):
        self.frames = []

    async def send_json(self, data):
        self.frames.append(data)


def run_generation(
    monkeypatch, token_sequence, tokens=3, use_kv=False, prefill_past=True
):
    calls = []

    def fake_forward(ids, model, decode_state, **kwargs):
        token_ids = ids[0].tolist()
        context_ids = kwargs.get("context_ids") or token_ids
        past = kwargs.get("past")
        use_cache = kwargs.get("use_cache", False)
        calls.append(
            {
                "ids": token_ids,
                "context": context_ids,
                "past": past,
                "use_cache": use_cache,
            }
        )
        n = len(calls)
        seq = len(token_ids)
        logits = torch.zeros(1, seq, 5)
        activations = {"layer_0": [float(n)]}
        attention = {
            "tokens": [f"t{i}" for i in range(len(context_ids))],
            "layers": {"layer_0": [[[0, 1.0]]]},
        }
        state = {
            "ids": context_ids,
            "labels": attention["tokens"],
            "text": "",
        }
        new_past = None
        if use_cache and (prefill_past or n > 1):
            new_past = {"length": len(context_ids)}
        return logits, activations, attention, state, new_past

    token_iter = iter(token_sequence)

    def fake_sample(logits, seen_ids, params):
        return next(token_iter)

    state = {
        "key": "gpt2",
        "bridge": object(),
        "tokenizer": FakeTokenizer(),
        "labels": ["layer_0"],
        "neurons_per_layer": 1,
        "module_path": server.MLP_NEURON_HOOK,
    }

    monkeypatch.setattr(server, "_current", state)
    monkeypatch.setattr(server, "load_lock", asyncio.Lock())
    monkeypatch.setattr(
        server, "build_input_ids", lambda *args: torch.tensor([[1, 2, 3]])
    )
    monkeypatch.setattr(server, "forward_once", fake_forward)
    monkeypatch.setattr(server, "sample_next_token", fake_sample)
    monkeypatch.setattr(server, "TOKEN_INTERVAL", 0)
    monkeypatch.setattr(server, "USE_KV_CACHE", use_kv)

    websocket = FakeWebSocket()
    req = server.GenerateRequest(model="gpt2", prompt="hola", tokens=tokens)

    asyncio.run(server.process_generate(websocket, req))
    return calls, websocket.frames


def test_warmup_logits_se_usan_para_el_primer_token(monkeypatch):
    calls, frames = run_generation(monkeypatch, [1, 2, 3], tokens=3)

    # Warmup + un forward por cada token salvo el último: no se repite el prompt.
    assert [call["ids"] for call in calls] == [
        [1, 2, 3],
        [1, 2, 3, 1],
        [1, 2, 3, 1, 2],
    ]
    assert all(call["use_cache"] is False for call in calls)
    assert all(call["past"] is None for call in calls)

    steps = [frame for frame in frames if frame["type"] == "step"]
    assert len(steps) == 3
    assert [step["activations"]["layer_0"] for step in steps] == [
        [1.0],
        [2.0],
        [3.0],
    ]
    assert [step["delta"] for step in steps] == ["a", "b", "c"]

    done = frames[-1]
    assert done["type"] == "done"
    assert done["tokens"] == 3
    assert done["generated_text"] == "abc"


def test_eos_no_dispara_un_forward_extra(monkeypatch):
    calls, frames = run_generation(monkeypatch, [1, 99], tokens=3)

    assert [call["ids"] for call in calls] == [
        [1, 2, 3],
        [1, 2, 3, 1],
    ]

    steps = [frame for frame in frames if frame["type"] == "step"]
    assert len(steps) == 1
    assert steps[0]["delta"] == "a"

    done = frames[-1]
    assert done["type"] == "done"
    assert done["stopped"] == "eos"
    assert done["tokens"] == 1


def test_kv_cache_manda_solo_el_token_nuevo_y_conserva_el_contexto(monkeypatch):
    calls, frames = run_generation(
        monkeypatch, [1, 2, 3], tokens=3, use_kv=True
    )

    assert [call["ids"] for call in calls] == [
        [1, 2, 3],
        [1],
        [2],
    ]
    assert [call["context"] for call in calls] == [
        [1, 2, 3],
        [1, 2, 3, 1],
        [1, 2, 3, 1, 2],
    ]
    assert [call["past"] for call in calls] == [
        None,
        {"length": 3},
        {"length": 4},
    ]
    assert all(call["use_cache"] is True for call in calls)

    steps = [frame for frame in frames if frame["type"] == "step"]
    assert [len(step["attention"]["tokens"]) for step in steps] == [3, 4, 5]
    assert frames[-1]["generated_text"] == "abc"


def test_kv_cache_no_cambia_la_salida(monkeypatch):
    _, full = run_generation(monkeypatch, [1, 2, 3], tokens=3, use_kv=False)
    _, kv = run_generation(monkeypatch, [1, 2, 3], tokens=3, use_kv=True)

    assert kv == full


def test_si_el_prefill_no_devuelve_cache_se_usa_el_contexto_completo(monkeypatch):
    calls, _ = run_generation(
        monkeypatch,
        [1, 2, 3],
        tokens=3,
        use_kv=True,
        prefill_past=False,
    )

    assert [call["ids"] for call in calls] == [
        [1, 2, 3],
        [1, 2, 3, 1],
        [1, 2, 3, 1, 2],
    ]
    assert calls[0]["use_cache"] is True
    assert all(call["use_cache"] is False for call in calls[1:])
    assert all(call["past"] is None for call in calls)
