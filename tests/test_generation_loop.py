"""Tests del bucle de generación: warmup, KV cache y conteo de forwards."""

import asyncio
import logging

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
    monkeypatch,
    token_sequence,
    tokens=3,
    use_kv=False,
    prefill_past=True,
    layers=1,
    selected_layer=0,
    eos_token_id=99,
    capture_all=False,
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
        activations = {f"layer_{layer}": [float(n + layer)] for layer in range(layers)}
        attention = {
            "tokens": [f"t{i}" for i in range(len(context_ids))],
            "layers": {f"layer_{layer}": [[[0, 1.0]]] for layer in range(layers)},
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

    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = eos_token_id
    state = {
        "key": "gpt2",
        "bridge": object(),
        "tokenizer": tokenizer,
        "labels": [f"layer_{layer}" for layer in range(layers)],
        "neurons_per_layer": 1,
        "module_path": server.MLP_NEURON_HOOK,
    }

    monkeypatch.setattr(server, "_current", state)
    monkeypatch.setattr(server, "load_lock", asyncio.Lock())
    monkeypatch.setattr(server, "build_input_ids", lambda *args: torch.tensor([[1, 2, 3]]))
    monkeypatch.setattr(server, "forward_once", fake_forward)
    monkeypatch.setattr(server, "sample_next_token", fake_sample)
    monkeypatch.setattr(server, "TOKEN_INTERVAL", 0)
    monkeypatch.setattr(server, "USE_KV_CACHE", use_kv)

    websocket = FakeWebSocket()
    req = server.GenerateRequest(
        model="gpt2",
        prompt="hola",
        tokens=tokens,
        capture_all=capture_all,
    )
    session = {"selected_layer": selected_layer, "last_activations": {}}

    asyncio.run(server.process_generate(websocket, req, session))
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


def test_eos_como_lista_detiene_la_generacion(monkeypatch):
    calls, frames = run_generation(monkeypatch, [1, 99], tokens=3, eos_token_id=[98, 99])

    assert len(calls) == 2
    assert len([frame for frame in frames if frame["type"] == "step"]) == 1
    assert frames[-1]["stopped"] == "eos"
    assert frames[-1]["tokens"] == 1


def test_kv_cache_manda_solo_el_token_nuevo_y_conserva_el_contexto(monkeypatch):
    calls, frames = run_generation(monkeypatch, [1, 2, 3], tokens=3, use_kv=True)

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


def test_step_envia_solo_la_capa_seleccionada_y_stats_globales(monkeypatch):
    _, frames = run_generation(
        monkeypatch,
        [1, 2],
        tokens=2,
        layers=2,
        selected_layer=1,
    )

    meta = frames[0]
    assert meta["protocol"] == 2
    assert meta["selected_layer"] == 1
    assert meta["layer"] == 1

    steps = [frame for frame in frames if frame["type"] == "step"]
    assert [set(step["activations"]) for step in steps] == [
        {"layer_1"},
        {"layer_1"},
    ]
    assert [step["activations"]["layer_1"] for step in steps] == [[2.0], [3.0]]
    assert [step["layer"] for step in steps] == [1, 1]

    # El overview sigue recibiendo todas las capas.
    assert [set(step["top"]) for step in steps] == [
        {"layer_0", "layer_1"},
        {"layer_0", "layer_1"},
    ]

    first_stats = steps[0]["stats"]
    assert first_stats["max_abs"] == {
        "value": 2.0,
        "layer": 1,
        "neuron": 0,
        "token": "a",
    }
    assert first_stats["max_signed"] == {
        "value": 2.0,
        "layer": 1,
        "neuron": 0,
        "token": "a",
    }
    assert first_stats["layer_means"] == {"layer_0": 1.0, "layer_1": 2.0}


def test_capture_all_envia_todas_las_capas_y_lo_declara_en_meta(monkeypatch):
    _, frames = run_generation(
        monkeypatch,
        [1, 2],
        tokens=2,
        layers=2,
        selected_layer=1,
        capture_all=True,
    )

    assert frames[0]["type"] == "meta"
    assert frames[0]["capture_all"] is True

    steps = [frame for frame in frames if frame["type"] == "step"]
    assert [set(step["activations"]) for step in steps] == [
        {"layer_0", "layer_1"},
        {"layer_0", "layer_1"},
    ]
    assert [step["activations"]["layer_0"] for step in steps] == [[1.0], [2.0]]
    assert [step["activations"]["layer_1"] for step in steps] == [[2.0], [3.0]]
    assert [step["layer"] for step in steps] == [1, 1]


def test_stats_globales_no_se_limitan_a_la_capa_visible(monkeypatch):
    _, frames = run_generation(
        monkeypatch,
        [1],
        tokens=1,
        layers=2,
        selected_layer=0,
    )

    step = next(frame for frame in frames if frame["type"] == "step")
    assert set(step["activations"]) == {"layer_0"}
    assert step["stats"]["max_abs"]["layer"] == 1
    assert step["stats"]["max_abs"]["value"] == 2.0
    assert step["stats"]["max_signed"]["layer"] == 1
    assert step["stats"]["layer_means"] == {"layer_0": 1.0, "layer_1": 2.0}


def test_activation_stats_usa_el_maximo_absoluto_y_conserva_el_signo():
    stats = server.activation_stats({"layer_0": [1.0, -4.0], "layer_1": [-5.0, 2.0]})

    assert stats["max_abs"] == {"value": 5.0, "layer": 1, "neuron": 0}
    assert stats["max_signed"] == {"value": -5.0, "layer": 1, "neuron": 0}
    assert stats["layer_means"] == {"layer_0": 2.5, "layer_1": 3.5}


def test_generacion_loguea_inicio_payload_y_fin(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="visualizador")

    run_generation(monkeypatch, [1], tokens=1)

    assert "generación inicio" in caplog.text
    assert "step 1/1" in caplog.text
    assert "payload=" in caplog.text
    assert "generación fin" in caplog.text
