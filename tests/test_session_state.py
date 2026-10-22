"""Tests del estado local de cada generación y del contrato de `forward_once`."""

import torch

import server


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + i) for i in ids)


class FakeBridge:
    """Bridge mínimo: devuelve cachés con las formas que consume forward_once."""

    def __init__(self, offset=0.0):
        self.offset = offset
        self.calls = 0
        self.input_lengths = []
        self.past_lengths = []
        self.use_cache_flags = []

    def run_with_cache(
        self, ids, names_filter=None, return_cache_object=False, pos_slice=None, **kwargs
    ):
        self.calls += 1
        assert pos_slice == -1
        seq = ids.shape[-1]
        past = kwargs.get("past_key_values")
        use_cache = kwargs.get("use_cache", False)
        self.input_lengths.append(seq)
        self.past_lengths.append(past["length"] if past else None)
        self.use_cache_flags.append(use_cache)
        total = (past["length"] if past else 0) + seq
        cache = {}
        for layer in range(2):
            cache[f"blocks.{layer}.mlp.hook_post"] = torch.full(
                (1, 1, 3), self.offset + layer + 1.0
            )
            cache[f"blocks.{layer}.attn.hook_pattern"] = torch.full((1, 2, 1, total), 1.0 / total)
        logits = torch.zeros(1, seq, 5)
        if use_cache:
            return (logits, {"length": total}), cache
        return logits, cache


class NoCacheBridge(FakeBridge):
    """Simula un modelo que acepta `use_cache` pero no devuelve caché."""

    def run_with_cache(self, ids, **kwargs):
        kwargs["use_cache"] = False
        return super().run_with_cache(ids, **kwargs)


def make_model(bridge=None):
    return {
        "bridge": bridge or FakeBridge(),
        "tokenizer": FakeTokenizer(),
        "labels": ["layer_0", "layer_1"],
        "neurons_per_layer": 3,
        "module_path": server.MLP_NEURON_HOOK,
    }


def test_current_solo_guarda_el_modelo_compartido():
    assert "activations" not in server._current
    assert "attention" not in server._current
    assert "decode_state" not in server._current


def test_forward_once_devuelve_activaciones_y_atencion_locales():
    model = make_model()
    logits, activations, attention, state, past = server.forward_once(
        torch.tensor([[1, 2]]), model, None
    )
    assert past is None

    assert logits.shape == (1, 2, 5)
    assert activations["layer_0"] == [1.0, 1.0, 1.0]
    assert activations["layer_1"] == [2.0, 2.0, 2.0]
    assert attention["tokens"] == ["a", "b"]
    assert set(attention["layers"]) == {"layer_0", "layer_1"}
    assert state["labels"] == ["a", "b"]

    # forward_once no debe escribir estado de generación en el modelo compartido.
    assert "activations" not in model
    assert "attention" not in model
    assert "decode_state" not in model


def test_forward_once_crece_el_estado_de_decodificacion():
    model = make_model()
    _, _, attention_1, state, _ = server.forward_once(torch.tensor([[1, 2]]), model, None)
    _, _, attention_2, state, _ = server.forward_once(torch.tensor([[1, 2, 3]]), model, state)

    assert attention_1["tokens"] == ["a", "b"]
    assert attention_2["tokens"] == ["a", "b", "c"]
    assert state["labels"] == ["a", "b", "c"]


def test_forward_once_con_kv_cache_usa_contexto_completo_para_las_etiquetas():
    bridge = FakeBridge()
    model = make_model(bridge)

    _, _, attention_1, state, past = server.forward_once(
        torch.tensor([[1, 2]]), model, None, use_cache=True
    )
    assert attention_1["tokens"] == ["a", "b"]
    assert past == {"length": 2}

    # El modelo recibe solo el token nuevo, pero la atención se etiqueta con
    # todo el contexto (prompt + token generado).
    _, _, attention_2, state, past = server.forward_once(
        torch.tensor([[3]]),
        model,
        state,
        context_ids=[1, 2, 3],
        past=past,
        use_cache=True,
    )

    assert bridge.input_lengths == [2, 1]
    assert bridge.past_lengths == [None, 2]
    assert bridge.use_cache_flags == [True, True]
    assert attention_2["tokens"] == ["a", "b", "c"]
    assert past == {"length": 3}


def test_forward_once_hace_fallback_si_el_modelo_ignora_use_cache():
    model = make_model(NoCacheBridge())

    _, _, attention, state, past = server.forward_once(
        torch.tensor([[1, 2]]), model, None, use_cache=True
    )

    assert past is None
    assert attention["tokens"] == ["a", "b"]
    assert state["labels"] == ["a", "b"]


def test_dos_generaciones_no_comparten_activaciones_ni_atencion():
    model = make_model()
    _, activations_1, attention_1, state_1, _ = server.forward_once(
        torch.tensor([[1, 2]]), model, None
    )
    _, activations_2, attention_2, state_2, _ = server.forward_once(
        torch.tensor([[3, 4, 5]]), model, state_1
    )

    assert activations_1 is not activations_2
    assert attention_1 is not attention_2
    assert attention_1["tokens"] == ["a", "b"]
    assert attention_2["tokens"] == ["c", "d", "e"]
    assert state_1["labels"] == ["a", "b"]
    assert state_2["labels"] == ["c", "d", "e"]
