"""Tests de detalles del servidor: ruta del índice y token EOS."""

import asyncio
from pathlib import Path

import server


class TokenizerSinEos:
    pass


class TokenizerConEosEscalar:
    eos_token_id = 7


class TokenizerConEosLista:
    eos_token_id = [7, 8, None]


def test_index_no_depende_del_directorio_de_trabajo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    response = asyncio.run(server.index())

    assert Path(response.path) == Path(server.__file__).resolve().parent / "index.html"


def test_eos_token_ids_normaliza_escalar_y_lista():
    assert server.eos_token_ids(TokenizerSinEos()) == set()
    assert server.eos_token_ids(TokenizerConEosEscalar()) == {7}
    assert server.eos_token_ids(TokenizerConEosLista()) == {7, 8}


def test_hook_de_ablacion_apaga_solo_la_neurona_y_no_muta_la_entrada():
    import torch

    original = torch.tensor([[[1.0, 2.0, 3.0]]])
    ablated = server.make_ablation_hook(1)(original, None)

    assert ablated.tolist() == [[[1.0, 0.0, 3.0]]]
    assert original.tolist() == [[[1.0, 2.0, 3.0]]]


class _FakeHookPoint:
    def __init__(self):
        self.hooks = []

    def add_hook(self, fn, level=None, **_kwargs):
        self.hooks.append((fn, level))

    def remove_hooks(self, level=None):
        self.hooks = [(fn, hook_level) for fn, hook_level in self.hooks if hook_level != level]

    def apply(self, tensor):
        for fn, _level in self.hooks:
            tensor = fn(tensor, None)
        return tensor


class _FakeBridge:
    context_level = 0

    def __init__(self):
        self.hook_point = _FakeHookPoint()

    def get_hook_point(self, _name):
        return self.hook_point

    def run_with_cache(self, ids, **_kwargs):
        import torch

        seq = ids.shape[-1]
        activations = self.hook_point.apply(torch.ones(1, seq, 3))
        cache = {
            "blocks.0.mlp.hook_post": activations,
            "blocks.0.attn.hook_pattern": torch.ones(1, 1, seq, seq),
        }
        return torch.zeros(1, seq, 5), cache


class _FakeTokenizer:
    all_special_ids = []

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + i) for i in ids)


def test_forward_once_aplica_y_retira_el_hook_de_ablacion():
    import torch

    bridge = _FakeBridge()
    model = {
        "bridge": bridge,
        "tokenizer": _FakeTokenizer(),
        "labels": ["layer_0"],
        "neurons_per_layer": 3,
        "key": "fake",
    }

    _, normal, _, _, _ = server.forward_once(torch.tensor([[1, 2]]), model, None)
    _, ablated, _, _, _ = server.forward_once(torch.tensor([[1, 2]]), model, None, ablation=(0, 1))

    assert normal["layer_0"] == [1.0, 1.0, 1.0]
    assert ablated["layer_0"] == [1.0, 0.0, 1.0]
    assert bridge.hook_point.hooks == []
