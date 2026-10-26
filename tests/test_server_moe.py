"""Tests del adaptador MoE del servidor, sin descargar modelos."""

import torch

import server


class FakeTokenizer:
    all_special_ids = []

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + i) for i in ids)


class FakeCfg:
    n_layers = 2
    d_mlp = 32
    num_experts = 8


class FakeParameter:
    def __init__(self, numel):
        self._numel = numel

    def numel(self):
        return self._numel


class FakeMoEBridge:
    """Bridge MoE mínimo: hook_post termina en d_model, no en d_mlp."""

    cfg = FakeCfg()
    context_level = 0
    hook_dict = {
        "blocks.0.mlp.hook_expert_weights": object(),
        "blocks.0.mlp.hook_expert_indices": object(),
        "blocks.1.mlp.hook_expert_weights": object(),
        "blocks.1.mlp.hook_expert_indices": object(),
    }

    def __init__(self):
        self.pos_slices = []

    def parameters(self):
        return [FakeParameter(1000), FakeParameter(4000)]

    def named_parameters(self):
        return iter(
            [
                ("model.layers.0.mlp.experts.0.w1.weight", FakeParameter(4000)),
            ]
        )

    def run_with_cache(self, ids, names_filter=None, pos_slice=None, **kwargs):
        self.pos_slices.append(pos_slice)
        seq = ids.shape[-1]
        # Pesos completos del router: solo los elegidos son distintos de cero.
        weights = torch.zeros(seq, 8)
        weights[-1, 6] = 0.6
        weights[-1, 7] = 0.4
        indices = torch.zeros(seq, 2, dtype=torch.long)
        indices[-1] = torch.tensor([6, 7])
        cache = {}
        for layer in range(2):
            cache[f"blocks.{layer}.mlp.hook_post"] = torch.full((1, seq, 8), 99.0)
            cache[f"blocks.{layer}.mlp.hook_expert_weights"] = weights
            cache[f"blocks.{layer}.mlp.hook_expert_indices"] = indices
            cache[f"blocks.{layer}.attn.hook_pattern"] = torch.full((1, 2, 1, seq), 1.0 / seq)
        return torch.zeros(1, seq, 5), cache


def make_moe_model(bridge):
    return {
        "bridge": bridge,
        "tokenizer": FakeTokenizer(),
        "labels": ["layer_0", "layer_1"],
        "neurons_per_layer": 8,
        "module_path": server.EXPERT_WEIGHTS_HOOK,
        "architecture": "moe",
        "unit": "expert",
        "n_experts": 8,
        "router_top_k": 2,
        "layers_meta": [
            {
                "layer": 0,
                "architecture": "moe",
                "unit": "expert",
                "units": 8,
                "d_mlp": 32,
                "n_experts": 8,
                "top_k": 2,
            },
            {
                "layer": 1,
                "architecture": "moe",
                "unit": "expert",
                "units": 8,
                "d_mlp": 32,
                "n_experts": 8,
                "top_k": 2,
            },
        ],
    }


def test_detect_layers_distingue_densa_de_moe():
    bridge = FakeMoEBridge()

    layers = server.detect_layers(bridge)
    summary = server.architecture_summary(layers)

    assert [layer["architecture"] for layer in layers] == ["moe", "moe"]
    assert [layer["units"] for layer in layers] == [8, 8]
    assert [layer["unit"] for layer in layers] == ["expert", "expert"]
    assert summary["architecture"] == "moe"
    assert summary["unit"] == "expert"
    assert summary["n_experts"] == 8
    # `d_mlp` sigue existiendo, pero no es la unidad del heatmap MoE.
    assert layers[0]["d_mlp"] == 32


def test_detect_layers_sin_hooks_de_ruteo_es_densa():
    class DenseBridge:
        cfg = FakeCfg()
        hook_dict = {"blocks.0.mlp.hook_post": object()}

    layers = server.detect_layers(DenseBridge())
    summary = server.architecture_summary(layers)

    assert [layer["architecture"] for layer in layers] == ["dense", "dense"]
    assert [layer["units"] for layer in layers] == [32, 32]
    assert summary["architecture"] == "dense"
    assert summary["unit"] == "neuron"


def test_forward_once_moe_usa_pesos_del_router_y_captura_ruteo():
    bridge = FakeMoEBridge()
    model = make_moe_model(bridge)

    _, activations, attention, _, _ = server.forward_once(torch.tensor([[1, 2]]), model, None)

    # El heatmap tiene una celda por experto, no d_model ni d_mlp.
    assert len(activations["layer_0"]) == 8
    assert activations["layer_0"][6] == 0.6
    assert activations["layer_0"][7] == 0.4
    assert activations["layer_0"][0] == 0.0
    # En MoE no se puede recortar el hook de ruteo con pos_slice=-1.
    assert bridge.pos_slices == [None]

    routing = attention["routing"]["layer_0"]
    assert routing["n_experts"] == 8
    assert routing["top_k"] == 2
    assert routing["experts"] == [6, 7]
    assert routing["weights"] == [0.6, 0.4]
    assert 0.0 <= routing["entropy"] <= 1.0
    assert "router_activations" not in routing


def test_layer_routing_reconstruye_pesos_recortados_a_top_k():
    indices = torch.tensor([[2, 0]])
    weights = torch.tensor([[0.8, 0.2]])

    cache = {
        "blocks.0.mlp.hook_expert_indices": indices,
        "blocks.0.mlp.hook_expert_weights": weights,
    }

    routing = server.layer_routing(cache, 0, n_experts_hint=4)

    assert routing["n_experts"] == 4
    assert routing["experts"] == [2, 0]
    assert routing["weights"] == [0.8, 0.2]
    assert routing["router_activations"] == [0.2, 0.0, 0.8, 0.0]


def test_hook_de_ablacion_de_experto_apaga_solo_su_columna():
    original = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    ablated = server.make_expert_ablation_hook(1)(original, None)

    assert ablated.tolist() == [[0.2, 0.0, 0.5]]
    assert original.tolist() == [[0.2, 0.3, 0.5]]


def test_ablacion_elige_el_hook_del_experto_en_moe():
    model = make_moe_model(FakeMoEBridge())

    hook_name, factory = server.ablation_hook(model, 1)

    assert hook_name == "blocks.1.mlp.hook_expert_weights"
    assert factory is server.make_expert_ablation_hook


def test_active_params_descuenta_expertos_no_elegidos():
    layers = [{"layer": 0, "architecture": "moe", "units": 8}]
    routing = {"layer_0": {"top_k": 2, "n_experts": 8}}

    summary = server.active_params_summary(FakeMoEBridge(), layers, routing)

    # 1.000 no-expertos + 4.000 de expertos; se descuentan 3/4 de los expertos.
    assert summary["total_params"] == 5000
    assert summary["active_params"] == 2000
    assert summary["active_fraction"] == 0.4
    assert summary["expert_params_total"] == 4000
