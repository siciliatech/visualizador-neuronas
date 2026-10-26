"""Tests del spike MoE: ruteo y estimación de cómputo activo, sin descargar modelos."""

import torch

from scripts import spike_moe


def test_routing_summary_normaliza_pesos_dispersos_y_cuenta_carga():
    indices = torch.tensor([[0, 1], [1, 2]])
    weights = torch.tensor(
        [
            [0.6, 0.4, 0.0],
            [0.0, 0.3, 0.7],
        ]
    )
    cache = {
        "blocks.0.mlp.hook_expert_indices": indices,
        "blocks.0.mlp.hook_expert_weights": weights,
    }

    summary = spike_moe.routing_summary(cache, n_layers=1, n_experts_hint=3)

    layer = summary["layer_0"]
    assert layer["n_experts"] == 3
    assert layer["top_k"] == 2
    assert layer["last_token_experts"] == [1, 2]
    assert layer["last_token_weights"] == [0.3, 0.7]
    assert layer["dead_experts"] == []
    assert layer["max_expert_share"] == 0.5


def test_routing_summary_acepta_pesos_ya_recortados_a_top_k():
    indices = torch.tensor([[2, 0]])
    weights = torch.tensor([[0.8, 0.2]])
    cache = {
        "blocks.0.mlp.hook_expert_indices": indices,
        "blocks.0.mlp.hook_expert_weights": weights,
    }

    summary = spike_moe.routing_summary(cache, n_layers=1, n_experts_hint=4)

    assert summary["layer_0"]["last_token_weights"] == [0.8, 0.2]


def test_active_params_descuenta_expertos_no_elegidos():
    routing = {
        "layer_0": {"top_k": 2, "n_experts": 8},
        "layer_1": {"top_k": 4, "n_experts": 8},
    }
    active = spike_moe.active_params_estimate(
        total_params=10_000,
        expert_params={0: 4_000, 1: 4_000},
        routing=routing,
    )

    # Capa 0: 75 % de 4.000 = 3.000 inactivos; capa 1: 50 % de 4.000 = 2.000.
    assert active["active_params_estimate"] == 5_000
    assert active["active_fraction"] == 0.5
    assert active["expert_params_total"] == 8_000


def test_active_params_sin_ruteo_devuelve_none():
    assert spike_moe.active_params_estimate(100, {}, {}) is None
