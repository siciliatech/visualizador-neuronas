"""Tests de las funciones puras de muestreo de `server.py` (sin cargar modelos)."""

import torch

import server


def logits(*values: float) -> torch.Tensor:
    return torch.tensor([list(values)], dtype=torch.float32)


def test_greedy_devuelve_el_argmax():
    params = server.SamplingParams(mode="greedy")
    assert server.sample_next_token(logits(0.1, 2.0, 0.5), [], params) == 1


def test_temperatura_cero_equivale_a_greedy():
    params = server.SamplingParams(temperature=0)
    assert server.sample_next_token(logits(0.1, 2.0, 0.5), [], params) == 1


def test_top_k_uno_solo_deja_el_mas_probable():
    params = server.SamplingParams(temperature=1.0, top_k=1)
    for _ in range(10):
        assert server.sample_next_token(logits(0.1, 2.0, 0.5), [], params) == 1


def test_top_p_pequeno_solo_deja_el_mas_probable():
    params = server.SamplingParams(temperature=1.0, top_k=0, top_p=0.5)
    for _ in range(10):
        assert server.sample_next_token(logits(20.0, 0.0), [], params) == 0


def test_penalizacion_por_repeticion_cambia_el_argmax():
    # 2.0 / 2 = 1.0 < 1.5: el token visto deja de ser el más probable.
    params = server.SamplingParams(mode="greedy", repetition_penalty=2.0)
    assert server.sample_next_token(logits(2.0, 1.5), [0], params) == 1


def test_penalizacion_no_afecta_tokens_no_vistos():
    # Se penaliza el token 1 (1.5 / 2 = 0.75); el 0 sigue ganando.
    params = server.SamplingParams(mode="greedy", repetition_penalty=2.0)
    assert server.sample_next_token(logits(2.0, 1.5), [1], params) == 0


def test_ngramas_repetidos_prohiben_el_token():
    logits_tensor = torch.zeros((1, 5))
    server.ban_repeated_ngrams(logits_tensor, [1, 2, 3, 1, 2], 3)
    assert logits_tensor[0, 3] == float("-inf")


def test_ngramas_con_n_uno_no_hace_nada():
    logits_tensor = torch.zeros((1, 4))
    server.ban_repeated_ngrams(logits_tensor, [1, 2, 1, 2], 1)
    assert torch.isfinite(logits_tensor).all()


def test_logits_todos_invalidos_devuelve_un_indice_valido():
    logits_tensor = torch.full((1, 5), float("-inf"))
    params = server.SamplingParams(temperature=1.0, top_k=0, top_p=1.0)
    result = server.sample_next_token(logits_tensor, [], params)
    assert isinstance(result, int)
    assert 0 <= result < 5
