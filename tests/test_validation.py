"""Tests de validación del mensaje `generate` (Pydantic, sin cargar modelos)."""

import pytest
from pydantic import ValidationError

import server


def test_mensaje_minimo_usa_los_defaults():
    req = server.GenerateRequest.model_validate({"prompt": "hola"})
    assert req.type == "generate"
    assert req.model == server.DEFAULT_MODEL
    assert req.prompt == "hola"
    assert req.tokens == server.DEFAULT_TOKENS
    assert req.params.mode == "sample"
    assert req.params.temperature == 0.8
    assert req.params.top_k == 40
    assert req.params.top_p == 0.95
    assert req.params.repetition_penalty == 1.0
    assert req.params.no_repeat_ngram == 0


def test_mensaje_completo_valido():
    req = server.GenerateRequest.model_validate(
        {
            "type": "generate",
            "model": "gpt2",
            "prompt": "La capital de Francia es",
            "tokens": 3,
            "params": {
                "mode": "greedy",
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 0.5,
                "repetition_penalty": 1.15,
                "no_repeat_ngram": 2,
            },
        }
    )
    assert req.model == "gpt2"
    assert req.tokens == 3
    assert req.params.mode == "greedy"
    assert req.params.no_repeat_ngram == 2


def test_temperature_texto_devuelve_error_de_validacion():
    with pytest.raises(ValidationError) as info:
        server.GenerateRequest.model_validate({"prompt": "hola", "params": {"temperature": "abc"}})
    message = server.validation_message(info.value)
    assert "params.temperature" in message
    assert "number" in message
    assert "Traceback" not in message


def test_modelo_desconocido_lista_las_opciones():
    with pytest.raises(ValidationError) as info:
        server.GenerateRequest.model_validate({"model": "no-existe"})
    message = server.validation_message(info.value)
    assert "no-existe" in message
    assert "gpt2" in message
    assert "opciones válidas" in message
    assert "Value error" not in message


def test_prompt_vacio_usa_el_default():
    req = server.GenerateRequest.model_validate({"prompt": "   "})
    assert req.prompt == server.DEFAULT_PROMPT


def test_prompt_demasiado_largo_se_rechaza():
    with pytest.raises(ValidationError) as info:
        server.GenerateRequest.model_validate({"prompt": "x" * (server.MAX_PROMPT_CHARS + 1)})
    assert "prompt" in server.validation_message(info.value)


def test_params_nulo_equivale_a_defaults():
    req = server.GenerateRequest.model_validate({"params": None})
    assert req.params.temperature == 0.8
    assert req.params.mode == "sample"


def test_type_desconocido_se_rechaza():
    with pytest.raises(ValidationError) as info:
        server.GenerateRequest.model_validate({"type": "select_layer"})
    assert "type" in server.validation_message(info.value)


def test_select_layer_valido():
    req = server.SelectLayerRequest.model_validate({"type": "select_layer", "layer": 3})
    assert req.layer == 3
    assert req.type == "select_layer"


def test_select_layer_negativo_se_rechaza():
    with pytest.raises(ValidationError) as info:
        server.SelectLayerRequest.model_validate({"type": "select_layer", "layer": -1})
    assert "layer" in server.validation_message(info.value)


def test_mensaje_que_no_es_objeto_se_rechaza():
    with pytest.raises(ValidationError) as info:
        server.GenerateRequest.model_validate([])
    assert "mensaje" in server.validation_message(info.value)


@pytest.mark.parametrize(
    "payload, campo",
    [
        ({"tokens": 0}, "tokens"),
        ({"tokens": 101}, "tokens"),
        ({"params": {"temperature": -0.1}}, "params.temperature"),
        ({"params": {"temperature": 5.1}}, "params.temperature"),
        ({"params": {"top_p": 1.1}}, "params.top_p"),
        ({"params": {"top_p": -0.1}}, "params.top_p"),
        ({"params": {"repetition_penalty": 0.0}}, "params.repetition_penalty"),
        ({"params": {"repetition_penalty": 5.1}}, "params.repetition_penalty"),
        ({"params": {"no_repeat_ngram": -1}}, "params.no_repeat_ngram"),
        ({"params": {"no_repeat_ngram": 11}}, "params.no_repeat_ngram"),
        ({"params": {"mode": "otro"}}, "params.mode"),
        ({"params": {"top_k": -1}}, "params.top_k"),
    ],
)
def test_valores_fuera_de_rango(payload, campo):
    with pytest.raises(ValidationError) as info:
        server.GenerateRequest.model_validate(payload)
    assert campo in server.validation_message(info.value)


def test_ablation_es_opcional_y_se_valida():
    req = server.GenerateRequest.model_validate({"prompt": "hola"})
    assert req.ablation is None

    req = server.GenerateRequest.model_validate(
        {"prompt": "hola", "ablation": {"layer": 2, "neuron": 7}}
    )
    assert req.ablation is not None
    assert req.ablation.layer == 2
    assert req.ablation.neuron == 7


@pytest.mark.parametrize(
    "ablation, campo",
    [
        ({"layer": -1, "neuron": 0}, "layer"),
        ({"layer": 0, "neuron": -1}, "neuron"),
        ({"layer": 0}, "neuron"),
        ({"layer": 0, "neuron": 0, "otra": 1}, "otra"),
    ],
)
def test_ablation_invalida_se_rechaza(ablation, campo):
    with pytest.raises(ValidationError) as info:
        server.GenerateRequest.model_validate({"prompt": "hola", "ablation": ablation})
    assert campo in server.validation_message(info.value)


@pytest.mark.parametrize("seed", [-1, 2**31])
def test_semilla_fuera_de_rango_se_rechaza(seed):
    with pytest.raises(ValidationError) as info:
        server.GenerateRequest.model_validate({"prompt": "hola", "seed": seed})
    assert "seed" in server.validation_message(info.value)


def test_semilla_valida_y_opcional():
    assert server.GenerateRequest.model_validate({"prompt": "hola"}).seed is None
    req = server.GenerateRequest.model_validate({"prompt": "hola", "seed": 0})
    assert req.seed == 0
