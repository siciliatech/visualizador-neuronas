"""Tests de construcción/truncado de input_ids, con un tokenizer falso."""

from types import SimpleNamespace

import torch

import server


class FakeTokenizer:
    """Tokeniza cada carácter a un id estable, sin descargar nada."""

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.calls = []
        self.chat_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.chat_calls.append((messages, tokenize, add_generation_prompt))
        return "<chat>" + messages[0]["content"] + "</chat>"

    def __call__(self, text, return_tensors=None):
        self.calls.append(text)
        ids = [ord(char) % 90 + 1 for char in text]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def test_prompt_normal_conserva_los_ids():
    tokenizer = FakeTokenizer()
    ids = server.build_input_ids(tokenizer, "hola", chat=False)
    assert ids.tolist() == [[ord(char) % 90 + 1 for char in "hola"]]


def test_prompt_largo_conserva_los_ultimos_tokens():
    tokenizer = FakeTokenizer()
    prompt = "a" * (server.MAX_INPUT_TOKENS + 50)
    ids = server.build_input_ids(tokenizer, prompt, chat=False)
    assert ids.shape[-1] == server.MAX_INPUT_TOKENS
    assert ids[0, -1].item() == ord("a") % 90 + 1


def test_chat_usa_el_template():
    tokenizer = FakeTokenizer(chat_template="plantilla")
    ids = server.build_input_ids(tokenizer, "hola", chat=True)
    assert tokenizer.chat_calls
    assert tokenizer.chat_calls[0][0] == [{"role": "user", "content": "hola"}]
    assert tokenizer.calls[0] == "<chat>hola</chat>"
    assert ids.shape[-1] == len("<chat>hola</chat>")


def test_chat_sin_template_usa_el_prompt_crudo():
    tokenizer = FakeTokenizer(chat_template=None)
    server.build_input_ids(tokenizer, "hola", chat=True)
    assert tokenizer.calls == ["hola"]
    assert tokenizer.chat_calls == []


def test_truncado_se_aplica_despues_del_template():
    tokenizer = FakeTokenizer(chat_template="plantilla")
    prompt = "b" * (server.MAX_INPUT_TOKENS + 20)
    ids = server.build_input_ids(tokenizer, prompt, chat=True)
    assert ids.shape[-1] == server.MAX_INPUT_TOKENS
