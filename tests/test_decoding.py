"""Tests de decodificación incremental con un tokenizer byte-level falso."""

import server


class ByteTokenizer:
    """Simula GPT-2: cada id aporta bytes y decode() reemplaza lo incompleto."""

    def __init__(self, chunks):
        self.chunks = chunks

    def decode(self, ids, skip_special_tokens=True):
        data = b"".join(self.chunks.get(i, b"") for i in ids)
        return data.decode("utf-8", errors="replace")


def make_tokenizer():
    return ByteTokenizer(
        {
            1: b"\xc2",              # primera mitad de "¿"
            2: b"\xbf",              # segunda mitad de "¿"
            3: b"Qu",
            4: b"\xc3\xa9",          # "é"
            5: b"",                  # token especial
            6: b" \xf0\x9f",         # espacio + inicio del emoji
            7: b"\x98\x80",          # resto del emoji
            8: b"ok",
        }
    )


def test_safe_decode_elimina_reemplazos_de_bytes_incompletos():
    tok = make_tokenizer()
    assert tok.decode([1]) == "\ufffd"
    assert server.safe_decode(tok, [1]) == ""
    assert server.safe_decode(tok, [1, 2]) == "¿"
    assert server.safe_decode(tok, [6]) == " "
    assert server.safe_decode(tok, [6, 7]) == " 😀"


def test_token_labels_asigna_el_caracter_al_token_que_lo_completa():
    tok = make_tokenizer()
    assert server.token_labels(tok, [1, 2, 3, 4]) == ["", "¿", "Qu", "é"]


def test_token_labels_deja_vacio_el_token_especial():
    tok = make_tokenizer()
    assert server.token_labels(tok, [3, 5, 4]) == ["Qu", "", "é"]


def test_token_labels_con_emoji_partido():
    tok = make_tokenizer()
    assert server.token_labels(tok, [6, 7]) == [" ", "😀"]


def test_decode_delta_acumula_sin_duplicar():
    tok = make_tokenizer()
    delta, full = server.decode_delta(tok, [1], "")
    assert (delta, full) == ("", "")
    delta, full = server.decode_delta(tok, [1, 2], full)
    assert (delta, full) == ("¿", "¿")
    delta, full = server.decode_delta(tok, [1, 2, 3], full)
    assert (delta, full) == ("Qu", "¿Qu")


def test_update_decode_state_crece_de_uno_en_uno():
    tok = make_tokenizer()
    state = server.update_decode_state(None, tok, [1])
    assert state["labels"] == [""]
    assert state["text"] == ""

    same = server.update_decode_state(state, tok, [1, 2])
    assert same is state
    assert state["labels"] == ["", "¿"]
    assert state["text"] == "¿"

    server.update_decode_state(state, tok, [1, 2, 3])
    assert state["labels"] == ["", "¿", "Qu"]
    assert state["text"] == "¿Qu"


def test_update_decode_state_reutiliza_el_mismo_contexto():
    tok = make_tokenizer()
    state = server.update_decode_state(None, tok, [1, 2])
    same = server.update_decode_state(state, tok, [1, 2])
    assert same is state
    assert state["labels"] == ["", "¿"]


def test_update_decode_state_recalcula_si_cambia_el_contexto():
    tok = make_tokenizer()
    state = server.update_decode_state(None, tok, [1, 2])
    nuevo = server.update_decode_state(state, tok, [3, 4])
    assert nuevo is not state
    assert nuevo["labels"] == ["Qu", "é"]
    assert nuevo["ids"] == [3, 4]


def test_update_decode_state_recalcula_si_el_texto_no_es_prefijo():
    tok = make_tokenizer()
    state = {"ids": [1], "labels": [""], "text": "texto-invalido"}
    nuevo = server.update_decode_state(state, tok, [1, 2])
    assert nuevo["labels"] == ["", "¿"]
    assert nuevo["text"] == "¿"


def test_update_decode_state_con_lista_vacia():
    tok = make_tokenizer()
    assert server.token_labels(tok, []) == []
    state = server.update_decode_state(None, tok, [])
    assert state["labels"] == []
    assert state["text"] == ""
