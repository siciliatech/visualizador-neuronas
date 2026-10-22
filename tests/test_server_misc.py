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
