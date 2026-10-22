"""Tests de `/health` y de la medición de payloads."""

import asyncio
import json

import server


def test_health_sin_modelo_no_carga_nada(monkeypatch):
    monkeypatch.setitem(server._current, "key", None)
    monkeypatch.setitem(server._current, "bridge", None)

    data = asyncio.run(server.health())

    assert data == {"status": "ok", "model_key": None, "ready": False}


def test_health_informa_modelo_listo(monkeypatch):
    monkeypatch.setitem(server._current, "key", "gpt2")
    monkeypatch.setitem(server._current, "bridge", object())

    data = asyncio.run(server.health())

    assert data == {"status": "ok", "model_key": "gpt2", "ready": True}


def test_payload_size_mide_el_json_compacto():
    payload = {"type": "step", "delta": "á", "valor": 1}
    expected = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    assert server.payload_size(payload) == expected
