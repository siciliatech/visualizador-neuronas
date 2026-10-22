"""Prueba de frontend con navegador real y WebSocket simulado.

No descarga modelos ni levanta FastAPI: sirve `index.html` desde un HTTP local,
reemplaza `WebSocket` por un doble controlable y reproduce el protocolo 2 a mano.
Se marca `ui` y se ejecuta con `pytest -q -m ui`.
"""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format, *_args):
        pass


@pytest.fixture()
def page():
    playwright = pytest.importorskip("playwright.sync_api")
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        functools.partial(_QuietHandler, directory=str(ROOT)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as exc:
                server.shutdown()
                pytest.skip(f"Chrome/Chromium no disponible para la prueba UI: {exc}")
        try:
            page = browser.new_page()
            page.add_init_script(
                """
                window.__sent = [];
                class FakeWebSocket {
                  static OPEN = 1;
                  static CLOSED = 3;
                  constructor(url) {
                    this.url = url;
                    this.readyState = 0;
                    window.__ws = this;
                    setTimeout(() => {
                      this.readyState = 1;
                      if (this.onopen) this.onopen();
                    }, 0);
                  }
                  send(data) { window.__sent.push(JSON.parse(data)); }
                  close() {
                    this.readyState = FakeWebSocket.CLOSED;
                    if (this.onclose) this.onclose();
                  }
                  emit(data) {
                    if (this.onmessage) this.onmessage({ data: JSON.stringify(data) });
                  }
                }
                window.WebSocket = FakeWebSocket;
                """
            )
            host, port = server.server_address
            page.goto(f"http://{host}:{port}/index.html")
            page.wait_for_function(
                "document.querySelector('#status')?.textContent.includes('Conectado')"
            )
            yield page
        finally:
            browser.close()
            server.shutdown()


def _emit(page, payload: dict) -> None:
    page.evaluate("payload => window.__ws.emit(payload)", payload)


def _meta(ablation=None) -> dict:
    return {
        "type": "meta",
        "protocol": 2,
        "model": "gpt2",
        "model_label": "GPT-2 de prueba",
        "chat": False,
        "prompt": "hola",
        "layers": 1,
        "selected_layer": 0,
        "layer": 0,
        "neurons_per_layer": 3,
        "top_k": 2,
        "capture_all": False,
        "seed": 123,
        "ablation": ablation,
        "attention": {"available": True, "heads": 1, "top_k": 1},
    }


def _step(activation: float, delta: str, neuron: int = 1) -> dict:
    values = [0.1, activation, 0.2]
    return {
        "type": "step",
        "layer": 0,
        "delta": delta,
        "token": delta,
        "activations": {"layer_0": values},
        "stats": {
            "max_abs": {"value": abs(activation), "layer": 0, "neuron": neuron},
            "max_signed": {"value": activation, "layer": 0, "neuron": neuron},
            "layer_means": {"layer_0": 0.4},
        },
        "top": {"layer_0": [[neuron, activation], [0, 0.1]]},
        "attention": {"tokens": [delta], "layers": {"layer_0": [[[0, 1.0]]]}},
    }


def _done(text: str) -> dict:
    return {
        "type": "done",
        "tokens": 1,
        "stopped": "limit",
        "text": text,
        "generated_text": text,
    }


@pytest.mark.ui
def test_ablation_reutiliza_la_semilla_y_actualiza_el_detalle(page):
    page.evaluate(
        """
        const select = document.querySelector('#model');
        const option = document.createElement('option');
        option.value = 'gpt2';
        option.textContent = 'GPT-2 de prueba';
        select.appendChild(option);
        select.value = 'gpt2';
        document.querySelector('#prompt').value = 'hola';
        document.querySelector('#tokens').value = '2';
        document.querySelector('#mode').value = 'sample';
        document.querySelector('#send').click();
        """
    )
    page.wait_for_function("window.__sent.length === 1")

    first = page.evaluate("window.__sent[0]")
    assert first["prompt"] == "hola"
    assert first["seed"] is None
    assert first["ablation"] is None

    _emit(page, _meta())
    _emit(page, _step(2.0, "a"))
    _emit(page, _done("a"))
    page.wait_for_function("document.querySelector('#stream').textContent === 'a'")

    page.locator("#topk .topk-row").first.click()
    page.wait_for_function("!document.querySelector('#ablate').disabled")
    page.click("#ablate")
    page.wait_for_function("window.__sent.length === 2")

    second = page.evaluate("window.__sent[1]")
    assert second["seed"] == 123
    assert second["ablation"] == {"layer": 0, "neuron": 1}
    assert second["prompt"] == "hola"

    _emit(page, _meta(ablation={"layer": 0, "neuron": 1}))
    _emit(page, _step(0.0, "b"))
    _emit(page, _done("b"))

    page.wait_for_function(
        "document.querySelector('#ablation-after-meta').textContent === 'texto distinto'"
    )
    assert page.locator("#ablation-panel").is_visible()
    assert page.locator("#ablation-before").inner_text() == "a"
    assert page.locator("#ablation-after").inner_text() == "b"
    assert "Semilla 123 reutilizada" in page.locator("#ablation-summary").inner_text()
    assert page.locator("#d-value").inner_text() == "0.000"

    report = page.locator("#report").inner_text()
    assert "Ablación activa" in report
    assert "semilla=123" in report


@pytest.mark.ui
def test_escala_por_capa_mejora_visibilidad_sin_cambiar_escala_global(page):
    meta = _meta()
    meta["layers"] = 2
    _emit(page, meta)

    step = _step(2.0, "a")
    step["stats"]["max_abs"] = {"value": 100.0, "layer": 1, "neuron": 0}
    _emit(page, step)
    _emit(page, _done("a"))

    page.wait_for_function(
        "document.querySelector('#heatmap-scale')?.textContent === 'global · 100.00'"
    )
    page.locator("#layer-scale").check()
    page.wait_for_function(
        "document.querySelector('#heatmap-scale')?.textContent === 'capa 0 · 2.00'"
    )
    assert page.locator("#layer-scale").is_checked()


@pytest.mark.ui
def test_iconos_de_ayuda_abren_y_cierran_el_popover(page):
    pop = page.locator("#help-pop")
    assert pop.is_hidden()

    page.locator('[data-help="top_k"]').click()
    assert pop.is_visible()
    assert "Top-k" in pop.inner_text()
    assert page.locator('[data-help="top_k"]').get_attribute("aria-expanded") == "true"

    page.keyboard.press("Escape")
    assert pop.is_hidden()

    page.locator('[data-help="mode"]').click()
    page.locator("h1").click()
    assert pop.is_hidden()
