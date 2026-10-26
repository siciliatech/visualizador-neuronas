"""Prueba de frontend con navegador real y WebSocket simulado.

No descarga modelos ni levanta FastAPI: sirve `index.html` desde un HTTP local,
reemplaza `WebSocket` por un doble controlable y reproduce el protocolo 2 a mano.
Se marca `ui` y se ejecuta con `pytest -q -m ui`.
"""

from __future__ import annotations

import functools
import http.server
import math
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


@pytest.mark.ui
@pytest.mark.parametrize("shape", ["brain", "chip"])
def test_forma_del_mapa_conserva_la_seleccion_de_neuronas(page, shape):
    _emit(page, _meta())
    _emit(page, _step(1.0, "a"))
    _emit(page, _done("a"))

    page.locator("#heatmap-shape").select_option(shape)
    page.locator("#heatmap").scroll_into_view_if_needed()
    point = page.evaluate(
        """() => {
          const i = 1;
          const r = heatmap.getBoundingClientRect();
          return {
            x: r.left + (layout.xs[i] + 0.5) * CELL * r.width / canvasW(heatmap),
            y: r.top + (layout.ys[i] + 0.5) * CELL * r.height / canvasH(heatmap),
          };
        }"""
    )
    assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in point.values())
    page.mouse.click(point["x"], point["y"])
    assert page.evaluate("selectedNeuron") == 1
    # Todas las neuronas tienen una celda propia y distinta.
    assert page.evaluate(
        "new Set(Array.from(layout.xs, (x, i) => x + ',' + layout.ys[i])).size"
        " === META.neurons_per_layer"
    )


@pytest.mark.ui
def test_modelo_moe_muestra_mapa_de_expertos_y_ruteo(page):
    meta = _meta()
    meta.update(
        {
            "model": "mixtral-tiny",
            "model_label": "Mixtral 8xTiny de prueba",
            "neurons_per_layer": 4,
            "architecture": "moe",
            "unit": "expert",
            "n_experts": 4,
            "router_top_k": 2,
            "routing_available": True,
            "layers_meta": [
                {
                    "layer": 0,
                    "architecture": "moe",
                    "unit": "expert",
                    "units": 4,
                    "d_mlp": 8,
                    "n_experts": 4,
                    "top_k": 2,
                }
            ],
            "active_params": {
                "total_params": 1000,
                "active_params": 500,
                "active_fraction": 0.5,
                "expert_params_total": 400,
            },
        }
    )
    _emit(page, meta)

    step = _step(0.7, "a")
    step["activations"] = {"layer_0": [0.0, 0.7, 0.3, 0.0]}
    step["top"] = {"layer_0": [[1, 0.7], [2, 0.3]]}
    step["routing"] = {
        "layer_0": {
            "n_experts": 4,
            "top_k": 2,
            "experts": [1, 2],
            "weights": [0.7, 0.3],
            "entropy": 0.88,
        }
    }
    _emit(page, step)
    _emit(page, _done("a"))

    page.wait_for_function("!document.querySelector('#moe-panel').hidden")
    assert page.locator("#chip-arch").inner_text() == "MoE"
    assert "experto" in page.locator("#heatmap-title").inner_text().lower()
    assert page.locator("#moe-grid .moe-cell.on").count() == 2
    assert page.locator("#topk-title").inner_text().lower() == "expertos con mayor peso"
    explainer = page.locator(".explainer").text_content().lower()
    assert "moe" in explainer
    assert "router" in explainer
    assert "expertos" in explainer
    moe_text = page.locator("#moe-panel").inner_text()
    assert "2 de 4" in moe_text
    assert "500" in moe_text or "0.5" in moe_text

    page.locator('#moe-grid .moe-cell[data-expert="2"]').click()
    assert page.evaluate("selectedNeuron") == 2
    assert page.locator("#d-neuron-label").inner_text() == "Experto"
    assert page.locator("#d-neuron").inner_text() == "2"
    assert page.locator("#d-value").inner_text() == "0.300"

    report = page.locator("#report").inner_text()
    assert "Ruteo MoE" in report
    assert "experto" in report.lower()


def _select_model(page, key: str, label: str) -> None:
    page.evaluate(
        """([key, label]) => {
            const select = document.querySelector('#model');
            if (!select.querySelector(`option[value="${key}"]`)) {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = label;
                select.appendChild(option);
            }
            select.value = key;
        }""",
        [key, label],
    )


def _start_capture_run(page, model: str, label: str, prompt: str) -> None:
    _select_model(page, model, label)
    page.evaluate(
        """prompt => {
            document.querySelector('#prompt').value = prompt;
            document.querySelector('#tokens').value = '2';
            document.querySelector('#capture_all').checked = true;
            document.querySelector('#mode').value = 'greedy';
            document.querySelector('#send').click();
        }""",
        prompt,
    )


def _dense_capture_meta() -> dict:
    meta = _meta()
    meta.update(
        {
            "model": "gpt2",
            "model_label": "GPT-2 de prueba",
            "prompt": "hola",
            "capture_all": True,
            "architecture": "dense",
            "unit": "neuron",
            "neurons_per_layer": 3,
            "layers_meta": [
                {
                    "layer": 0,
                    "architecture": "dense",
                    "unit": "neuron",
                    "units": 3,
                    "n_experts": None,
                    "top_k": None,
                }
            ],
            "active_params": {
                "total_params": 1000,
                "active_params": 1000,
                "active_fraction": 1.0,
                "expert_params_total": 0,
            },
        }
    )
    return meta


def _moe_capture_meta() -> dict:
    meta = _meta()
    meta.update(
        {
            "model": "mixtral-tiny",
            "model_label": "Mixtral 8xTiny de prueba",
            "prompt": "hola",
            "capture_all": True,
            "architecture": "moe",
            "unit": "expert",
            "neurons_per_layer": 4,
            "n_experts": 4,
            "router_top_k": 2,
            "routing_available": True,
            "layers_meta": [
                {
                    "layer": 0,
                    "architecture": "moe",
                    "unit": "expert",
                    "units": 4,
                    "n_experts": 4,
                    "top_k": 2,
                }
            ],
            "active_params": {
                "total_params": 2000,
                "active_params": 1000,
                "active_fraction": 0.5,
                "expert_params_total": 800,
            },
        }
    )
    return meta


def _capture_step(values, delta, top, routing=None) -> dict:
    step = _step(values[1], delta)
    step["activations"] = {"layer_0": values}
    step["top"] = {"layer_0": top}
    if routing:
        step["routing"] = {"layer_0": routing}
    return step


@pytest.mark.ui
def test_comparador_denso_vs_moe_usa_metricas_normalizadas(page):
    # Pasada A: modelo denso.
    _start_capture_run(page, "gpt2", "GPT-2 de prueba", "hola")
    _emit(page, _dense_capture_meta())
    _emit(page, _capture_step([0.1, 0.5, 0.2], "a", [[1, 0.5], [2, 0.2]]))
    _emit(page, _done("a"))
    page.locator("#save-a").click()

    page.wait_for_function("!document.querySelector('#compare-panel').hidden")
    assert page.locator("#compare-tab-a").is_enabled()
    assert not page.locator("#compare-tab-b").is_enabled()
    assert page.locator("#compare-tab-a").get_attribute("aria-selected") == "true"
    a_text = page.locator("#compare-body").inner_text()
    assert "GPT-2 de prueba" in a_text
    assert "denso" in a_text.lower()
    assert "100.0%" in a_text
    assert "Ruteo" not in a_text

    # Pasada B: modelo MoE.
    _start_capture_run(page, "mixtral-tiny", "Mixtral 8xTiny de prueba", "hola")
    _emit(page, _moe_capture_meta())
    _emit(
        page,
        _capture_step(
            [0.0, 0.6, 0.4, 0.0],
            "b",
            [[1, 0.6], [2, 0.4]],
            {
                "n_experts": 4,
                "top_k": 2,
                "experts": [1, 2],
                "weights": [0.6, 0.4],
                "entropy": 0.97,
            },
        ),
    )
    _emit(
        page,
        _capture_step(
            [0.0, 0.5, 0.0, 0.5],
            "c",
            [[1, 0.5], [3, 0.5]],
            {
                "n_experts": 4,
                "top_k": 2,
                "experts": [1, 3],
                "weights": [0.5, 0.5],
                "entropy": 1.0,
            },
        ),
    )
    _emit(page, _done("bc"))
    page.locator("#compare-runs").click()

    page.wait_for_function(
        "document.querySelector('#compare-tab-compare').getAttribute('aria-selected') === 'true'"
    )
    compare_text = page.locator("#compare-body").inner_text()
    assert "métricas normalizadas" in compare_text.lower()
    assert "Parámetros activos" in compare_text
    assert "50.0%" in compare_text
    assert "Cobertura media de expertos" in compare_text
    assert "Entropía media del ruteo" in compare_text
    assert "denso" in compare_text.lower()
    assert "MoE" in compare_text
    assert page.locator("#compare-controls").is_hidden()
    assert not page.locator("#compare-overlay").is_enabled()

    # Las pestañas A y B muestran cada pasada por separado.
    page.locator("#compare-tab-a").click()
    assert "GPT-2 de prueba" in page.locator("#compare-body").inner_text()
    page.locator("#compare-tab-b").click()
    b_text = page.locator("#compare-body").inner_text()
    assert "Mixtral 8xTiny de prueba" in b_text
    assert "cobertura media" in b_text.lower()

    # Volver a Comparar restaura la tabla normalizada.
    page.locator("#compare-tab-compare").click()
    assert "métricas normalizadas" in page.locator("#compare-body").inner_text().lower()


@pytest.mark.ui
def test_comparador_mismo_modelo_conserva_el_diff_por_neurona(page):
    _start_capture_run(page, "gpt2", "GPT-2 de prueba", "hola")
    _emit(page, _dense_capture_meta())
    _emit(page, _capture_step([0.1, 0.5, 0.2], "a", [[1, 0.5], [2, 0.2]]))
    _emit(page, _done("a"))
    page.locator("#save-a").click()

    _start_capture_run(page, "gpt2", "GPT-2 de prueba", "hola")
    _emit(page, _dense_capture_meta())
    _emit(page, _capture_step([0.1, 0.8, 0.2], "b", [[1, 0.8], [2, 0.2]]))
    _emit(page, _done("b"))
    page.locator("#compare-runs").click()

    page.wait_for_function(
        "document.querySelector('#compare-tab-compare').getAttribute('aria-selected') === 'true'"
    )
    text = page.locator("#compare-body").inner_text()
    assert "neurona" in text.lower()
    assert "Δ (B − A)" in text
    assert "métricas normalizadas" not in text.lower()
    assert page.locator("#compare-controls").is_visible()
    assert page.locator("#compare-overlay").is_enabled()

    page.locator("#compare-overlay").click()
    page.wait_for_function(
        "document.querySelector('#heatmap-title').textContent.includes('Diferencia B − A')"
    )
    assert page.locator("#compare-overlay").inner_text() == "Ver heatmap actual"

    page.locator("#compare-tab-a").click()
    assert page.locator("#compare-controls").is_hidden()
    assert "GPT-2 de prueba" in page.locator("#compare-body").inner_text()

    page.locator("#compare-tab-compare").click()
    assert "Δ (B − A)" in page.locator("#compare-body").inner_text()


@pytest.mark.ui
def test_inicio_ofrece_recorridos_y_prepara_comparacion(page):
    assert page.locator("#intent-title").inner_text().lower() == "¿qué querés hacer?"
    assert page.locator('.intent-option[data-intent="explore"]').is_visible()
    assert page.locator('.intent-option[data-intent="compare"]').is_visible()
    assert "comparar" in page.locator('.intent-option[data-intent="compare"]').inner_text().lower()
    assert not page.locator("#capture_all").is_checked()

    # Los detalles arrancan colapsados y se abren con «Ver más».
    for intent in ("explore", "compare"):
        details = page.locator(f'.intent-option[data-intent="{intent}"] details')
        body = page.locator(f'.intent-option[data-intent="{intent}"] .intent-body')
        assert not details.evaluate("el => el.open")
        assert not body.is_visible()
        page.locator(f'.intent-option[data-intent="{intent}"] summary').click()
        assert details.evaluate("el => el.open")
        assert body.is_visible()
        page.locator(f'.intent-option[data-intent="{intent}"] summary').click()
        assert not details.evaluate("el => el.open")

    page.locator("#intent-compare").click()
    assert page.locator("#capture_all").is_checked()
    assert page.locator("#intent-status").is_visible()
    assert "todas las capas" in page.locator("#intent-status").inner_text().lower()
    assert page.locator("#intent-compare").get_attribute("aria-pressed") == "true"
    assert page.locator("#intent-explore").get_attribute("aria-pressed") == "false"

    page.locator("#intent-explore").click()
    assert not page.locator("#capture_all").is_checked()
    assert "exploración" in page.locator("#intent-status").inner_text().lower()
    assert page.locator("#intent-explore").get_attribute("aria-pressed") == "true"


@pytest.mark.ui
def test_atencion_incluye_lectura_en_lenguaje_llano(page):
    meta = _meta()
    meta["attention"] = {"available": True, "heads": 1, "top_k": 8}
    _emit(page, meta)

    step = _step(1.0, "a")
    step["attention"] = {
        "tokens": ["hola", "mundo"],
        "layers": {"layer_0": [[[0, 0.7], [1, 0.3]]]},
    }
    _emit(page, step)

    page.wait_for_function("document.querySelector('#attn-reading')?.textContent.includes('hola')")
    reading = page.locator("#attn-reading").inner_text()
    assert "«hola» (70 %)" in reading
    assert "«mundo»" in reading
