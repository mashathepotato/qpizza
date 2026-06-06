from pathlib import Path
from triage.rubric import AdvantageRecord
from triage.dashboard import render

def _rec(method, **kw):
    base = dict(
        method=method, candidate="D", config_id=f"{method}0",
        quantum_metric=0.95, classical_metric=0.9, metric_name="auc",
        advantage_direction="win", advantage_magnitude=0.05,
        scaling_signature=4.0, quantum_native_litmus=True,
        sim_runnable=True, q50_faithful_runnable=True,
        demo_naturalness=0.95, op_business_fit=0.95, notes="x")
    base.update(kw)
    return AdvantageRecord(**base)

def test_render_writes_self_contained_html(tmp_path):
    out = tmp_path / "dash.html"
    render([_rec("fraud_qml"), _rec("qae")], tmp_path / "plots", out, 2, 5)
    html = out.read_text()
    assert "<html" in html.lower()
    assert "http-equiv=\"refresh\"" in html
    assert "fraud_qml" in html and "qae" in html
    assert "2/5" in html
    assert "data:image/png;base64," in html

def test_render_handles_empty(tmp_path):
    out = tmp_path / "dash.html"
    render([], tmp_path / "plots", out, 0, 3)
    assert "Waiting" in out.read_text()

def test_dashboard_shows_scaling_curve_with_multiple_records(tmp_path):
    recs = [
        _rec("qae", config_id="qae0", metric_name="samples_to_eps",
             sweep_label="epsilon", sweep_value=0.1,
             quantum_metric=100.0, classical_metric=400.0),
        _rec("qae", config_id="qae1", metric_name="samples_to_eps",
             sweep_label="epsilon", sweep_value=0.05,
             quantum_metric=200.0, classical_metric=1600.0),
        _rec("qae", config_id="qae2", metric_name="samples_to_eps",
             sweep_label="epsilon", sweep_value=0.02,
             quantum_metric=500.0, classical_metric=10000.0),
    ]
    out = tmp_path / "dash.html"
    render(recs, tmp_path / "plots", out, 3, 3)
    html = out.read_text()
    # bar (best config) + scaling curve => more than one PNG for the qae method
    assert html.count("data:image/png;base64,") > 1
    # the scaling caption names the swept axis
    assert "scaling across epsilon" in html


def test_dashboard_has_explanatory_content(tmp_path):
    """The dashboard must contain how-to-read panel, good-result copy,
    score-formula reference, and glossary terms."""
    recs = [
        _rec("qae", metric_name="samples_to_eps",
             quantum_metric=100.0, classical_metric=400.0,
             advantage_direction="win"),
        _rec("fraud_qml", metric_name="auc",
             quantum_metric=0.95, classical_metric=0.90,
             advantage_direction="win"),
    ]
    out = tmp_path / "dash.html"
    render(recs, tmp_path / "plots", out, 2, 5)
    html = out.read_text()

    # How-to-read panel
    assert "adversarial" in html.lower() or "How to read" in html

    # Good-result copy present for at least one method
    assert "Good result" in html

    # Score-formula mention (measured-advantage weight)
    assert "0.45" in html

    # Glossary terms
    assert "approx_ratio" in html
    assert "samples_to_eps" in html
    assert "auc" in html or "ROC-AUC" in html


def test_card_interprets_result_in_plain_language(tmp_path):
    """For a qae win with quantum=10, classical=100, samples_to_eps metric,
    the ratio 100/10 = 10x must appear in the rendered card."""
    rec = _rec(
        "qae",
        metric_name="samples_to_eps",
        quantum_metric=10.0,
        classical_metric=100.0,
        advantage_direction="win",
    )
    out = tmp_path / "dash.html"
    render([rec], tmp_path / "plots", out, 1, 3)
    html = out.read_text()
    # The plain-language interpretation line must mention "10x fewer" or "10.0x fewer"
    assert "10x fewer" in html or "10.0x fewer" in html
