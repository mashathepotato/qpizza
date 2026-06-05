from pathlib import Path
from triage.rubric import AdvantageRecord
from triage.dashboard import render

def _rec(method):
    return AdvantageRecord(
        method=method, candidate="D", config_id=f"{method}0",
        quantum_metric=0.95, classical_metric=0.9, metric_name="auc",
        advantage_direction="win", advantage_magnitude=0.05,
        scaling_signature=4.0, quantum_native_litmus=True,
        sim_runnable=True, q50_faithful_runnable=True,
        demo_naturalness=0.95, op_business_fit=0.95, notes="x")

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
