from pathlib import Path
from triage.rubric import AdvantageRecord
from triage.digest import write_report


def _rec(method, cand, direction, q, c, name):
    return AdvantageRecord(
        method=method, candidate=cand, config_id=f"{cand}0",
        quantum_metric=q, classical_metric=c, metric_name=name,
        advantage_direction=direction, advantage_magnitude=1.0,
        scaling_signature=1.0, quantum_native_litmus=True,
        sim_runnable=True, q50_faithful_runnable=(method != "qae"),
        demo_naturalness=0.9 if method == "fraud_qml" else 0.4,
        op_business_fit=0.9, notes="")


def test_report_names_a_winner_and_lists_all_methods(tmp_path):
    recs = [_rec("qae", "B", "win", 10, 100, "samples_to_eps"),
            _rec("fraud_qml", "D", "win", 0.95, 0.9, "auc")]
    out = tmp_path / "REPORT.md"
    write_report(recs, out)
    text = out.read_text()
    assert "# Triage report" in text
    assert "Recommendation" in text
    assert "fraud_qml" in text and "qae" in text


def test_report_handles_empty_ledger(tmp_path):
    out = tmp_path / "REPORT.md"
    write_report([], out)
    assert "no records" in out.read_text().lower()
