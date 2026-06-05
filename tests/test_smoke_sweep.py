from pathlib import Path
import yaml
from triage.orchestrator import run_sweep, load_records

def test_smoke_sweep_runs_end_to_end(tmp_path):
    spec = yaml.safe_load(Path("sweeps/all.yaml").read_text())["smoke"]
    ledger = tmp_path / "rec.jsonl"
    run_sweep(spec, ledger, tmp_path / "plots", tmp_path / "dash.html",
              tmp_path / "ckpt.json")
    recs = load_records(ledger)
    methods = {r.method for r in recs}
    assert methods == {"qae", "qaoa", "fraud_qml"}
    assert (tmp_path / "dash.html").exists()
