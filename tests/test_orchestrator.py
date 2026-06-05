import json
from pathlib import Path
from triage.orchestrator import run_sweep, load_records

def test_run_sweep_writes_one_record_per_config(tmp_path):
    spec = {"qae": [{"config_id": "b0", "candidate": "B", "p": 0.3,
                     "epsilon": 0.1, "backend": "local_aer"}]}
    out = tmp_path / "records.jsonl"
    run_sweep(spec, ledger=out, plots_dir=tmp_path / "plots",
              dashboard=tmp_path / "dash.html")
    recs = load_records(out)
    assert len(recs) == 1 and recs[0].method == "qae"

def test_failing_config_is_logged_not_fatal(tmp_path):
    spec = {"qae": [{"config_id": "bad"}],  # missing required keys -> harness errors
            "qaoa": [{"config_id": "a0", "candidate": "A", "n_assets": 4,
                      "k": 2, "reps": 1, "seed": 0, "backend": "local_aer"}]}
    out = tmp_path / "records.jsonl"
    run_sweep(spec, ledger=out, plots_dir=tmp_path / "plots",
              dashboard=tmp_path / "dash.html")
    recs = load_records(out)
    assert any(r.method == "qaoa" for r in recs)

def test_checkpoint_skips_completed_configs(tmp_path):
    spec = {"qaoa": [{"config_id": "a0", "candidate": "A", "n_assets": 4,
                      "k": 2, "reps": 1, "seed": 0, "backend": "local_aer"}]}
    out = tmp_path / "records.jsonl"
    ckpt = tmp_path / "ckpt.json"
    run_sweep(spec, ledger=out, plots_dir=tmp_path / "plots",
              dashboard=tmp_path / "dash.html", checkpoint=ckpt)
    run_sweep(spec, ledger=out, plots_dir=tmp_path / "plots",
              dashboard=tmp_path / "dash.html", checkpoint=ckpt)
    assert len(load_records(out)) == 1
