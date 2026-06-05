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
    # epsilon=0 -> ValueError inside qae.run -> must be caught, not fatal.
    spec = {"qae": [{"config_id": "bad", "candidate": "B", "p": 0.3,
                     "epsilon": 0.0, "backend": "local_aer"}],
            "qaoa": [{"config_id": "a0", "candidate": "A", "n_assets": 4,
                      "k": 2, "reps": 1, "seed": 0, "backend": "local_aer"}]}
    out = tmp_path / "records.jsonl"
    run_sweep(spec, ledger=out, plots_dir=tmp_path / "plots",
              dashboard=tmp_path / "dash.html")
    recs = load_records(out)
    methods = [r.method for r in recs]
    # the failing qae config produced NO record...
    assert "qae" not in methods
    # ...but the good qaoa config still ran (isolation worked)
    assert "qaoa" in methods

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
