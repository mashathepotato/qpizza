"""Sweep orchestrator: run each config, append to the JSONL ledger, refresh the
dashboard, checkpoint. Catches per-config failures so the night never crashes."""
from __future__ import annotations
import argparse
import json
import traceback
from pathlib import Path

import yaml

from triage.rubric import AdvantageRecord
from triage.harness import qae, qaoa, fraud_qml

_HARNESS = {"qae": qae.run, "qaoa": qaoa.run, "fraud_qml": fraud_qml.run}


def load_records(ledger: Path) -> list[AdvantageRecord]:
    ledger = Path(ledger)
    if not ledger.exists():
        return []
    out = []
    for line in ledger.read_text().splitlines():
        if line.strip():
            out.append(AdvantageRecord.from_dict(json.loads(line)))
    return out


def _load_checkpoint(ckpt: Path | None) -> set[str]:
    if ckpt and Path(ckpt).exists():
        return set(json.loads(Path(ckpt).read_text()))
    return set()


def _save_checkpoint(ckpt: Path | None, done: set[str]) -> None:
    if ckpt:
        Path(ckpt).write_text(json.dumps(sorted(done)))


def run_sweep(spec: dict, ledger: Path, plots_dir: Path, dashboard: Path,
              checkpoint: Path | None = None) -> None:
    ledger = Path(ledger); plots_dir = Path(plots_dir); dashboard = Path(dashboard)
    plots_dir.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(checkpoint)
    total = sum(len(v) for v in spec.values())
    completed = len(done)
    for method, configs in spec.items():
        harness = _HARNESS.get(method)
        if harness is None:
            print(f"[skip] unknown method {method}")
            continue
        for cfg in configs:
            cid = cfg.get("config_id", f"{method}_{completed}")
            if cid in done:
                continue
            try:
                rec = harness(cfg)
                with ledger.open("a") as fh:
                    fh.write(json.dumps(rec.to_dict()) + "\n")
            except Exception:
                print(f"[fail] {method}/{cid}\n{traceback.format_exc()}")
            finally:
                done.add(cid)
                completed += 1
                _save_checkpoint(checkpoint, done)
                _refresh_dashboard(ledger, plots_dir, dashboard, completed, total)


def _refresh_dashboard(ledger, plots_dir, dashboard, completed, total):
    from triage.dashboard import render
    render(load_records(ledger), plots_dir, dashboard, completed, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--grid", default="all", help="top-level key set: 'all' or 'smoke'")
    ap.add_argument("--ledger", default="triage/records.jsonl")
    ap.add_argument("--plots", default="triage/plots")
    ap.add_argument("--dashboard", default="triage/dashboard.html")
    ap.add_argument("--checkpoint", default="triage/.checkpoint.json")
    ap.add_argument("--report", default="triage/REPORT.md")
    args = ap.parse_args()
    raw = yaml.safe_load(Path(args.sweep).read_text())
    spec = raw[args.grid] if args.grid in raw else raw
    run_sweep(spec, Path(args.ledger), Path(args.plots), Path(args.dashboard),
              Path(args.checkpoint))
    from triage.digest import write_report
    write_report(load_records(Path(args.ledger)), Path(args.report))
    print(f"Done. Report at {args.report}")


if __name__ == "__main__":
    main()
