# Overnight runbook

## Launch (terminal, not launchd — macOS TCC on ~/Documents)
    cd /Users/masha/Documents/qhack/.claude/worktrees/triage-lab
    caffeinate -i uv run python -m triage.orchestrator --sweep sweeps/all.yaml --grid all

## Watch
Open `triage/dashboard.html` in a browser (auto-refreshes every 30s).

## Morning
- `triage/REPORT.md` — ranked recommendation.
- `triage/dashboard.html` — per-method visuals.
- Demo: `uv run streamlit run demo/app.py`

## Resume after a crash
Re-run the same launch command; the checkpoint (`triage/.checkpoint.json`) skips completed configs.

## Real Q50 (on-site only)
Set IQM_SERVER_URL + IQM_TOKEN, then switch a config's `backend: q50_hw`.
Never used overnight.

## Real fraud data (optional)
Drop the ULB Kaggle `creditcard.csv` at `data/raw/creditcard.csv`; the fraud
harness auto-loads it instead of the synthetic stand-in.
