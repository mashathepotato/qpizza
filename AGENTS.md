# AGENTS.md — brief for any AI agent or teammate working in this repo

**Project:** Junction Helsinki × OP Pohjola hackathon — Quantum Computing for Finance.
**Team:** 2 CS + 2 physics. **Submission:** tomorrow 10:00 AM.

## Chosen direction (LOCKED): "The Madness of People Is Quantum"
We model investor irrationality with **quantum probability** (the field of *quantum cognition*).
A quantum-circuit "investor" reproduces a real **question-order effect** that a classical
Bayesian model provably cannot, and satisfies the parameter-free **QQ-equality** (PNAS 2014).

## ⚠️ Hard guardrail — keep in all external-facing material
We do **NOT** claim the brain is a quantum computer (no "quantum consciousness"). We claim human
judgment violates *classical* probability exactly as *quantum* probability predicts. Quantum-*like*
math, not quantum neurons.

## Start here
1. **[`context/VISION.md`](./context/VISION.md)** — single source of truth: pitch, science,
   build status, task board, conventions. **Read this before doing anything.**
2. **[`context/quantum-investor.html`](./context/quantum-investor.html)** — visual explainer +
   full 24-hour plan. Open in a browser.
3. **[`/quantum_investor`](./quantum_investor)** — working, verified code.

## Run the code
```bash
cd quantum_investor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py        # the demo
python test_model.py  # self-checks
```

## Working conventions
- Source of truth is `context/VISION.md`; if direction/plan changes, update it + its task board.
- Keep `/quantum_investor` runnable: `python main.py` and `python test_model.py` must pass before pushing.
- Don't commit `.venv/` or `__pycache__/`.
- Small focused commits to `main`. Agents: end commit messages with a Co-Authored-By line.
- Label any results on synthetic data as such — the win is REAL human data (see VISION §8).
