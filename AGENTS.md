# AGENTS.md — brief for any AI agent or teammate working in this repo

**Project:** Junction Helsinki × OP Pohjola hackathon — Quantum Computing for Finance.
**Team:** 2 CS + 2 physics. **Submission:** tomorrow 10:00 AM.

## Chosen direction (LOCKED): "The Madness of People Is Quantum" → quantum option pricing
Two layers:
- **Motivation —** markets are non-classical (Newton's "madness of people"; Feynman). Human
  financial decisions follow *quantum* probability (the field of *quantum cognition*). A quantum
  "investor" reproducing a real question-order effect is our opener.
- **Technical core —** because the market is quantum-like, we **compute with it**: a **quantum
  option pricer** that loads all `2^M` price paths into a superposition (free, exact, `O(M)`) and
  reads the fair price off it via **QNDM** phase encoding + **QAE/QSVT** — quadratic speed-up over
  Monte Carlo, no costly loading oracle. Full math in [`paper/main_V2.tex`](./paper/main_V2.tex).

## ⚠️ Hard guardrail — keep in all external-facing material
We do **NOT** claim the brain is a quantum computer (no "quantum consciousness"). We claim human
judgment violates *classical* probability exactly as *quantum* probability predicts. Quantum-*like*
markets, real quantum algorithms.

## Start here
1. **[`context/VISION.md`](./context/VISION.md)** — single source of truth: pitch, science,
   build status, task board, conventions. **Read this before doing anything.**
2. **[`paper/main_V2.tex`](./paper/main_V2.tex)** — the full option-pricing derivation (the technical core).
3. **[`context/quantum-investor.html`](./context/quantum-investor.html)** — visual explainer +
   full 24-hour plan. Open in a browser.
4. **[`/quantum_investor`](./quantum_investor)** — the motivation demo (cognition investor), verified.

## Run the code (motivation demo; pricer is the build target)
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
