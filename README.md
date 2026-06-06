# qpizza — "The Madness of People Is Quantum"

Junction Helsinki × OP Pohjola hackathon — **Quantum Computing for Finance**.

> Newton couldn't predict "the madness of people"; Feynman said classical intuition fails on the
> quantum world. They're the same statement: **markets are non-classical.** So we compute *with*
> that: a **quantum option pricer** that loads every price path into a superposition and reads the
> fair price off it (QNDM + QAE/QSVT) — quadratic speed-up over Monte Carlo, no costly loading
> oracle. The quantum-cognition "investor" (a quantum model beating a classical one on a real
> behavioral effect) is the motivation that justifies the lens.

## Where to look
- **[`AGENTS.md`](./AGENTS.md)** — brief for agents/teammates (start here if you're an AI agent).
- **[`context/VISION.md`](./context/VISION.md)** — single source of truth: pitch, science, build
  status, task board.
- **[`context/quantum-investor.html`](./context/quantum-investor.html)** — visual explainer + the
  24-hour plan (open in a browser).
- **[`paper/main_V2.tex`](./paper/main_V2.tex)** — the full quantum option-pricing derivation (technical core).
- **[`quantum_investor/`](./quantum_investor)** — the motivation demo (cognition investor), verified (`python main.py`).

## Quick run
```bash
cd quantum_investor && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python main.py
```

**Guardrail:** quantum-*like* markets (quantum cognition is the motivation), real quantum
algorithms for pricing — **not** a claim that the brain is a quantum computer.
