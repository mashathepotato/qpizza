# qpizza — "The Madness of People Is Quantum"

Junction Helsinki × OP Pohjola hackathon — **Quantum Computing for Finance**.

> Newton couldn't predict "the madness of people"; Feynman said classical intuition fails on the
> quantum world. They're the same statement: **human financial decision-making obeys *quantum*
> probability, not classical.** We model investor irrationality as a quantum circuit and reproduce
> behavioral effects a classical model provably can't.

## Where to look
- **[`AGENTS.md`](./AGENTS.md)** — brief for agents/teammates (start here if you're an AI agent).
- **[`context/VISION.md`](./context/VISION.md)** — single source of truth: pitch, science, build
  status, task board.
- **[`context/quantum-investor.html`](./context/quantum-investor.html)** — visual explainer + the
  24-hour plan (open in a browser).
- **[`quantum_investor/`](./quantum_investor)** — working, verified code (`python main.py`).

## Quick run
```bash
cd quantum_investor && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python main.py
```

**Guardrail:** quantum-*like* math (quantum cognition), **not** a claim that the brain is a
quantum computer.
