# Resources & references

## Quantum frameworks (OP-recommended)
- **Qiskit** (IBM) — richest finance tooling. `qiskit-finance`, `qiskit-optimization`,
  `qiskit-algorithms`. Has QAOA, VQE, Amplitude Estimation, QUBO converters out of the box.
- **PennyLane** (Xanadu) — best for quantum ML / variational / autodiff; clean qGAN, quantum
  kernels, integrates with PyTorch/JAX.
- **Cirq** (Google) — lower-level; fine if we want circuit control.

Recommendation: **Qiskit for optimization/QAE**, **PennyLane for QML/generative**.

## Key algorithm references
- **QAOA** — Farhi, Goldstone, Gutmann (2014), "A Quantum Approximate Optimization Algorithm".
- **VQE** — Peruzzo et al. (2014), variational eigensolver.
- **Amplitude Estimation** — Brassard et al. (2002); **Iterative QAE** (Grinko et al. 2019)
  and **Maximum Likelihood QAE** (Suzuki et al. 2020) are NISQ-friendlier, no QPE needed.
- **Option pricing via QAE** — Stamatopoulos et al. (2020), "Option Pricing using Quantum
  Computers" (the canonical reference; Qiskit Finance tutorials mirror it).
- **Risk (VaR/CVaR) via QAE** — Woerner & Egger (2019), "Quantum Risk Analysis".
- **Portfolio optimization** — Egger et al., "Quantum Computing for Finance" survey.
- **qGAN distribution loading** — Zoufal, Lucchi, Woerner (2019).

## Datasets (public)
- Yahoo Finance via `yfinance` — equity prices/returns (portfolio, vol).
- Stooq / Quandl / FRED — macro & rates.
- For options: build synthetic surfaces (Black–Scholes / Heston) if real chains are hard.
- OP will provide on-site guidance on dataset sources — ask mentors early.

## Qiskit Finance tutorials worth cloning
- Portfolio optimization (QAOA/VQE) tutorial.
- European call option pricing (QAE) tutorial.
- Credit/portfolio risk analysis (QAE) tutorial.
These are excellent *starting scaffolds* — but add a real twist so it's not a tutorial rerun.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install qiskit qiskit-finance qiskit-optimization qiskit-aer pennylane yfinance \
            numpy pandas matplotlib scipy
```

## Practical NISQ notes
- Use **simulators** (`qiskit-aer`, PennyLane `default.qubit`) for the demo. Real hardware
  has queue + noise; treat live-hardware as a bonus, not a dependency.
- Keep qubit counts small (≤ ~10) so circuits actually run and we can show results.
- Always run a **classical baseline** alongside for the advantage comparison.
