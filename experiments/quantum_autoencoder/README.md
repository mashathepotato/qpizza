# Quantum autoencoder on NOKIA.HE returns — experimental sidecar

**Status:** experiment / honest negative-parity result. Isolated from the
`quantum_pricer` package on purpose (separate branch + folder).

## What this is

A Romero–Olson–Aspuru-Guzik (2017) style **quantum autoencoder** (PennyLane
`default.qubit`, 4 qubits, 2 latent / 2 trash, ~16 params) trained on *calm*-period
NOKIA.HE 4-day log-return windows so the trash qubits disentangle to `|00>`. The
anomaly score is the reconstruction infidelity `1 − P(trash=|00>)`. Classical
baseline: 2-component **PCA** reconstruction error (numpy SVD, 8 params), plus a
trivial energy `‖z‖²` baseline. Everything is fully simulable on a laptop.

Design spec: `docs/superpowers/specs/2026-06-06-quantum-autoencoder-anomaly-design.md`.

## The honest finding (real NOKIA.HE, 1y daily)

There are no true anomaly labels, so we use a **transparent proxy**: top-decile
`|return|` days (volatility events). Two framings:

| Task | What it asks | QAE | PCA-2 | energy ‖z‖² | note |
|---|---|---|---|---|---|
| **Concurrent detection** | flag day *t* (in the window) | 0.90 | 0.75 | 0.83 | **leaky** — `last \|z\|` alone = AUC 1.00 |
| **Next-day prediction** (this figure) | foreshadow day *t+1* (not in window) | 0.64 | 0.68 | 0.64 | clean; only vol-clustering signal |

(held-out ROC-AUC; QAE is seed-stable, σ≈0.000 over 8 seeds.)

- The flattering concurrent "win" is an artifact of **label leakage**: the flagged
  return is also one of the four model inputs, so a bare `|return|` detector scores
  a perfect 1.00. Only the QAE-vs-classical *gap* there is real content.
- On the honest **next-day** task (target not in the inputs), every model clusters
  at ~0.64–0.68 — that's real volatility clustering (GARCH-like) — and **the quantum
  model does not beat classical PCA.** No quantum advantage for prediction.

This matches the project's stance: you cannot out-predict a near-random-walk, and
quantum doesn't change that. The deliverable is a fully-simulable QAE plus an honest
parity/negative comparison — not a leaderboard win.

## Files

- `qautoencoder.py` — windowing, z-scoring, `QuantumAutoencoder`, `PCAReconstructor`, `roc_auc`.
- `test_qautoencoder.py` — 10 unit tests (circuit shape, training reduces loss, spike > calm, PCA, AUC).
- `run_predictive.py` — the honest next-day experiment → `qml_autoencoder_predictive.png`.
- `requirements.txt` — adds `pennylane` on top of the pricer venv.

## Run

```bash
quantum_pricer/.venv/bin/python -m pip install -r experiments/quantum_autoencoder/requirements.txt
quantum_pricer/.venv/bin/python -m pytest experiments/quantum_autoencoder/test_qautoencoder.py -q
quantum_pricer/.venv/bin/python experiments/quantum_autoencoder/run_predictive.py
```
