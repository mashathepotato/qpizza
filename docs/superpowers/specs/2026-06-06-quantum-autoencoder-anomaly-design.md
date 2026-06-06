# Quantum Autoencoder for Nokia Return-Window Anomaly Detection

**Date:** 2026-06-06
**Status:** Approved (feasibility-exploration mode)
**Author:** Masha + Claude

## Goal

Feasibility study: can a small, fully-simulable **quantum autoencoder** (QAE, in the
Romero–Olson–Aspuru-Guzik 2017 sense) trained on NOKIA.HE daily-return windows produce a
useful **reconstruction-error anomaly signal**, and how does it compare to a classical PCA
baseline? We are NOT trying to beat a random walk on price prediction — daily equity returns
are near-unpredictable, and any such claim would be a red flag. The honest deliverable is a
reconstruction-error signal plus a transparent quantum-vs-classical comparison; the framing
(useful signal vs parameter-efficiency parity vs null result) is decided **after** seeing the
numbers.

## Non-goals

- Point price prediction / beating persistence on RMSE.
- Hardware execution. Everything runs on PennyLane `default.qubit` (statevector).
- Replacing any existing option-pricing route. This is an additive experiment.

## Data

- Source: NOKIA.HE daily closes via `yfinance` (same provider as `quantum_pricer/data.py`),
  with a labelled synthetic fallback when offline.
- Transform: log-returns `r_t = log(C_t / C_{t-1})`.
- Windows: sliding windows of `k = 4` consecutive returns (stride 1) → one sample per day.
- Scaling: z-score each window using **training-period** mean/std only (no leakage), then map
  to rotation angles. Angle encoding (not amplitude) is deliberate — it preserves window
  magnitude, which is the volatility signal an anomaly detector needs.
- Split: chronological. Train on an early "calm" stretch (default: first 50% of windows);
  evaluate on the full series (train + held-out), reporting held-out separately.

## Quantum autoencoder (`quantum_pricer/qautoencoder.py`)

- Framework: PennyLane `default.qubit`, 4 wires.
- Encoding: each of the 4 scaled returns → `RY(angle)` on its wire (optional `RZ` second axis).
- Encoder ansatz: 2 layers of parameterized `RY`/`RZ` rotations + a CZ entangling ring.
  ~16–24 trainable params.
- Bottleneck: latent = 2 wires, trash = 2 wires.
- Training objective (no explicit decoder needed): maximize the probability that the trash
  wires are in `|00>`. Loss = `1 - P(trash = |00>)`, averaged over calm-period windows.
- Optimizer: PennyLane Adam via autograd (parameter-shift compatible). Fixed seed.
- `score(window) -> float`: reconstruction infidelity `1 - P(trash = |00>)`. Higher = more
  anomalous.

## Classical baseline (`quantum_pricer/qautoencoder.py` or experiment file)

- PCA reconstruction error: fit PCA (2 components, matching the 2-qubit latent) on calm-period
  windows via numpy SVD (zero new deps). Score = squared reconstruction residual norm.
- Parameter count reported for both (PCA: components × k; QAE: ansatz params).

## Evaluation (`results/qml_autoencoder.py`)

No true anomaly labels exist. Use a **transparent proxy**, clearly labelled in code, figure,
and prose: "anomaly" = top-decile absolute-return days (volatility events).

Metrics:
1. ROC-AUC of each model's anomaly score vs the proxy labels (whole series + held-out only).
2. Trainable parameter counts (quantum vs classical).
3. Figure `results/figures/qml_autoencoder.png`: top panel = price with proxy events marked;
   bottom panel = QAE and PCA reconstruction-error traces over time, train/test split shaded.

Reproducibility: seed numpy and PennyLane; the experiment prints a provenance line (data
source, n_obs, date range) like the existing backtests.

## Tests (`quantum_pricer/tests/test_qautoencoder.py`)

- Circuit shape: 4 wires, expected trainable-param count.
- Training reduces trash-loss on a clean toy signal (sanity: model learns *something*).
- `score` is deterministic under a fixed seed.
- Injected-spike test: a window with an outsized return scores higher than calm windows.
- PCA baseline: reconstruction error is non-negative and higher on an injected spike.

## Dependencies

- Add `pennylane>=0.35` to `quantum_pricer/requirements.txt` (installed: 0.45.0).
- No sklearn — PCA is numpy SVD.

## Honest-result clause

Report AUC / params / figure exactly as produced. If the QAE merely matches PCA, that is the
finding and the parameter-count comparison becomes the headline. If it underperforms, we say
so. No dressing up a null result.
