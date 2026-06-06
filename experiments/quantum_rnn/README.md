# Recurrent & quantum-recurrent benchmark on NOKIA.HE — experimental sidecar

**Status:** experiment / honest parity result. Isolated from the `quantum_pricer`
package on purpose (separate folder, experimental branch).

## What this is

A fair, same-task comparison of sequence models on the honest NOKIA.HE next-day
volatility task. One supervised task for every model: given a length-`L=10` sequence
of daily log-returns ending day *t*, predict whether day *t+1* is a top-decile
`|return|` day (a volatility event). **The target return is not in the input
sequence — no leakage.** Held-out ROC-AUC, chronological 50/50 split.

Design spec: `docs/superpowers/specs/2026-06-06-quantum-recurrent-benchmark-design.md`.

### Models
- **Classical recurrent** (PyTorch): `RNN`, `GRU`, `LSTM` (`hidden=8`).
- **Quantum recurrent** (PennyLane `default.qubit`, 4 qubits, data-reuploading VQC
  cells, Chen et al. 2020): `QRNN`, `QGRU`, `QLSTM`. Each gate is a small
  `StronglyEntanglingLayers` circuit reused across timesteps; fully simulable.
- **Transformers**: a classical `Transformer` encoder (multi-head self-attention +
  FFN, sinusoidal positional encoding) and a `QTransformer` — quantum self-attention
  (QSANN, Li et al. 2022) whose Q/K/V projections are VQCs, with classical softmax
  attention. Both mean-pool and use a linear head.
- **Autoregressive baselines**: `GARCH(1,1)` (the textbook volatility-clustering
  model, `arch`) and `AR(p)` on `|returns|` (statsmodels).
- **Floor baselines**: persistence (`|return|_t`) and logistic regression.

## The honest finding (real NOKIA.HE, 1y daily; 5-seed mean, matched regularization)

| Model | family | held-out AUC | params |
|---|---|---|---|
| QLSTM | quantum | 0.647 ± 0.008 | 101 |
| **GARCH(1,1)** | autoregressive | **0.639** | **4** |
| LSTM | classical | 0.637 ± 0.010 | 361 |
| QTransformer | quantum | 0.628 ± 0.009 | 85 |
| QGRU | quantum | 0.625 ± 0.022 | 77 |
| Transformer | classical | 0.621 ± 0.046 | 185 |
| Persistence | floor | 0.620 | 0 |
| AR(\|r\|,5) | autoregressive | 0.617 | 6 |
| GRU | classical | 0.615 ± 0.004 | 273 |
| QRNN | quantum | 0.612 ± 0.032 | 29 |
| RNN | classical | 0.575 ± 0.024 | 97 |
| Logistic | floor | 0.564 | 11 |

**Takeaways:**
- **No model meaningfully beats GARCH(1,1)** — a 4-parameter classical model from
  1986. The best performer (QLSTM, 0.647) ties it within one standard deviation.
- Everything (recurrent, transformer, quantum, classical) clusters ~0.61–0.65,
  barely above the persistence floor (0.620). On near-random-walk daily returns,
  only volatility clustering is learnable and a tiny GARCH captures it as well as
  anything. **Architecture barely matters** — including attention: the classical
  Transformer is actually the *least stable* model (±0.046), over-parameterized for
  120 training samples.
- **Quantum is on par with classical, not ahead.** The quantum cells reach the
  cluster with far fewer parameters (29–101 vs 97–361) and no explicit
  regularization — a mild parameter-efficiency observation, not a predictive win.

### A methodology note we left in on purpose
The *first* run (no weight decay, single seed) showed the quantum models apparently
**crushing** classical (QGRU 0.665 vs LSTM 0.556). That was an artifact: the larger
classical models overfit this tiny (120-sample) training set. Adding L2 weight decay
to *all* torch models and averaging over seeds collapses the gap — classical LSTM
rises 0.556 → 0.635. The lesson (fair baselines + regularization + seed averaging
before believing a surprising win) is part of the result.

## Files

- `seqdata.py` — sequence/label construction (no leakage), z-scoring, `roc_auc`.
- `classical_rnn.py` — `RNN/GRU/LSTMClassifier` + shared `train/scores/count_params`.
- `qcell.py` — shared VQC builders (`make_gate_layer` for recurrence, `make_token_layer` for attention Q/K/V).
- `quantum_rnn.py`, `quantum_gru.py`, `quantum_lstm.py` — `QRNN/QGRU/QLSTM`.
- `classical_tf.py`, `quantum_tf.py` — `TransformerClassifier`, `QuantumTransformer` (QSANN).
- `ar_baseline.py` — `garch_forecast`, `ar_abs_forecast`.
- `baselines.py` — persistence + logistic.
- `run_rnn_bench.py` — the benchmark → `rnn_benchmark.png`.
- `predictions_plot.py` — every model's per-day held-out prediction overlaid vs the
  true vol-event days → `predictions_per_day.png` (v1 static; animation is future work).
- `ensemble_plot.py` — frozen-expert ensembles: classical (4 NN + GARCH(1,1)) vs
  quantum (4 NN). Each member's held-out score is min-max normalized then uniformly
  averaged (no learned gate) into two prediction lines → `ensemble_predictions.png`.
  Classical 0.65 vs quantum 0.65 held-out AUC — a dead heat (single-threaded for
  reproducibility; folding in GARCH lifts classical from 0.63 to match quantum).
- `test_*.py` — 41 unit tests (harness, every recurrent + transformer cell, baselines).
- `requirements.txt` — `torch`, `pennylane`, `arch`.

## Run

```bash
quantum_pricer/.venv/bin/python -m pip install -r experiments/quantum_rnn/requirements.txt
quantum_pricer/.venv/bin/python -m pytest experiments/quantum_rnn/ -q
quantum_pricer/.venv/bin/python experiments/quantum_rnn/run_rnn_bench.py   # ~5 min
```
