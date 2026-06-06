# Recurrent + Quantum-Recurrent Benchmark on NOKIA.HE Next-Day Volatility

**Date:** 2026-06-06
**Status:** Approved (feasibility-exploration mode; branch `experiment/quantum-autoencoder`)
**Author:** Masha + Claude

## Goal

Extend the honest NOKIA.HE volatility study with sequence models. Build a family of
recurrent classifiers -- classical RNN/GRU/LSTM (PyTorch) and their quantum analogues
QRNN/QGRU/QLSTM (PennyLane data-reuploading VQC cells, Chen et al. 2020 style) -- plus
classical statistical baselines, all benchmarked on ONE supervised task with held-out
ROC-AUC. This is a fair same-task comparison and a set of clean, fully-simulable
quantum-recurrent implementations, NOT an attempt to beat a random walk.

## Honest expectation (stated up front)

Daily equity returns are near-unpredictable; the only learnable signal is volatility
clustering. We expect all models to cluster ~0.62-0.68 held-out AUC and quantum is
unlikely to beat classical. GARCH(1,1) is the principled classical model for this
signal; if quantum cannot beat GARCH, that is the headline honest result. Deliverables:
(a) working simulable QRNN/QGRU/QLSTM, (b) fair comparison table, (c) the
parameter-efficiency observation (quantum cells have far fewer params).

## Task (shared by every model)

- Source: NOKIA.HE daily closes via yfinance (labelled synthetic fallback offline).
- Sequence: `L = 10` consecutive log-returns ending day t (one sample per day).
- Label (proxy, transparent): next-day `|return|` (day t+1, NOT in the sequence) in the
  top decile of the training distribution -> binary "volatility event".
- Scaling: z-score sequence inputs using TRAINING stats only.
- Split: chronological 50/50 (train early, test late).
- Metric: held-out ROC-AUC (rank-based, tie-aware). BCE training loss.
- Floor baselines: persistence (score = `|return|_t`), logistic regression on the
  flattened window.

## Models

### Classical recurrent (PyTorch) -- `classical_rnn.py`
`RNNClassifier`, `GRUClassifier`, `LSTMClassifier`: `nn.RNN/GRU/LSTM(input_size=1,
hidden_size=8, batch_first=True)` -> last hidden -> `Linear(8,1)` -> sigmoid.

### Quantum recurrent (PennyLane + `qml.qnn.TorchLayer`) -- `quantum_rnn.py`
- 4 qubits, `default.qubit`, torch interface (backprop). Hidden = 4 ⟨Z⟩ expectations.
- Per timestep, data re-uploading: encode `[x_t, hidden]` as RY/RZ angles, apply a
  variational entangling block (StronglyEntanglingLayers), measure ⟨Z⟩ -> new hidden.
- `QRNN`: one VQC recurrence; final hidden -> `Linear(4,1)` -> sigmoid.
- `QGRU`: update/reset gates each a small VQC; classical GRU combination of carried
  state with VQC gate outputs.
- `QLSTM`: forget/input/output/cell gates each a small VQC (Chen et al. 2020);
  classical LSTM cell-state recurrence.
- Inputs are batched via broadcasting; sequence length looped in Python.

### Autoregressive statistical baselines -- `ar_baseline.py`
- **GARCH(1,1)** (headline): fit on training log-returns (in %), roll one-step-ahead
  conditional-volatility forecast across the series; score = forecast vol for day t+1.
  Via the `arch` package.
- **AR(p) on |returns|** (the plain "ARIMA-flavored" model): fit AR(p) (p=5) on absolute
  returns (statsmodels, pulled in by `arch`); score = next-day `|return|` forecast.

Both are scored by the SAME AUC against the SAME proxy label, so they sit in one table.

## Benchmark + figure -- `run_rnn_bench.py`

- Train all 6 recurrent models + GARCH + AR + persistence + logistic.
- Print + plot one held-out AUC table/bar chart with trainable-param counts.
- Per-day score overlay (top panel: price + next-day vol events; bottom: best-classical
  vs best-quantum vs GARCH normalized scores), consistent with the AE figure style.
- Reuse the repo `results.style` look when importable; honest caption + provenance stamp.

## Files (new folder `experiments/quantum_rnn/`)

- `seqdata.py` -- sequence building, z-scoring, label proxy, train/test split, `roc_auc`
  (vendored ~15-line copy to keep the folder self-contained).
- `classical_rnn.py`, `quantum_rnn.py`, `ar_baseline.py`, `baselines.py`
- `run_rnn_bench.py` -> `rnn_benchmark.png`
- `test_seqdata.py`, `test_classical_rnn.py`, `test_quantum_rnn.py`, `test_ar_baseline.py`
- `requirements.txt` (`torch`, `pennylane`, `arch`), `README.md`

## Testing (TDD for every cell)

- seqdata: window/label shapes, no-leakage (target not in sequence), z-score train-only,
  roc_auc perfect/random sanity.
- classical + quantum cells: output shape (batch,1) in [0,1]; forward deterministic under
  seed; one training step reduces BCE loss on a toy separable signal; param count.
- ar_baseline: GARCH fits and produces positive vol forecasts; AR forecast finite;
  spike day forecast >= calm forecast on a synthetic clustered series.

## Implementation approach

Shared harness + classical trio + baselines first (fast, verify the task end-to-end),
then the three quantum cells built subagent-driven in parallel (independent, meaty),
each TDD'd, then the AR baselines, then the combined benchmark figure.

## Dependencies

Add to `experiments/quantum_rnn/requirements.txt`: `torch` (CPU), `pennylane>=0.35`,
`arch>=6.0`. Install into `quantum_pricer/.venv`. The `quantum_pricer` package stays
untouched.

## Honest-result clause

Report the AUC table exactly as produced. If quantum ties or trails classical and GARCH
(expected), that is the finding. No dressing up a null result.
