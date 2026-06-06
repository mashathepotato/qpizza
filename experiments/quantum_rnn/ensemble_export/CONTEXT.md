# Frozen-expert ensemble — raw plot data

This folder is the **raw data behind the `ensemble_predictions.png` figure**, exported so
you can build your own plots. It is self-contained: you only need these CSV/JSON files.

## What the experiment is

We test whether classical and quantum neural sequence models differ at predicting
**next-day volatility** on NOKIA.HE daily data. Important: **the models do NOT predict
price.** Daily returns are a near-random walk and unpredictable; the only learnable signal
is *volatility clustering*. So the task is binary classification:

> Given the last 10 daily log-returns (ending day *t*), is day *t+1* a **top-decile
> |return| day** (a "volatility event")?

Two "frozen-expert" ensembles are compared (mixture-of-experts style, uniform average, no
learned gate):

- **classical** = RNN, GRU, LSTM, Transformer (each a small PyTorch net, seed-averaged over
  5 seeds) **+ GARCH(1,1)** (classic volatility model).
- **quantum** = QRNN, QGRU, QLSTM, QTransformer (PennyLane variational-circuit cells,
  data re-uploading / QSANN), seed-averaged over 5 seeds.

Each member's held-out score is **min-max normalized to [0,1]** over the test window, then
the members are averaged within each family to give the two ensemble lines.

## Files

### `ensemble_timeseries.csv` — the main plot data (one row per held-out trading day)
| column | meaning |
|---|---|
| `trading_day` | index of the day in the NOKIA.HE series (held-out region only) |
| `price_eur` | NOKIA.HE closing price that day (context; **not** a prediction target) |
| `true_vol_event` | ground truth: 1 if next day is a top-decile \|return\| day, else 0 |
| `next_abs_return` | the actual next-day \|log-return\| (the magnitude behind the label) |
| `classical_ensemble_score` | classical ensemble's predicted score (0–1), the mean of its normalized members |
| `quantum_ensemble_score` | quantum ensemble's predicted score (0–1) |

The committed figure is: **top panel** = `price_eur` vs `trading_day` with dots where
`true_vol_event==1`; **bottom panel** = `classical_ensemble_score` and
`quantum_ensemble_score` vs `trading_day`, with `true_vol_event` days shaded.

### `members_normalized.csv` — per-model breakdown (one row per held-out day)
`trading_day` + one column per individual member (RNN, GRU, LSTM, Transformer, GARCH,
QRNN, QGRU, QLSTM, QTransformer), each the **normalized [0,1]** held-out score that feeds
the ensemble. By construction, the mean of the 5 classical columns equals
`classical_ensemble_score`, and the mean of the 4 quantum columns equals
`quantum_ensemble_score`. Use this if you want to plot individual experts or re-aggregate.

### `summary.json` — metadata and headline metrics
Dataset/provenance, task settings (L=10, 90th-percentile proxy, train/test split, seeds,
epochs), per-member held-out **ROC-AUC**, and the two ensemble AUCs.

## Headline result

Classical and quantum ensembles are a **dead heat** (~0.64–0.66 held-out AUC, both barely
above the ~0.62 persistence floor and around GARCH's 0.639). No classical/quantum
separation. See `summary.json` for the exact numbers of this run.

## Caveats
- Scores are normalized per member; they are relative ranking signals, not calibrated
  probabilities. AUC (rank-based) is the metric, so normalization does not affect it.
- ~0.01 run-to-run AUC noise on retrain (CPU BLAS nondeterminism); these files are the
  canonical data for the committed figure — plot from them rather than re-running.
- "Volatility event" is a transparent proxy (top-decile next-day |return|), not an
  external label.
