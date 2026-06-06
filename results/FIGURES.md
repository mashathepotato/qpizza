# Results figures — a guide for the team

Every figure in `results/figures/` (and shown on the dashboard, `results/index.html`).
For each: **what it shows**, **how to read it**, the **honest caveat**, and **how to
regenerate**. Rebuild the whole dashboard with:

```bash
quantum_investor/.venv/bin/python -m results.build_dashboard   # -> results/index.html
```

The pricer scripts need the pricer venv + network (yfinance):
`PYTHONPATH=$(pwd) quantum_pricer/.venv/bin/python results/<script>.py`.

---

## Quantum option pricer (the technical core)

### `benchmark_scoreboard.png`  — START HERE
- **What:** one table comparing every method (Classical MC, Black-Scholes, our QNDM
  Fourier/QAE/QSVT, the quantum SOTA oracle-QAE) on accuracy, CZ depth, qubits, and
  ε-scaling. Numbers are read from the real runs (`verify_backtest.csv` + `results.json`).
- **Read:** our **QNDM-QAE** (green row, ★) is the headline — matches the price at the
  shallowest depth (16 CZ). The **SOTA** row (pink) is the standard quantum approach we
  beat on loading depth (100 CZ).
- **Caveat:** accuracy ground truths differ — our routes vs the exact binomial tree, SOTA
  vs Black-Scholes continuum. Depth/qubits are like-for-like; the price errors are not.
- **Regenerate:** `results/scoreboard.py` (needs `verify_backtest.csv` + `results.json`).

### `price_forecast.png`  — what we price
- **What:** from today's spot S0=13.09, the CRR risk-neutral tree's predicted price
  distribution over the 1-year horizon + the terminal distribution the call is priced on.
- **Read:** left = forecast fan (expected/forward path + 5-95% cone); right = terminal
  price distribution with the in-the-money region shaded.
- **Caveat:** risk-neutral (pricing) measure, not a real-world price forecast.
- **Regenerate:** `results/price_forecast.py`.

### `model_results.png`  — actual model outputs
- **What:** all four routes run for real. Left: each route's price vs the exact-tree
  ground truth (errors annotated). Right: actual transpiled CZ depth + qubits per route.
- **Read:** every route lands on the true price; QAE is the shallowest quantum route.
- **Caveat:** M=3 (small) — shows correctness, not scale.
- **Regenerate:** `results/model_experiments.py`.

### `error_vs_queries.png`  — empirical speedup (cf. Stamatopoulos Fig. 11)
- **What:** estimation error ε vs number of oracle queries / samples (log-log).
- **Read:** Amplitude Estimation descends steeper (slope ~−0.80) than Monte Carlo
  (~−0.58); guides show the ideal −1 and −1/2.
- **Caveat:** AE is under the ideal −1.0 (small-M finite-shot saturation); queries ≠ samples.
- **Regenerate:** `results/fig11_error_vs_queries.py`.

### `speedup_compare.png`  — head-to-head speedup (MC vs Greek-SOTA vs ours)
- **What:** estimation error vs oracle queries / samples (log-log) for Monte Carlo, the
  Greek-paper SOTA oracle-QAE (Stamatopoulos), and our QNDM-QAE + QSVT — all real runs,
  finite shots, seed-averaged. This is the figure that puts QSVT and the Greek baseline on
  the same speedup axes.
- **Read:** our **QNDM-QAE** gets the ideal quantum slope (**−0.99**) and the lowest error;
  the **Greek SOTA** floors high (~0.2) at its rescaling-linearization bias; **QSVT** floors
  at its polynomial-degree bias (~0.02); **MC** scales ~−0.5.
- **Caveat:** each error is vs the value that method estimates (ours/MC → exact tree; Greek →
  its discretized lognormal). Floors are model biases, not estimation noise.
- **Regenerate:** `results/speedup_compare.py` (~6–8 min; writes `speedup_compare.json`).

### `complexity.png`  — query-complexity advantage (theory)
- **What:** queries to reach accuracy ε: MC O(1/ε²) (slope 2.0) vs QAE O(1/ε) (slope 1.0).
- **Read:** the quadratic separation — the core theoretical advantage.
- **Caveat:** both curves are analytic/theoretical (cite Montanaro 2015), not measured.
- **Regenerate:** part of `quantum_pricer/demo.py`.

### `speedup.png`  — empirical RMS error vs queries
- **What:** seed-averaged measured RMS error vs queries; the measured counterpart to
  `complexity.png` (QAE ~−0.80, MC ~−0.58).
- **Regenerate:** part of `quantum_pricer/demo.py`.

### `depth_crossover.png`  — the novelty
- **What:** transpiled CZ depth of the payoff phase oracle: naive 2^M vs Hamming-weight
  poly(M).
- **Read:** crossover at M=6; naive is infeasible from M≈14 while the Hamming oracle prices
  M=14 to 3e-4 error on 19 qubits. This is our strongest, most defensible result.
- **Caveat:** a circuit-feasibility result, not a runtime claim.
- **Regenerate:** part of `quantum_pricer/demo.py`.

### `backtest_routes_timeseries.png`  — full verification (option price)
- **What:** an ATM call priced on EVERY one of the 192 sliding windows with all four
  routes (real circuit runs). Top: price per route vs exact tree (they overlap = agree).
  Bottom: absolute error per route (log).
- **Read:** Fourier ~1e-8 (exact) < QAE ~4e-5 < MC ~3e-3 < QSVT ~5e-3 (straddle floor).
- **Caveat:** this is the OPTION price (algorithm correctness), not a price forecast.
- **Regenerate:** `results/verify_backtest.py` (~10 min; writes `verify_backtest.csv`).

### `backtest_rolling.png`  — price PREDICTION + forecast error
- **What:** predicting the UNDERLYING NOKIA price 30 trading days ahead. Sliding 60-day
  windows (stride 1), 30 calibrate / 30 predict (M=8), averaged per day. Top: predicted
  price + cone vs realized. Bottom: forecast error (predicted − realized) over time.
- **Read:** the error goes negative — the risk-neutral forecast under-shoots the rally.
- **Caveat:** underlying-price prediction (route-independent), NOT the option price.
- **Regenerate:** `results/backtest_rolling.py`.

### `backtest_drift_compare.png`  — risk-neutral vs real-world drift
- **What:** same rolling backtest, side by side. Left: risk-neutral drift r (pricing
  measure). Right: real-world drift μ̂ (estimated per window). Rows: price prediction /
  forecast error.
- **Read:** real-world μ̂ tracks the trend (lower MAE 0.74 vs 0.93) but is noisier and
  overshoots reversals (coverage 57% vs 67%).
- **Caveat:** 30-day μ̂ is statistically noisy; this is a model-assumption comparison.
- **Regenerate:** `results/backtest_drift_compare.py`.

### `backtest_walkforward.png`  — non-overlapping chunked backtest
- **What:** the earlier version: 4 non-overlapping 60-day chunks. Top: per-chunk forecast
  cone vs realized. Bottom: per-chunk option price by all four routes vs exact tree (bars).
- **Read:** complements the rolling figure; the bars make per-chunk route agreement explicit.
- **Regenerate:** `results/backtest.py`.

---

## Cognition (the motivation: "the madness of people is quantum")

### `figure.png`
- **What:** real Clinton/Gore question-order data (PNAS 2014). Left: humans answer
  differently by order; the classical model predicts the same value twice. Right: the
  order effect is large yet the parameter-free QQ-equality q ≈ 0 (the quantum signature).
- **Read:** the win is the parameter-free QQ-equality (q = −0.003), not a fit-quality contest.
- **Caveat:** a single-qubit model reproduces the structure, not the full joint fit.
- **Regenerate:** `cd quantum_investor && python main.py`.

---

## Triage (overnight method sweep)

### `qae_scaling.png`
- **What:** measured QAE oracle queries vs MC samples to reach accuracy ε (log-log).
- **Read:** QAE slope ~−0.82 vs MC ~−2.0 — the quadratic edge, but only at tight ε.
- **Caveat:** at coarse ε the fixed per-round cost lets classical win.
- **Regenerate:** triage worktree (`triage/harness/qae.py`).
