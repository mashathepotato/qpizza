# quantum-pricer

A working implementation of quantum option pricing for the OP Pohjola
QHack 2026 hackathon.  The package benchmarks **three routes** for pricing a
European (or Asian) call option on Nokia (NOKIA.HE):

| Route | Query complexity | Circuit depth (IQM CZ) | Why it matters |
|---|---|---|---|
| Classical MC | O(1/ε²) samples | 0 | Baseline |
| QNDM Fourier | O(1/ε²) shots* | shallow (~112 CZ at M=4) | Q50-feasible depth, exact loading |
| QNDM QAE | **O(1/ε)** oracle queries | ~16 CZ | Quadratic quantum speedup |
| Novel QSVT | **O(1/ε)** oracle queries | ~2240 CZ | Honest straddle construction |

*Fourier's shot complexity is O(1/ε²), parallel to classical MC.  Its advantage
is shallow circuit depth and exact state loading — not ε-scaling.

---

## Ground truth and reference points

* **Ground truth:** exact CRR binomial-tree price (`tree.exact_tree_price`).
  This is the quantity all quantum routes estimate.  It converges to Black-Scholes
  as M → ∞.
* **Continuum check:** Black-Scholes (`classical.black_scholes_call`).
  The gap with the tree shrinks as M increases; it is *not* an error in either.

---

## QSVT route — honest construction

The call payoff max(f−K, 0) has a kink at f=K and is one-sided, so a single
QSP/QET-U sequence cannot represent it directly.  The QSVT route instead
genuinely measures the **symmetric straddle**

    E[|f − K|]   (even-parity in θ(x) = c(f(x)−K), QSP-friendly)

and recovers the call and put via **put-call parity** using the a-priori forward
price F = S0 · e^(rT):

    call = e^(−rT) (E[|f−K|] + (F−K)) / 2
    put  = e^(−rT) (E[|f−K|] − (F−K)) / 2

**Nothing in the pricing chain uses the known option answer.**  Every constant
comes from:
- the payoff *value range* → Cmax = max_x |f(x)−K|, coupling c = 1/Cmax
- the chosen QSP phases → transfer-function probe (w₀, κ), calibrated on swept
  known phases, never on the option answer
- the forward F from market inputs (S0, r, T, M) only

**Known limitation:** degree-60 Chebyshev approximation of √(A + B|θ|) near the
kink at θ=0 leaves a polynomial-approximation residual of ~1.4% at degree 60.
This residual shrinks monotonically with higher degree and is annotated in every
benchmark row (note="qsvt_approx_floor").  It is an honest physics constraint,
not a code bug.

---

## Quadratic quantum advantage — query-complexity plot

The "money slide" (`complexity.png`, saved by the demo) shows:

- **Classical MC (analytic CLT):** N_MC(ε) = (σ_payoff / ε)²   — slope 2 on a
  log-log(1/ε) vs queries plot
- **QAE theoretical:** N_QAE(ε) ≈ π / (2ε)   — slope 1 (quadratic speedup)
- **QAE empirical (IAE):** actual oracle queries from the simulator; at small M
  the Grover-power schedule saturates (all fine-ε targets report the same query
  count) — annotated honestly with note="qae_saturated"

The empirical fitted slopes from the demo (Nokia live data, M=4):

    MC analytic slope  ~2.00  (theory = 2)
    QAE theory slope   ~1.00  (theory = 1)

---

## Market data

- **Network available:** real Nokia (NOKIA.HE) close prices via `yfinance`;
  annualized realized volatility from 252-day log-returns; spot S0 = last close.
- **Network unavailable:** clearly-labelled synthetic fallback
  S0=4.20, σ=0.30, r=0.03 — every output is annotated `[SYNTHETIC FALLBACK]`.
- Hardware test (q50_fake backend, M=1) shows ~15% noise error; reported as-is.

---

## Quick start

```bash
pip install -r quantum_pricer/requirements.txt
python -m quantum_pricer.demo           # end-to-end benchmark, saves complexity.png
python -m quantum_pricer.run_hardware --backend q50_fake --M 1
python -m pytest quantum_pricer/tests -v
```

The demo completes in well under a minute on a laptop (M=3 for per-route prices,
M=4 for the benchmark/resource table; QSVT is skipped in the error-vs-queries
sweep at M=4 to stay fast, but is priced at M=3 with an honest residual note).

---

## Module map

| File | Role |
|---|---|
| `data.py` | Nokia market data (yfinance + synthetic fallback) |
| `tree.py` | CRR tree: loading angles, path probabilities, exact tree price |
| `classical.py` | Black-Scholes + Monte Carlo on the CRR tree |
| `fourier.py` | QNDM Fourier route (characteristic-function inversion) |
| `qae.py` | QNDM amplitude-QAE route (IterativeAmplitudeEstimation) |
| `qsvt.py` | Novel QSVT route (straddle polynomial + put-call parity) |
| `oracles.py` | Shared quantum oracles (path loader, payoff amplitude, Fourier) |
| `benchmark.py` | `error_vs_queries`, `resource_table`, `queries_to_accuracy`, plot savers |
| `backends.py` | IQM basis gates, q50_fake noise model |
| `run_hardware.py` | CLI for hardware/fake-hardware experiments |
| `demo.py` | Runnable end-to-end benchmark (this demo) |
| `tests/` | pytest suite (~29 tests) |
