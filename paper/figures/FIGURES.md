# Figures for the quantum-option-pricing paper

Generated from the working models in `quantum_pricer/` (NOKIA.HE ATM European call,
exact CRR binomial tree as ground truth). Regenerate with:

```bash
PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python results/model_experiments.py
PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python results/fig11_error_vs_queries.py
PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python quantum_pricer/demo.py   # complexity/speedup/depth_crossover
```

Suggested order of importance: **depth_crossover → error_vs_queries → model_results → complexity → speedup**.

---

## 1. `depth_crossover.png` — lead result (novelty)
The strongest, most defensible contribution: a Hamming-weight phase oracle with
**poly(M)** CZ depth versus the naive **2^M** diagonal. Crossover at M=6; the naive
oracle is infeasible from M≈14 (16,384-entry diagonal), while the Hamming route
prices M=14 to abs error 3e-4 on 19 qubits.

> **Figure caption (LaTeX-ready):** Transpiled CZ depth (IQM \{r, cz\} basis) of the
> payoff phase oracle versus the number of binomial time steps $M$: naive diagonal
> synthesis ($2^M$, exponential) versus the Hamming-weight construction
> (polynomial in $M$). The naive route becomes infeasible for $M \gtrsim 14$,
> whereas the Hamming-weight oracle remains shallow and prices an $M=14$ tree to
> absolute error $3\times10^{-4}$ on 19 qubits.

This is a **circuit-complexity / feasibility** result — not an apples-to-oranges
runtime claim. Safe to lead with.

## 2. `error_vs_queries.png` — empirical scaling (cf. Stamatopoulos Fig. 11)
Estimation error $\varepsilon$ versus number of samples / oracle queries $M$ (log-log),
following arXiv:1905.02666 Fig. 11. Amplitude Estimation descends with measured
slope **−0.80** versus Monte Carlo **−0.58**; faint guides show the ideal −1 and −1/2.

> **Caption:** Estimation error $\varepsilon$ as a function of the number of samples /
> oracle queries $M$ for the QNDM+QAE pricer (blue) and classical Monte Carlo (red),
> log-log. Measured slopes are $-0.80$ (AE) and $-0.58$ (MC); reference lines mark the
> asymptotic $1/M$ and $1/\sqrt{M}$ scalings. Convention follows
> Stamatopoulos et al. (2020), Fig. 11.

> **Honesty note (keep in text):** the measured AE slope is $-0.80$, below the ideal
> $-1.0$, because finite-shot IAE saturates at the small $M$ accessible to statevector
> simulation; the $x$-axis compares AE *oracle queries* to MC *samples* — a
> query-complexity comparison, not wall-clock.

## 3. `model_results.png` — actual model outputs
Two panels. Left: every route (classical MC, QNDM Fourier, QNDM QAE, novel QSVT)
recovers the exact-tree price (2.7752) within its error (MC −5e-4 with sampled
stderr, Fourier ≈0, QAE +3e-4, QSVT +9.2e-3 ≈ its 1.4% straddle floor). Right: actual
transpiled IQM CZ depth + qubits per route (MC 0, Fourier 112, QAE **16**, QSVT 2240).

> **Caption:** Left: option price from each route versus the exact binomial-tree
> ground truth (dashed) at $M=3$, with signed errors annotated. Right: transpiled
> circuit cost per route (IQM CZ depth, log scale; qubit count). QNDM+QAE is the
> shallowest quantum route ($\sim$16 CZ); the QSVT straddle route is deepest.

> **Honesty note:** $M=3$ — this panel demonstrates *correctness*, not scale.

## 4. `complexity.png` — query-complexity speedup (theory)
Asymptotic query complexity: MC sample count $\sim(\sigma/\varepsilon)^2$ (slope 2.0)
versus QAE $\sim\pi/(2\varepsilon)$ (slope 1.0) — the textbook quadratic separation.

> **Caption:** Query complexity to reach target accuracy $\varepsilon$: analytic
> Monte-Carlo sample count (slope $2.0$, $O(1/\varepsilon^2)$) versus the amplitude-
> estimation query count (slope $1.0$, $O(1/\varepsilon)$).

> **Honesty note:** both curves are analytic/theoretical (Montanaro 2015; Brassard
> et al. 2002) — present in a *query-complexity* section, not as empirical evidence.

## 5. `speedup.png` — empirical RMS error vs queries
Seed-averaged RMS price error versus queries; companion to `complexity.png` showing
the *measured* descent (MC ≈ −0.58, QAE ≈ −0.80).

> **Caption:** Seed-averaged RMS pricing error versus number of queries/samples;
> finite-shot QAE (slope $\approx-0.80$) versus Monte Carlo ($\approx-0.58$).

---

**One-line framing for the paper's claims:** asymptotic quadratic *query-complexity*
advantage (standard, reproduced) **plus** a novel poly(M) Hamming-weight loading
oracle that makes exact path-loading feasible at depths naive synthesis cannot reach.
Do **not** claim a demonstrated wall-clock speedup over Monte Carlo.
