# Design — Quantum Herding bridge (cognition ↔ pricer)

**Date:** 2026-06-06
**Status:** approved (brainstorming), ready for implementation plan
**Owner:** _unclaimed_

## 1. Purpose

Bridge the project's two halves into one coherent story, backed by a runnable demo:

- **Motivation (existing):** "The Madness of People Is Quantum" — investor decisions are
  non-classical (order effects, interference). The crowd's decisions are *correlated*, not
  independent ("interference of intentions", Yukalov–Sornette).
- **Technical core (existing, `quantum_pricer/`):** a quantum option pricer over the binomial
  tree, whose classical baseline assumes price moves are an **independent product** measure
  `p(x) = ∏ qᵢ^{xᵢ}(1−qᵢ)^{1−xᵢ}`.

**The bridge:** make the crowd's non-independence concrete by adding a tunable **herding entangler
`J`** to the path-loading circuit, producing a **non-product** path measure. `J=0` recovers the
classical independent tree exactly; `J>0` correlates consecutive moves (momentum) → **fatter tails
in the terminal price** → the **European** option reprices (a volatility-smile / tail-mispricing
effect). This is the genuinely-quantum, falsifiable content: delete `J` and the model collapses to
the classical CRR tree.

## 2. Scope

- **In:** one new module `quantum_pricer/herding.py`; a `J`-sweep driver folded into the results
  dashboard; a short narrative section in `context/quantum-investor.html` + `context/VISION.md`.
- **Core instrument:** **European call on Nokia** (the realistic, natural instrument; carries the
  fat-tails / smile business case).
- **Stretch:** **Asian** option, only to show the herding effect is *larger* on a path-dependent
  payoff (the literal "order matters" bonus). Built only if the core is solid.
- **Out (YAGNI):** barrier options, multi-asset, real hardware runs, deriving `J` from cognition
  data, training a loader. Stress-view (non-recentered) is a framing note, not extra code.

## 3. The herding layer (key technical decisions)

### 3.1 Must be non-diagonal — use controlled-`RY`, not `RZZ`
`RZZ(J)` is diagonal in the computational basis: it only adds phases and **leaves `p(x)`
unchanged** → it would not affect the price. The herding layer must mix amplitudes. We use a
**controlled-`RY(J)` ladder**: for each consecutive pair, `CRY(J)` controlled on path-qubit `i`,
target `i+1`. Reading: *if step `i` went up, nudge step `i+1`'s up-probability higher* =
momentum/herding. `J=0` ⇒ identity ⇒ exact CRR product measure.

This entangles the path register and makes the joint measure **order-dependent and non-product**
— the real quantum content.

### 3.2 Arbitrage-free re-centering (honest subtlety)
Herding changes `E[S_T]`, which **breaks the risk-neutral martingale** — the raw herded measure is
not an arbitrage-free pricing measure. The design re-imposes the forward:

- **Default (arbitrage-free):** after building the herded terminal distribution, apply a
  deterministic measure change / drift recenter so `E_herded[S_T] = S₀·e^{rT}` (the forward), then
  price. Tails stay fat; the mean is corrected → a valid risk-neutral price with fat tails.
- **Framing-only (stress view):** optionally present the un-recentered physical measure as a
  risk-desk stress scenario ("how does the price distribution change shape if the crowd herds?"),
  explicitly **not** an arbitrage-free quote. No extra code — a labelling choice in the dashboard.

### 3.3 Calibration anchor
`J` is **not** invented and **not** derived from cognition. It is calibrated to market data:
`calibrate_J_to_kurtosis(returns, M)` picks `J` so the model's terminal log-return **excess
kurtosis** matches Nokia's realized excess kurtosis. This gives `J` an empirical anchor (fat tails
are the market's own non-independence fingerprint).

## 4. Architecture — one module, maximal reuse

`quantum_pricer/herding.py`:
- `herded_loader(angles, J)` → circuit: `R_y(θᵢ)` loading **+** the `CRY(J)` neighbour ladder.
- `herded_path_probabilities(S0,K,r,sigma,T,M,J)` → `|⟨x|ψ⟩|²` from the statevector (the
  non-product measure), length `2^M`.
- `recenter_to_forward(probs, values_ST, target_forward)` → measure change so `E[S_T]=forward`.
- `price_herded(S0,K,r,sigma,T,M,J,option,kind,recenter=True)` → discounted `Σ p(x)·payoff(x)` by
  **direct statevector expectation** (small `M`). **Reuses `tree.payoff_variable_values`** for
  payoffs — only the probabilities change. `option ∈ {european, asian}`.
- `calibrate_J_to_kurtosis(daily_log_returns, M, ...)` → scalar `J`.

Driver (extend `quantum_pricer/make_results.py`):
- sweep `J ∈ [0, J_max]`; record European price vs `J` (and Asian for the stretch);
- plot **price-vs-`J`**, annotate `J=0` = classical and the kurtosis-calibrated `J`;
- show terminal-distribution kurtosis vs `J`; fold a labelled section into `results/dashboard.html`
  (same THEORETICAL/SIMULATED/REFERENCE/MARKET tagging discipline).

Narrative: a section in `context/quantum-investor.html` (and a §in `VISION.md`) that states the
bridge: cognition interference/order → correlated moves (herding `J`) → fat tails → European
repricing; with the `J=0`-litmus and the honest non-claims.

## 5. Honest claims / non-claims

- **Claim:** `J=0` reproduces the exact classical tree price to machine precision (regression).
- **Claim:** the quantum-essential ingredient is the non-product measure (entanglement); delete `J`
  → classical CRR. Passes the "delete the quantum part" litmus.
- **Claim:** `J` is anchored to Nokia's realized kurtosis (market data), not fitted to the option.
- **Non-claim:** we do **not** derive `J` from the cognition model — cognition *motivates* herding;
  calibration is to market data.
- **Non-claim:** no production quantum advantage; this is a small-`M` statevector demo of the
  *measure*, separate from the QAE/QSVT speed-up axis.

## 6. Validation / tests

1. **`J=0` regression:** `herded_path_probabilities(..., J=0)` == `tree.path_probabilities(...)` to
   1e-9; `price_herded(..., J=0)` == `tree.exact_tree_price(...)`.
2. **Monotone tails:** terminal-price excess kurtosis increases monotonically with `J` over the
   sweep range.
3. **Forward preserved:** after `recenter_to_forward`, `E[S_T] == S₀·e^{rT}` to 1e-9.
4. **Order-sensitivity (stretch):** the Asian price's fractional move under a given `J` exceeds the
   European's (herding bites harder on the path-dependent payoff).
5. **Smile direction:** for the recentered measure, OTM European call value rises with `J` (fat
   tails raise convex-payoff value) — a smile-like effect.

## 7. Deliverables

- `quantum_pricer/herding.py` (+ tests in `quantum_pricer/tests/`).
- `J`-sweep + plots folded into `quantum_pricer/make_results.py` → `results/dashboard.html`.
- Narrative bridge section in `context/quantum-investor.html` and `context/VISION.md`.

## 8. Risks

- **Martingale break** (addressed: §3.2 recenter). Verify test 3.
- **Statevector cost** at larger `M` — keep demo at `M ≤ ~8` (consistent with the existing pricer).
- **Parallel work:** a separate session owns `results/index.html`; this work stays additive
  (`dashboard.html` + new module), does not edit their rollup.
