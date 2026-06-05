# Idea candidates

Status legend: 💡 raw · 🔬 exploring · ✅ chosen · ❌ dropped

For each idea, fill the originality litmus test (strategy.md): *if we delete the quantum
part, does it collapse to a standard classical method?* The best ideas don't.

---

## A. Portfolio optimization (QAOA / VQE) — 💡
- **Problem**: pick assets/weights to maximize return for a risk budget, with constraints
  (cardinality, sector limits, transaction costs) → maps to QUBO / Ising.
- **Quantum-native angle**: discrete/combinatorial constraints (cardinality, integer lots)
  are *hard* classically and natural for QAOA. Lean into the combinatorial version, not the
  continuous mean-variance one (which is just convex QP — classical wins).
- **Risk**: very common hackathon pick. Need a twist to stand out (e.g. constraints that are
  genuinely NP-hard, or a novel mixer/ansatz, or warm-start from classical).
- **Data**: equity returns (Yahoo Finance / public indices).

## B. Option pricing via Quantum Amplitude Estimation — 💡
- **Problem**: price exotic/path-dependent options; QAE gives quadratic speedup over Monte
  Carlo in convergence (O(1/ε) vs O(1/ε²)).
- **Quantum-native angle**: the *quadratic speedup* is a textbook, defensible quantum
  advantage. Load a distribution into amplitudes, encode payoff, estimate expectation.
- **Strong fit** for "technical depth" + clear advantage narrative.
- **Risk**: distribution loading is the expensive part; be honest about it. Use a tractable
  payoff (European/Asian/barrier) on a small qubit count.

## C. Quantum generative model for volatility surfaces — 💡
- **Problem**: model/complete the implied-vol surface; classical surfaces are sparse & noisy.
- **Quantum-native angle**: qGAN / quantum circuit Born machine learns a distribution in a
  Hilbert space; sample-efficient generative modeling.
- **Risk**: hard to show clear advantage; training instability. High novelty though.

## D. Quantum ML — anomaly detection / risk premia — 💡
- **Problem**: detect fraud/anomalies or compute asset-pricing risk premia with a quantum
  kernel / variational classifier.
- **Quantum-native angle**: quantum feature maps reach feature spaces classically expensive
  to compute (quantum kernel). Frame around *which* kernels are hard classically.
- **Risk**: "quantum kernel SVM" is well-trodden; needs a finance-specific, defensible reason
  the quantum feature map matters.

## E. Our own twist (encouraged!) — 💡
Space to invent. Promising combinations:
- **QAE for risk measures** (VaR / CVaR), not just option price — directly relevant to OP's
  insurance + banking risk capital. Strong business case, clean quantum advantage story.
- **Portfolio + CVaR constraint** combining A and the risk-measure angle.
- Anything tied to OP's actual lines: **insurance reserving, credit risk, liquidity**.

---

## Cross-domain combinations (from `literature-review.md`) — 🔬
These pair two papers from very different fields so the quantum part is *essential*. Full
detail + citations + scoring table in `literature-review.md`. Ranked shortlist:

- **A. Tensor networks ⊕ quantum portfolio/risk** ✅ — import qMERA/qTTN topology from
  condensed-matter → *trainable-by-design* ansatz that dodges barren plateaus. Highest
  novelty+depth; likely white-space. (Martin-Plekhanov-Lubasch 2023 + Buonaiuto 2023)
- **B. Open-system noise physics ⊕ robust CVaR** ✅ — amplitude-damping "noise-induced limit
  set" reinterpreted as a model-uncertainty band on tail risk. Imperfection-as-feature.
  (Singkanipa-Lidar 2025 + Woerner-Egger 2019)
- **C. qGAN ⊕ QAE end-to-end pricing** ✅ — learn a fat-tailed distribution, load in poly(n)
  gates, price with QAE. **Lowest finish risk → our demoable core.** (Zoufal 2019 +
  Stamatopoulos 2020)
- **D. Epidemiology contagion ⊕ QAOA systemic risk** 🔶 — default cascade = epidemic on a
  graph → QAOA. Core to a systemic bank like OP.
- **E. Topological data analysis ⊕ quantum TDA** 💭 — Betti-number crash early-warning,
  exponential speedup. Max novelty, higher research risk; verify qTDA paper first.
- **F. Brain reservoir computing ⊕ quantum reservoir** 💭 — training-free quantum dynamics for
  volatility/regime forecasting; sidesteps barren plateaus entirely.

**Suggested play:** A or B for the wow-factor, C as the reliable demo — or fuse them
(tensor-net-structured qGAN→QAE pipeline, or a noise-aware CVaR engine).

---

## Decision (fill in at event)
- Chosen idea: ___
- Why it survives the originality litmus test: ___
- 24h scope: ___
- Stretch goal: ___
