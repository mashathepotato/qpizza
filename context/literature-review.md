# Literature Review — Quantum Computing for Finance + Cross-Domain Idea Combinations

> Produced by a fan-out deep-research pass (6 search angles, 25 sources fetched, 25 claims
> adversarially verified — 22 confirmed, 3 refuted). Confidence tags below reflect that
> verification. **Read the "Honest framing" box before pitching anything.**

---

## TL;DR for the team
1. Quantum finance has **three well-mapped pillars**: portfolio optimization (QAOA/VQE →
   QUBO/Ising), derivative pricing + risk (QAE → quadratic speedup over Monte Carlo), and
   QML (kernels / QNNs / qGANs, advantage unproven).
2. Because all four judging criteria are equal, our edge is a **quantum-native cross-domain
   bridge** — import structure from another field (condensed-matter tensor networks,
   open-system noise physics, topology, epidemiology, neuroscience) so the quantum part is
   *essential*, not bolted on.
3. The single richest, most defensible technical seam is **trainability** (barren plateaus /
   noise-induced barren plateaus). It's a real obstacle *and* the natural doorway into
   physics most teams won't touch.

---

## ⚠️ Honest framing (do not skip — OP's judges are real quantum researchers)
- **QAE/QML speedups are ASYMPTOTIC and fault-tolerant.** Resource estimates need thousands
  of logical qubits + huge T-depth. **No near-term practical speedup exists today.** Frame
  business value as: hybrid pipelines, better ansatz/algorithm *design*, and future-readiness
  — never "we beat Monte Carlo today."
- The 2026 consensus (Gong et al.) is explicit: advantage is **conditional and hybrid**.
  QAOA/VQE matter *when constrained combinatorial search dominates*; QAE matters *when
  repeated expectation evaluation is the binding cost*; QML is *task-dependent*. A credible
  pitch states which regime its problem is in.
- **Three claims were REFUTED in verification — do NOT assert them:**
  - The qGAN paper does **not** demonstrate a quadratic QAE pricing advantage — cite it only
    for `O(poly(n))` data loading.
  - NIBP does **not** universally forbid all QAOA/VQE finance proposals — the result is
    conditioned on ansatz depth growing linearly with qubit count.
  - One survey (2204.10026) does **not** name VQE/QAOA as its core portfolio algorithms.

---

## State of the art — the three pillars

### Pillar 1 — Portfolio optimization (combinatorial)
- **Buonaiuto et al. 2023**, *"Best practices for portfolio optimization by quantum
  computing, experimented on real quantum devices"*, Sci. Rep. 13:19434. ✅ verified.
  Constrained mean-variance → QUBO via binary encoding + penalty constraints; solved with VQE
  on real IBM hardware. Contribution = empirical best ansatz + optimizer choices.
  → Lean into the **discrete/combinatorial** form (cardinality, integer lots, sector caps).
  The continuous mean-variance problem is just convex QP — classical wins there.

### Pillar 2 — Option pricing & risk via Quantum Amplitude Estimation
- **Stamatopoulos et al. 2020**, *"Option Pricing using Quantum Computers"*, Quantum 4, 291.
  ✅ verified. QAE gives a **quadratic speedup** (`O(1/ε)` vs `O(1/ε²)`) over Monte Carlo;
  covers vanilla, multi-asset, and path-dependent/barrier options. Theory root: Montanaro 2015.
- **Woerner & Egger 2019**, *"Quantum Risk Analysis"*, npj Quantum Information 5:15.
  ✅ verified. QAE for **VaR and CVaR** on gate-based hardware; near-quadratic speedup. This
  is the seminal QAE-for-risk reference — directly relevant to OP's banking/insurance lines.

### Pillar 3 — Quantum ML & generative models
- **Zoufal, Lucchi, Woerner 2019**, *"Quantum Generative Adversarial Networks for learning
  and loading random distributions"*, npj QI. ✅ verified. A **qGAN** (quantum generator +
  classical discriminator) loads an arbitrary distribution into an n-qubit state with
  `O(poly(n))` gates instead of `O(2^n)` — this cracks the **data-loading bottleneck** that
  otherwise kills QAE's advantage. (Caveat: loading is approximate; qGAN training itself
  hits barren plateaus.)
- QML supervised methods (quantum kernels, variational classifiers, QNNs) are surveyed but
  carry **no blanket proven advantage** — must be justified task-by-task.

### The cross-cutting obstacle — trainability
- **Wang et al. 2021**, *"Noise-induced barren plateaus in variational quantum algorithms"*,
  Nat. Commun. 12:6961. ✅ verified. Under local Pauli (unital) noise the cost gradient
  vanishes **exponentially in qubit count** when ansatz depth grows linearly → NIBPs.
- **Singkanipa & Lidar 2025**, *"Beyond unital noise in variational quantum algorithms"*,
  Quantum (Jan 30 2025). ✅ verified. **Non-unital** (Hilbert-Schmidt-contractive) noise —
  which *includes physically realistic amplitude damping* — does NOT necessarily cause
  barren plateaus; instead it yields **"noise-induced limit sets"** where cost converges to a
  *range* of values.

---

## ★ The main event — cross-domain paper combinations

Each pairs two papers from **very different fields** that synergize into one quantum-native
idea. Tags: ✅ = both papers verified by research; 🔶 = grounded but one link is my synthesis;
💭 = speculative, verify the second paper before relying on it.

### Combo A — Condensed-matter tensor networks ⊕ quantum portfolio/risk ✅
**Pair:** Martin, Plekhanov & Lubasch 2023, *"Barren plateaus in quantum tensor network
optimization"* (Quantum 7, 974) **+** Buonaiuto et al. 2023 (VQE portfolio QUBO, Sci. Rep.).

**The bridge:** In condensed-matter physics, tensor-network *topology* controls trainability:
qMPS gradients decay **exponentially** (barren), but hierarchical **qTTN and qMERA decay only
polynomially** (trainable at scale) — variance shrinks with a Hamiltonian term's distance
from the network's "canonical centre". Import the MERA/TTN architecture from condensed-matter
into the *portfolio/risk ansatz design* → a **trainable-by-design** quantum portfolio engine
that dodges barren plateaus, instead of the usual hardware-efficient ansatz that doesn't.

**Why it's quantum-native:** the entanglement-structure-as-architecture argument is pure
many-body physics; delete the quantum part and there's nothing left. Plays directly to your
2 physicists. **Open white-space** (per research): no one has clearly benchmarked a
MERA/TTN-structured ansatz for VaR/CVaR or portfolios on real financial data.
*Business case:* trainable ansätze = the difference between a quantum risk model that scales
and one that flatlines — future-readiness for OP's quantum unit.

### Combo B — Open quantum systems noise physics ⊕ robust risk (VaR/CVaR) ✅
**Pair:** Singkanipa & Lidar 2025, *"Beyond unital noise…"* (Quantum) **+** Woerner & Egger
2019, *"Quantum Risk Analysis"* (npj QI).

**The bridge:** Reframe a hardware *imperfection as a feature*. Non-unital amplitude-damping
noise produces a **"noise-induced limit set"** — the cost settles into a *band* of values, not
a point. Reinterpret that band as a **model-uncertainty / robustness envelope on CVaR**: the
noise spread becomes an ensemble view of tail risk rather than an error to suppress.

**Why it's quantum-native:** the open-system master-equation behaviour *is* the financial
quantity — you literally couldn't state this idea classically. Deeply original, very on-brand
for "rethink the foundations." *Risk:* needs a rigorous financial interpretation of the limit
set (see open questions) — present as a research direction with a clean toy demonstration.

### Combo C — qGAN distribution learning ⊕ QAE pricing (end-to-end) ✅
**Pair:** Zoufal et al. 2019 (qGAN data loading) **+** Stamatopoulos et al. 2020 (QAE option
pricing).

**The bridge:** The classic data-loading wall is what makes QAE impractical. Chain them:
qGAN *learns* the asset-price distribution from market data and loads it in `O(poly(n))`
gates → QAE estimates the expected payoff on top. A complete, runnable pipeline from real
data to a price. *Twist to stand out:* learn a **fat-tailed / regime-switching** distribution
(not log-normal) so the generative step earns its keep, then price a path-dependent option.
**Honesty rule:** cite the qGAN for *loading*, not for a speedup claim (that framing was
refuted). Most "finishable in 36h" of the three — good fallback / core demo.

### Combo D — Epidemiology contagion ⊕ quantum systemic-risk optimization 🔶
**Pair:** a financial-contagion / SIR-on-networks paper (research surfaced epidemiology↔finance
crossover sources, e.g. arXiv:2303.12601 and PMC9998608) **+** QAOA-on-graphs (the
combinatorial-optimization machinery from Pillar 1).

**The bridge:** Default/liquidity contagion in an interbank network is mathematically an
**epidemic spreading on a graph**. Cast "which capital buffers minimize systemic cascade" as
a constrained graph optimization → QAOA. Imports the **SIR/network-epidemic** formalism from
epidemiology into systemic-risk management.

**Why it fits:** combinatorial search on a network is exactly QAOA's credible regime, and
systemic risk is core to a systemically-important bank like OP. *Status:* verify the specific
epidemiology paper's mapping before committing — the QAOA half is solid.

### Combo E — Topological data analysis ⊕ quantum crash detection 💭
**Pair:** a market-topology / persistent-homology crash-detection paper (TDA has documented
use detecting pre-crash signatures via changing market topology) **+** **quantum TDA**
(Lloyd, Garnerone & Zanardi, *"Quantum algorithms for topological and geometric analysis of
data"*, Nat. Commun. 2016 — exponential speedup for Betti numbers).

**The bridge:** Persistent homology spots topological change in the correlation network
*before* a crash; quantum TDA computes Betti numbers exponentially faster. A
**quantum-accelerated early-warning system** for market instability.

**Why it's striking:** topology + quantum + finance is maximally "different fields," very high
novelty. *Status:* 💭 the quantum-TDA paper is well-known from domain knowledge but was **not**
in the verified source set — confirm it (and recent NISQ-friendly qTDA variants) before
building. Strong story, higher research risk.

### Combo F — Neuroscience reservoir computing ⊕ quantum volatility/regime modeling 💭
**Pair:** a reservoir-computing / recurrent-dynamics paper (brain-inspired computation where a
fixed random dynamical "reservoir" does the heavy lifting) **+** **quantum reservoir
computing** applied to financial time series (volatility / regime detection).

**The bridge:** A quantum system's natural many-body dynamics *is* the reservoir — no
variational training of the quantum part, which **sidesteps barren plateaus entirely**.
Use it for volatility forecasting / regime-switch detection. This is the closest analogue to
your "neural networks in the brain + quantum for finance" example.

**Why it's appealing:** trainability-free is a genuine NISQ advantage, and it's an unusual,
memorable angle. *Status:* 💭 verify the specific quantum-reservoir-computing-for-finance
literature before committing.

---

## How the combos map to the team & to scoring
| Combo | Quantum-native? | Physics depth | CS/demo load | Finish risk | OP business hook |
|---|---|---|---|---|---|
| A (tensor nets) | ★★★ essential | high | medium | medium | scalable risk/portfolio engine |
| B (noise→risk) | ★★★ essential | high | low–med | med-high | robust CVaR / model uncertainty |
| C (qGAN+QAE) | ★★ strong | medium | medium | **low** | faster derivative/risk pricing |
| D (contagion) | ★★ strong | medium | medium | medium | **systemic risk** (core to OP) |
| E (quantum TDA) | ★★★ essential | high | high | high | crash early-warning |
| F (q reservoir) | ★★ strong | high | medium | med-high | volatility/regime forecasting |

**Recommendation to discuss:** **Combo A or B** for maximum novelty+depth (best fit for a
2-physicist team and OP's researcher audience), with **Combo C as the safe, demoable core** —
or fuse them: a **tensor-network-structured (A) qGAN→QAE pricing pipeline (C)**, or a
**noise-aware (B) CVaR engine**. Decide in `ideas.md`.

---

## Open questions the research flagged (these are our differentiators if answered)
1. Has anyone built a **MERA/TTN-structured ansatz for portfolio/VaR/CVaR** and benchmarked it
   vs hardware-efficient VQE on real financial data? (Likely white-space.)
2. Can the **NILS spread under amplitude damping** be given a rigorous financial meaning
   (a robustness band on CVaR) carrying real information, not just noise floor?
3. For OP's realistic problem sizes, is the binding cost **combinatorial search** (→ QAOA/VQE)
   or **repeated expectation evaluation** (→ QAE)? Advantage is entirely contingent on this.
4. What's the **best classical baseline** (tensor-network / Monte Carlo / commercial QUBO) for
   our chosen task, and can the quantum-native method credibly compete at near-term scale?

## Key references (verified unless noted)
- Herman et al. 2022, *A Survey of Quantum Computing for Finance*, arXiv:2201.02773.
- Gong et al. 2026, *Quantum Computing for Financial Transformation*, arXiv:2604.08180. (single recent review — hybrid/conditional framing)
- Stamatopoulos et al. 2020, *Option Pricing using Quantum Computers*, Quantum 4, 291.
- Woerner & Egger 2019, *Quantum Risk Analysis*, npj QI 5:15.
- Buonaiuto et al. 2023, *Best practices for portfolio optimization…*, Sci. Rep. 13:19434.
- Zoufal, Lucchi, Woerner 2019, *qGANs for learning and loading random distributions*, npj QI, arXiv:1904.00043.
- Wang et al. 2021, *Noise-induced barren plateaus…*, Nat. Commun. 12:6961.
- Singkanipa & Lidar 2025, *Beyond unital noise in VQAs*, Quantum (q-2025-01-30-1617).
- Martin, Plekhanov, Lubasch 2023, *Barren plateaus in quantum tensor network optimization*, Quantum 7, 974.
- 💭 (verify) Lloyd, Garnerone, Zanardi 2016, *Quantum algorithms for topological and geometric analysis of data*, Nat. Commun.
- 🔶 (verify mapping) epidemiology↔finance contagion: arXiv:2303.12601; PMC9998608.
