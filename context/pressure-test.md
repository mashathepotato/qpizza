# Pressure-Test — Combos A, C, F (with 36h scope)

> Each combo verified by a dedicated research agent (web search + source fetch). Focus:
> **prior art / is it actually novel**, **what breaks**, **what's buildable in 36h**,
> **classical baseline**, **fallback**. Read the verdicts — two combos have novelty-killers.

## Scoreboard
| | Combo | Novel? | Finishable 36h? | Biggest risk | Recommendation |
|---|---|---|---|---|---|
| **A** | Tensor-net ansatz → portfolio/CVaR | ✅ **yes** (narrow, real) | ✅ comfortably | over-claiming "BP-free" | **TOP PICK** |
| **C** | qGAN→QAE fat-tail pricing | ⚠️ thin (mostly published) | ✅ yes | **library breakage** | reliable demo / fallback |
| **F** | Quantum reservoir → volatility | ❌ **already published** | ✅ easily | classical baseline wins | only if reframed |

---

## COMBO A — Tensor-network ansatz for portfolio / CVaR ✅ TOP PICK

**Pair:** Martin–Plekhanov–Lubasch 2023 (*Barren plateaus in quantum tensor network
optimization*, Quantum 7, 974) + Buonaiuto 2023 (VQE portfolio, Sci. Rep. 13:19434).

### Novelty — confirmed, but state it precisely
- ✅ **White-space confirmed**: no one has used a qTTN/qMERA *variational ansatz* for
  portfolio / mean-variance / VaR / CVaR. Targeted searches returned nothing.
- ⚠️ Do **NOT** say "tensor networks for finance is new" — it isn't. Mugel et al. (Phys. Rev.
  Research 4, 013006, 2022) use TNs as a *classical/quantum-inspired solver*. And qTTN/qMERA
  ansätze are established for *classification* (Pesah/Cerezo QCNN, PRX 2021, arXiv:2011.02966;
  TTN classifiers PRA 104, 042408).
- ✅ **Exact honest claim to use:** *"qTTN/qMERA ansätze are established for classification and
  their barren-plateau properties characterized; we are the first to benchmark them as the
  ansatz for the portfolio/CVaR variational problem."*

### The killer technical insight (this is our headline, not a footnote)
The favorable result (qTTN/qMERA gradient variance decays **polynomially**, not exponentially)
is proven **for cost Hamiltonians that are sums of LOCAL terms**, with variance shrinking by a
term's *distance from the network's canonical centre*. But the **mean-variance portfolio QUBO
is all-to-all dense** (O(N²) ZZ covariance couplings) — many terms sit far from any centre.
**So the clean advantage may NOT transfer to a dense financial cost.**
→ Turn this into the research question: *"Does the qTTN/qMERA barren-plateau advantage survive
an all-to-all portfolio/CVaR cost, or only local ones?"* That is genuine, defensible, and
exactly what physics judges respect. Also note: polynomial gradient variance ≠ guaranteed
convergence (gradients are *measurable*, not the landscape *benign*).

### 36h scope
- **Core (first ~18h):** PennyLane `qml.TTN` / `qml.MERA` templates (circuit construction is
  *not* hard — templates exist). Build N-asset QUBO (N=6–12 qubits, 1 qubit/asset). Three
  ansätze at matched param count: TTN, MERA, hardware-efficient (HEA, the BP-prone baseline).
- **Plot 1 (the money plot):** Var[∂C/∂θ] vs qubit count N (4→16-20), ~200 random-param
  samples. HEA falls ~exponentially; TTN/MERA ~polynomially — reproduces Quantum 7, 974 on a
  *financial* cost. **Report honestly if the all-to-all QUBO degrades the advantage — that's a
  finding, not a failure.** Sim is cheap (log-depth) → push to ~20+ qubits for the scaling plot.
- **Plot 2:** N=8–12 VQE/CVaR-VQE, cost vs iteration + approximation ratio vs brute-force optimum.
- **CVaR twist (cheap, high business value):** use CVaR aggregation of measured bitstrings as
  the objective (Barkoutsos et al.; Qiskit `08_cvar_optimization` tutorial) — ties ansatz story
  to a real OP risk metric.
- **Stack:** PennyLane (`lightning.qubit`), Qiskit-Finance to *generate* the QUBO, cvxpy/yfinance.

### Classical baseline (mandatory honesty)
Exact brute force for N≤~20 (true optimum → real approximation ratio); Markowitz via
cvxpy/PyPortfolioOpt; simulated annealing (`dwave-neal`) on the same QUBO; optional DMRG/TN
classical solver. **Frame as a trainability/landscape study, NOT a speedup** — classical wins
wall-clock at these sizes; say so.

### Data
yfinance daily adj-close, ~12–18 tickers (use **OMX Helsinki** names — nice OP/Finland nod),
2–3 yrs → log-returns → μ, Σ → QUBO H = q·xᵀΣx − μᵀx + λ(Σxᵢ − k)².

### Verdict + fallback
**Finishable, comfortably.** Physics owns ansatz+BP interpretation; CS owns data/QUBO/baseline/
plots. **Biggest risk = over-claiming** ("BP-free for portfolios" → judge asks about the
all-to-all assumption). Mitigation: make that the headline finding. **Killer fallback:** ship
*only Plot 1* — gradient-variance scaling of qTTN/qMERA vs HEA on a real portfolio QUBO,
4→20 qubits. No full VQE convergence needed, cheap, striking, extends a published result into
an unstudied cost class, defensible on all four axes.

---

## COMBO C — qGAN → QAE fat-tail option pricing ⚠️ buildable, novelty thin

**Pair:** Zoufal 2019 (qGAN loading, arXiv:1904.00043) + Stamatopoulos 2020 (QAE pricing,
Quantum 4, 291).

### Novelty — mostly already published (cite these YOURSELF, pre-empt the judge)
- ❌ "qGAN learns non-log-normal dist" — Zoufal already did bimodal/triangular targets.
- ❌ "qGAN+QAE prices a European call" — that's the **Qiskit Finance tutorial** verbatim.
- ❌ "barrier/path-dependent via QAE" — in Stamatopoulos 2020 (arXiv:1905.02666).
- ❌ "regime-switching + QAE pricing" — **Ghysels, Morgan & Mohammadbagherpoor 2023/2024**
  (arXiv:2311.00825, IEEE QCE 2024) did exactly this *and explicitly discuss qGAN loading*.
- ✅ **Genuine (thin) novelty line:** the *integrated, data-driven* pipeline — qGAN-learned
  **fat-tail / regime-switching** dist → QAE → **path-dependent (barrier/Asian)** payoff, fit
  to real OP-relevant data, with **tail-fidelity metrics** (tail-KL / tail-quantile error,
  under-reported) and an **honest sample-complexity** comparison. It's an *integration +
  empirical-characterization* contribution, not algorithmic. Pitch it that way or get sunk.
- Bonus depth: document qGAN **training failure on the tail** (mode collapse) as a negative
  result — judges reward honesty.

### ⚠️ Biggest risk = LIBRARY BREAKAGE (read before coding)
- The old `qiskit_machine_learning.algorithms.QGAN` class was **deprecated 0.5.0 and REMOVED
  in 0.7.0** — no drop-in replacement. Use the hand-rolled **PyTorch qGAN** (`SamplerQNN` +
  `TorchConnector` generator + MLP discriminator + manual Adam loop) per the current
  `04_torch_qgan` tutorial.
- `qiskit.algorithms` was split out → import QAE from standalone **`qiskit-algorithms`**
  (`IterativeAmplitudeEstimation`, `EstimationProblem`), consuming Aer primitives.
- **Any tutorial older than ~2023 calling `QGAN(...)` will not run.** **Pin all versions in a
  fresh venv on hour 0 and smoke-test the full import chain in hour 1.** This is the #1 way the
  hack dies at hour 30.
- Use **IQAE** (no QFT, shallow), not canonical QPE-based AE.

### 36h scope
- **MVP (~first 18h):** 1D, **3-qubit** (8-bin) generator, train on a fat-tail target
  (Student-t / 2-state mixture), load → QAE European call. (This reproduces the tutorial = a
  *checkpoint*, not the deliverable.) qGAN: shallow ansatz, multiple seeds, expect minutes ×
  restarts (no convergence guarantee).
- **Twist (~next 12h):** swap payoff to a **barrier (knock-in/out)** or tail/CVaR payoff via
  `LinearAmplitudeFunction`/comparator; instrument **tail fidelity** + deep-OTM strike.
- **Plots:** learned vs target PMF (tail inset); training curves incl. a *failed* run; QAE vs
  analytic BS + classical MC **with error bars**; **error vs #queries** log-log (QAE ∝1/N vs
  MC ∝1/√N) — the only legitimate "advantage" visual.
- **Qubits:** 3 (dist) + 1 (payoff) + few ancilla ≈ 4–7. Don't scale for show (buys BPs).

### Classical baseline + honest framing
Analytic Black-Scholes (correctness ground truth) + classical Monte Carlo on the same dist.
**NEVER claim wall-clock speedup** (sim is slower than MC). Only honest claim: *query-model*
scaling 1/N vs 1/√N (Montanaro), asymptotic + fault-tolerant only.

### Data
**Synthetic primary** (Merton/Kou jump-diffusion, Bates/Heston, or regime-switching GBM — known
ground-truth tail to validate the qGAN). **Real garnish:** yfinance daily log-returns of an
index/FX, show fat tails (kurtosis/QQ-plot), fit qGAN to that — makes it "OP", not "textbook".

### Verdict + fallback
**Finishable** if versions pinned + PyTorch qGAN + 1D/3-qubit. Defensible as integration +
honest characterization, **not** algorithmic novelty. **Fallback (wire early):** if qGAN won't
train, **load the fat-tail dist analytically** (`StatePreparation`/qiskit-finance loaders) and
keep the full QAE barrier-pricing + scaling story — lose only the "learned loading" claim, keep
~80% of the demo, present the qGAN failure as a documented finding. **Project cannot fully fail.**

---

## COMBO F — Quantum reservoir computing for volatility ❌ already published

**Pair:** Fujii–Nakajima 2017 (QRC foundation, arXiv:1602.08159) + a volatility/finance app.

### Novelty — the exact idea is already peer-reviewed (this is decisive)
- ❌ **Li, Mukhopadhyay, Bayat & Habibnia, "Quantum Reservoir Computing for Realized Volatility
  Forecasting," arXiv:2505.13933 (May 2025), published in Phys. Rev. Research (2026)** — uses a
  transverse-field Ising reservoir with distinct input/memory qubits, forecasts **realized
  volatility**, benchmarks vs HAR/HARX/FFN/LSTM/classical RC, claims it "consistently
  outperforms." *(This is literally one of the sources our lit-review search surfaced.)*
- ❌ Also: stock-trend QRC (arXiv:2602.13094, claims >86% but weak/promotional, no baseline).
- ✅ The brain/echo-state→quantum-reservoir framing is *historically legit* (Jaeger ESN 2001,
  Maass liquid-state machines 2002 = neuroscience) — but it's inspiration, not a theorem.
- **Verdict: cannot claim "first QRC for volatility/finance."** Novelty must shift to:
  **(a) regime-switch *detection*** (less covered than RV regression), **(b)** a clean
  **QRC-vs-matched-classical-ESN** study isolating what the quantum part buys at fixed scale,
  or **(c)** interpretability/feature-importance. Lead with problem framing + training-free
  narrative, citing Li et al. as launch point (a "critical replication + extension").

### The danger zone — classical baseline likely wins
**GARCH(1,1) / HAR-RV are notoriously hard to beat**, and a **classical ESN at matched scale
will likely match or beat** a 6–8 qubit reservoir and runs instantly. Honest framing: QRC is
interesting because it's **training-free (no barren plateaus), expressive per-qubit (~5–7 qubits
≈ hundreds of classical nodes claimed), natural for temporal data** — frame results as
*"competitive with classical RC at comparable scale, path to advantage as qubits grow,"* **NOT
"beats GARCH."** Claiming the latter on a 7-qubit sim gets demolished.

### 36h scope (easily finishable)
- **Easiest sim route:** density-matrix in NumPy/PennyLane (`default.mixed`) — inject input via
  partial state replacement, evolve `expm(-iHτ)`, read ⟨Zᵢ⟩, carry state forward. No
  shots/mid-circuit headaches; 6–8 qubits = ≤256×256 matrix.
- **Reservoir:** fixed random fully-connected TFIM (seeded). **Temporal multiplexing** (V≈10
  virtual nodes/step) is *essential* for enough features. Wash-out transient for echo-state /
  fading-memory (pure unitary has perfect memory → needs re-injection/dissipation — own this).
- **Readout:** scikit-learn `Ridge` on ⟨Zᵢ⟩ (+ optional ⟨Xᵢ⟩,⟨ZᵢZⱼ⟩) features.
- **Plots:** pred vs actual RV; QRC vs HAR vs ESN vs GARCH (QLIKE/RMSE); qubit/virtual-node
  sweep; regime-detection overlay (stretch).
- **Templates:** QHACK23 PennyLane QRC repos, QuantumReservoirPy.

### Data
yfinance (SPY/^GSPC, ^VIX). **RV caveat:** true RV needs intraday returns; **Oxford-Man
Realized Library is discontinued** — use yfinance 1-min (~30d) for Σr², or a proxy (squared
daily returns, Garman-Klass/Parkinson from OHLC, or predict ^VIX). Regimes: threshold RV or fit
a 2-state Gaussian HMM / Markov-switching model.

### Verdict + fallback
**Finishable easily, team fit ideal**, but **novelty is the problem** (peer-reviewed prior art).
**Only pursue if reframed** as: training-free quantum temporal learning + honest baseline study
+ regime-detection extension + interpretability, *citing Li et al. as motivation* (critical
replication, not discovery). **Fallback framing:** "a controlled study of *when* quantum
reservoir dynamics help vs matched classical reservoirs — value is training-free expressivity /
BP avoidance, not raw accuracy at NISQ scale." Honest, deep, judge-proof — but it's a *study*,
not a *win*.

---

## Bottom line / recommendation
- **Combo A is the strongest bet**: real (narrow) novelty, ideal physics+CS split, a headline
  research question (does the BP advantage survive a dense financial cost?), and a cheap
  can't-fail fallback (the single gradient-variance plot). Best on novelty + technical depth +
  problem formulation; business case = trainability is *the* blocker for scaling quantum
  portfolio optimization.
- **Combo C is the safe demo** — reliable, visual, end-to-end — but thin on novelty and exposed
  to library breakage; use as core/fallback, framed as integration + honest characterization.
- **Combo F only with a reframe** — the headline application is already peer-reviewed; pivot to
  regime-detection + a critical QRC-vs-ESN study or drop it.
- **Strongest single play:** Combo A as the hero. **Strongest fused play:** A's tensor-network
  ansatz structure applied inside C's pricing/risk pipeline (TN-structured loader/ansatz for a
  CVaR objective) — novel *and* demoable, if time allows.
