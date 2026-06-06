# Quantum Option Pricer + 3-Way Benchmark on Nokia — Design

**Date:** 2026-06-06
**Status:** Approved direction (brainstorming complete)
**Source paper:** `paper/main_V2.tex` — "Quantum finance — Complete procedure for option pricing" (QNDM + Fourier / QAE / QSVT)
**Vision:** `context/VISION.md` — "The Madness of People Is Quantum" → quantum option pricer is the technical core
**Hackathon:** Junction Helsinki × OP Pohjola — Quantum Computing for Finance. Judged 25% each: novelty · problem formulation · technical depth · business case.

---

## 1. Goal

Implement the option-pricing procedure of `paper/main_V2.tex` and use it to run a **three-way benchmark** on a real underlying (Nokia), reproducing empirically the paper's own §"Comparative analysis and scaling":

1. **Classical SOTA** — Monte Carlo on the binomial tree (the "Markov chain" of GBM price paths). `O(1/ε²)`.
2. **SOTA quantum** — the established QNDM approach (the paper *minus its last section*): the **Fourier route** (§5a) as the spine, plus the **amplitude-QAE route** (§5b). `O(1/ε)`.
3. **Our novel approach** — the paper's **last section**: **QNDM-powered single-run QAE via QSVT** (§5c). Phase oracle + polynomial ReLU + one amplitude estimation; no λ-sweep, no price register; fewest qubits. `O(1/ε)`. **This is the centerpiece.**

The shared win across all quantum routes: the CRR binomial tree has a **product** path measure, so the full `2^M`-path superposition `Σ_x √p(x)|x⟩` loads **exactly** with `M` single-qubit `R_Y(θ_i)` rotations — no costly distribution-loading oracle (the dominant cost of oracle-QAE / Stamatopoulos 2020).

## 2. Scope decisions (locked)

| Decision | Choice |
|---|---|
| Data source | Real underlying, σ estimated from historical log-returns |
| Underlying | **Nokia (`NOKIA.HE`)**, via `yfinance` |
| Validation ground truth | **Exact binomial-tree price** (classical full enumeration at small M); Black–Scholes is the *continuum* sanity check |
| Quantum hardware | Both quantum routes target Q50 via `q50_fake → q50_hw`; classical stays classical; NISQ depth limits reported honestly |
| Asian (path-dependent) option | **Nice-to-have** — protect the European 3-way benchmark first |
| QSVT route | **Centerpiece** — push hard, including its Q50 run as the headline novel result |
| Stack | **Qiskit** (only path to Q50 via `iqm.qiskit_iqm`; reuses existing `backends/` ladder) |

The PennyLane `quantum_investor` cognition demo is **untouched** — it remains the motivational opener, not part of this build.

## 3. The ground-truth subtlety (keeps the benchmark honest)

The quantum routes estimate the **binomial-tree expectation** `E_Q[max(f−K,0)]` under the tree measure. Therefore the validation target is the **exact tree price** (enumerate all `2^M` paths classically at small M, or backward induction on the recombining lattice), **not** Black–Scholes. BS is reported separately as the continuum reference, and tree→BS convergence is its own `O(1/M)` discretization story. This prevents attributing lattice discretization bias to the quantum machinery.

## 4. Architecture — new package `quantum_pricer/`

| File | Purpose | Depends on |
|---|---|---|
| `data.py` | `yfinance` `NOKIA.HE` → daily log-returns → annualized realized σ; emit `S₀, r, σ` params | yfinance |
| `tree.py` | CRR tree: `u,d`, risk-neutral `q_i = (e^{rΔt}−d)/(u−d)`, `θ_i = 2·arcsin√q_i`; **exact tree price** by enumeration (ground truth) | numpy |
| `classical.py` | Black–Scholes closed form (continuum check) + Monte-Carlo "Markov chain" pricer (`O(1/ε²)`) | numpy, scipy |
| `fourier.py` | **SOTA-quantum spine**: QNDM Fourier route — load paths + detector `|+⟩` + diagonal phase `e^{iλ(f−K)}` + X/Y-basis measure → `Re/Im G(λ)` → Gil-Peláez / FFT inversion | qiskit |
| `qae.py` | QNDM amplitude-QAE route: build `𝒜` (load + reversible payoff `ℱ` → work register + controlled-`R_Y` into target + uncompute), run `IterativeAmplitudeEstimation` | qiskit, qiskit-algorithms |
| `qsvt.py` | **Novel centerpiece**: QNDM phase oracle `O = diag(e^{ic(f(x)−K)})` + `pyqsp` ReLU polynomial → signal ancilla amplitude → single QAE on that ancilla | qiskit, pyqsp |
| `backends.py` | reuse triage-lab backend abstraction (`local_aer / lumi_aer / q50_fake / q50_hw`); transpile with `basis_gates=["r","cz"]` | qiskit-aer, iqm.qiskit_iqm |
| `benchmark.py` | budget sweep across contenders → error-vs-queries log-log plot + empirical resource table | matplotlib |
| `tests/` | self-checks: each route's price matches the exact tree price on statevector sim within tolerance, before any benchmark claim | pytest |
| `demo.py` / notebook | runs top-to-bottom for the live demo | — |

**Phase-oracle sharing:** `fourier.py` and `qsvt.py` use the *same* diagonal phase accumulation (`e^{iλ(f(x)−K)}` vs `e^{ic(f(x)−K)}` at a single coupling). Build the diagonal-phase primitive once (in `fourier.py` or a shared `phase_oracle.py`) and reuse it in `qsvt.py`. Building Fourier first therefore de-risks QSVT.

**Small-M phase encoding:** at `M=1` the European payoff phase is a single controlled-phase on the detector (shallow → ideal first Q50 run). For `M=2,3` on a recombining tree, use the **Hamming-weight trick** (paper §3 remark): `S_t` depends only on the up-move count `w`, stored in `⌈log₂(M+1)⌉` qubits, phase conditioned on `w` — avoids full multipliers.

## 5. The three contenders, concretely

1. **Classical MC** — sample `N` paths (Bernoulli(`q_i`) per step), average `max(f−K,0)`, discount `e^{−rT}`. Statistical error `~σ_MC/√N`.
2. **SOTA-quantum (QNDM):**
   - *Fourier:* per λ in a grid, run load+detector+phase+strike, measure detector in X (→ Re G) and Y (→ Im G); FFT/Gil-Peláez invert `G(λ)` → `E[max(f−K,0)]`.
   - *Amplitude-QAE:* build `𝒜` per paper §5b Fig. 1, run `IterativeAmplitudeEstimation`, recover `a = E[max(f−K,0)]/C_max`.
3. **Novel (QSVT):** package QNDM accumulation at single coupling `c` as phase oracle `O`; `pyqsp` produces phases `{φ_k}` for a degree-`d` polynomial `p(θ) ≈ √(max(f−K,0)/C_max)` (linear encoding); interleave controlled-`O` with `R_Z(φ_k)` on a signal ancilla; single QAE on the ancilla → price. No λ-sweep, no price register.

## 6. Benchmark outputs (the two money slides)

- **Error-vs-queries log-log plot.** x = number of queries (MC: `N` samples; quantum: oracle/`𝒜` calls = QAE precision or λ-grid size). y = absolute error vs exact tree price. Expected slopes: classical ≈ −½, quantum ≈ −1 — the visual proof of the quadratic speed-up.
- **Empirical resource table** (reproduces paper §scaling): qubits, two-qubit (CZ) depth after transpile to `{r,cz}`, state-prep gate count, queries-to-ε — per contender, at growing `M`.

## 7. Q50 ladder & build staging (risk-managed)

Each stage gated on the previous passing its exact-tree-price self-check:

1. `data.py` + `tree.py` + `classical.py` → Nokia params; exact tree / BS / MC prices agree.
2. `fourier.py` European `M=1→3` on `local_aer` → matches exact tree price. (shallow anchor)
3. Same on `q50_fake` (noisy IQM `IQMFakeAphrodite`), then `q50_hw` on-site → the live-hardware moment.
4. `qsvt.py` on `local_aer` (centerpiece) → shows qubit / no-λ-sweep advantage; then `q50_fake` / `q50_hw`, **honest NISQ-depth caveat** if noise-limited. `qae.py` similarly.
5. **Nice-to-have:** Asian (`f = S̄`) — same circuits, one scalar change, validated vs MC.

**QSVT fallback:** if QSVT is not hardware-solid by deadline, the statevector-sim result (qubit/no-λ-sweep advantage) is still the novelty headline; the Q50 run is best-effort with an honest caveat. QSVT must not block the Fourier Q50 demo.

## 8. Testing & honesty

- **TDD** (per `superpowers:test-driven-development`): the exact-tree-price recovery test is written *before* each route's circuit.
- Every route validated on statevector sim against the exact tree price within tolerance before any benchmark claim.
- Real-data provenance (Nokia, date range, σ estimate) printed in the demo; any synthetic fallback labelled as such.
- Hardware results reported with shot counts and noise caveats; no hiding NISQ-depth degradation.

## 9. Out of scope

- American / early-exercise options.
- Multi-asset / basket options.
- Dividend modelling beyond a flat continuous yield (if needed).
- Training a distribution-loading oracle / qGAN (the whole point is we don't need one).
- Any modification to the `quantum_investor` cognition demo.

## 10. How this scores

- **Novelty:** the QSVT single-run route (our last section) on real hardware + free exact path loading — distinct from textbook oracle-QAE.
- **Problem formulation:** price = discounted `E[max(f−K,0)]` over a quantum path superposition; falsifiable quadratic-speed-up claim measured directly.
- **Technical depth:** three contenders, empirical error-vs-queries + resource table, honest NISQ limits, run on Q50.
- **Business case:** Nokia options are OP-relevant; Asian + time-dependent parameters are the genuinely hard, high-value regime where the advantage bites.
