# VISION — "The Madness of People Is Quantum" → Quantum Option Pricing (LOCKED DIRECTION)

> **This is the chosen project.** Single source of truth for the team and for any AI agent
> working in this repo. Self-contained: read this and you can contribute without reading
> anything else first. Deeper detail in [`newton-feynman.md`](./newton-feynman.md); visual
> explainer in [`quantum-investor.html`](./quantum-investor.html); the full pricing math in
> [`../paper/main_V2.tex`](../paper/main_V2.tex); motivation/cognition code in
> [`/quantum_investor`](../quantum_investor).

> **PIVOT (current direction).** Two layers:
> 1. **Motivation —** "The Madness of People Is Quantum": markets are non-classical, so classical
>    intuition (Newton) is the wrong tool. Human financial decisions follow *quantum* probability.
> 2. **Technical core —** because the market is quantum-like, we stop fighting it and **compute with
>    it**: a **quantum option pricer** that loads every price path into a real superposition and reads
>    the fair price off it (QNDM phase encoding + QAE/QSVT), beating classical Monte Carlo
>    quadratically with **no costly distribution-loading oracle**. Math is fully worked out in
>    `paper/main_V2.tex`; the pricing code is the build target.

**Hackathon:** Junction Helsinki × OP Pohjola — Quantum Computing for Finance.
**Team:** 4 people (2 CS + 2 physics). **Judged 25% each:** novelty · problem formulation ·
technical depth · business case. **Submission:** tomorrow 10:00 AM.

---

## 1. The one-sentence pitch
> **Newton said he could not predict "the madness of people"; Feynman said you cannot understand
> the world with classical intuition. Both are the same statement: markets are non-classical. So
> instead of fighting that, we *compute with it* — we load the whole tree of future price paths into
> a quantum superposition and price options (European and path-dependent Asian) by reading the
> expected payoff off that superposition, with a quadratic speed-up over Monte Carlo and none of the
> expensive state-loading that bottlenecks other quantum pricers.**

The cognition result (a quantum "investor" beating a classical model on a real behavioral effect) is
the **motivation** that justifies the quantum lens; the **option pricer** is the technical payoff.

## 2. Why it resolves the two quotes (the "whoa")
- **Newton (1720, after the South Sea Bubble):** *"I can calculate the motion of the heavenly
  bodies, but not the madness of people."* → classical/deterministic mechanics fails on markets.
- **Feynman:** *"If you think you understand quantum mechanics, you don't understand quantum
  mechanics."* → classical intuition is the wrong tool; you need a non-classical probability.
- **Resolution:** Newton failed because he used the wrong physics. Human judgment is
  **non-commutative** (asking A-then-B ≠ B-then-A), **context-dependent**, and **interferes** —
  the defining signatures of *quantum probability*. The "madness" isn't noise; it's structured
  non-classicality. Feynman tells us why classical was doomed; quantum cognition tells us what to
  use instead. **The madness of people is quantum.**

## 3. The science is real (this is not a metaphor)
**Motivation layer — quantum cognition / quantum decision theory** (peer-reviewed):
- **Wang, Solloway, Shiffrin & Busemeyer, PNAS 111:9431 (2014)** — the *parameter-free*
  **QQ-equality** for question-order effects, confirmed across **70 national surveys**.
- **Pothos & Busemeyer, Proc. R. Soc. B 276:2171 (2009)** — quantum explains sure-thing-principle
  violations; an equivalent classical model **provably cannot**.
- **Yukalov & Sornette, Theory and Decision 70:283 (2011)** — quantum decision theory for risky
  financial *prospects* (Sornette = crash-modeling econophysicist → our finance bridge).

**Technical layer — quantum option pricing** (see `paper/main_V2.tex` for the full derivation):
- **Stamatopoulos et al., Quantum 4:291 (2020)** — the reference oracle-QAE pricing scheme we
  improve on (we remove its dominant cost, the distribution-loading oracle).
- **Montanaro, Proc. R. Soc. A 471:20150301 (2015)** — quantum speed-up of Monte Carlo (the
  1/ε² → 1/ε that powers QAE).
- **Cox–Ross–Rubinstein, J. Financial Econ. 7:229 (1979)** — the binomial tree; its **product**
  path measure is what lets us load all 2^M paths exactly with M single-qubit rotations.
- **Gilyén et al. (2019) / Martyn et al. (2021)** — QSVT, the spectral bridge for the single-run
  QNDM-powered route.

## 4. ⚠️ THE GUARDRAIL — every team member must say this, proactively
**We do NOT claim the brain is a quantum computer.** That is the fringe "quantum consciousness /
Orch-OR" zone — never go there. We claim: *human judgment violates classical probability exactly
as quantum probability predicts and quantitatively fits.* **Quantum-*like* math, not quantum
neurons.** Saying this first is a credibility win with the physics-literate judges.

## 5. What we are building
**Technical core — a quantum option pricer.** Pipeline (full math in `paper/main_V2.tex`):
1. **Step 0 (classical):** binomial tree (Cox–Ross–Rubinstein) → risk-neutral up-probabilities
   `q_i` → rotation angles `θ_i = 2·arcsin√q_i`.
2. **Load paths:** one `R_y(θ_i)` per time step → the exact superposition `Σ_x √p(x)|x⟩` of all
   `2^M` price paths. **Free and exact** (product distribution, no loading oracle) — this is the win.
3. **QNDM phase encoding:** a detector ancilla in `|+⟩`; kick the payoff quantity `f(x)` into its
   phase `e^{iλ(f−K)}` (never collapses the paths).
4. **Read the price:** (a) **Fourier route** — measure the ancilla to get the characteristic
   function `G(λ)`, Gil–Peláez invert; or (b) **QAE route** — payoff → amplitude, amplitude
   estimation, `O(1/ε)`; or (c) **QSVT route** — keep the phase oracle and get a single-run estimate.
5. **Discount:** `C = e^{−rT}·E[max(f−K,0)]`.

`f = S_T` gives a **European** option; `f = S̄` (path average) gives a path-dependent **Asian**
option — the only difference is one scalar. Time-dependent `σ(t), r(t)` are handled natively.

**Motivation demo (already built):** the cognition "investor" in `/quantum_investor` — a quantum
model that fits a question-order effect a classical model can't. We use it as the *opener* that
justifies the quantum lens, not as the main contribution.

## 6. Current status (as of this writing)
- ✅ **Pricing math fully derived** in [`../paper/main_V2.tex`](../paper/main_V2.tex): all three routes
  (Fourier / QAE / QSVT), a complexity table, and an honest-limitations section.
- ✅ **Motivation demo BUILT and VERIFIED** in [`/quantum_investor`](../quantum_investor): quantum
  fits **~131×** better than classical, classical order effect = 0, QQ-equality = 0, self-checks pass.
  *(Data there is illustrative/synthetic — label it as such.)*
- ⏳ **The pricer code is the build target** (not yet implemented). See §8.

## 7. How to run the code
**Motivation demo** (cognition investor):
```bash
cd quantum_investor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # autoray pinned <0.7 for pennylane 0.38
python main.py        # prints the story + numbers, saves figure.png
python test_model.py  # self-checks
```
**Pricer:** to be built (target dir e.g. `quantum_pricer/`), same PennyLane/Qiskit stack,
statevector simulator. Benchmark against a closed-form Black–Scholes + Monte-Carlo reference.

## 8. What's next — task board (claim a task, push to main)
**MUST-HAVE (protect these):**
- [ ] **Pricer core** — implement Steps 0–1 (tree → angles → `R_y` path loading) and the QNDM +
      Fourier route for a **European call at small M (M≈3)**; recover a known price. — _owner: ___
- [ ] **Classical baseline** — Black–Scholes closed form + Monte-Carlo pricer; the
      **error-vs-queries** plot showing quantum `~1/ε` vs MC `~1/ε²`. — _owner: ___
- [ ] **Deck (5 beats):** lament → market-is-a-superposition → read price without collapse →
      quadratic speed-up → Asian/time-dependent payoff. — _owner: ___
- [ ] **Notebook runs top-to-bottom** for the live demo; record a 2-min fallback video. — _owner: ___

**NICE-TO-HAVE (only if must-haves are safe; freeze new features by Hr 15 of the plan):**
- [ ] **Asian option** — swap `f = S_T → S̄`; price a path-dependent option from the same circuit. — _owner: ___
- [ ] **Single-run QAE / QSVT route** — payoff into an amplitude; one estimation, no λ sweep. — _owner: ___
- [ ] **Time-dependent σ(t), r(t)** + a qubit-count / state-prep comparison vs. oracle-QAE. — _owner: ___

See [`quantum-investor.html`](./quantum-investor.html) for the full hour-by-hour 24h plan and
team swimlanes.

## 9. How this scores (keep all four alive)
- **Novelty (25%):** Newton + Feynman framing → a QNDM phase-encoded pricer with **free, exact**
  path loading — distinct from the textbook oracle-QAE everyone else will show.
- **Problem formulation (25%):** "price = discounted `E[max(f−K,0)]` over a quantum path
  superposition" — crisp, with a falsifiable quadratic speed-up claim.
- **Technical depth (25%):** three routes (Fourier / QAE / QSVT), a full complexity table, honest
  limits — grounded in Stamatopoulos 2020 & Montanaro 2015.
- **Business case (25%):** derivative pricing is OP's daily bread; **Asian + time-dependent**
  options are the genuinely hard, high-value cases where the quantum advantage bites.

## 10. Conventions for agents working here
- This file is the source of truth; if the plan changes, **update this file** and the task board.
- Keep the **guardrail (§4)** intact in any external-facing material.
- Motivation code lives in `/quantum_investor`; keep it runnable (`python main.py` +
  `python test_model.py` must pass) before pushing. The pricer code (build target) should ship with
  its own self-checks against a Black–Scholes / Monte-Carlo reference. Don't commit `.venv/` or `__pycache__/`.
- Small, focused commits to `main`; end commit messages with the Co-Authored-By line if an agent.
- Honesty: results on synthetic data must be labelled as such; the win is REAL data.
