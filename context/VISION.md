# VISION — "The Madness of People Is Quantum" (LOCKED DIRECTION)

> **This is the chosen project.** Single source of truth for the team and for any AI agent
> working in this repo. Self-contained: read this and you can contribute without reading
> anything else first. Deeper detail in [`newton-feynman.md`](./newton-feynman.md); visual
> explainer in [`quantum-investor.html`](./quantum-investor.html); working code in
> [`/quantum_investor`](../quantum_investor).

**Hackathon:** Junction Helsinki × OP Pohjola — Quantum Computing for Finance.
**Team:** 4 people (2 CS + 2 physics). **Judged 25% each:** novelty · problem formulation ·
technical depth · business case. **Submission:** tomorrow 10:00 AM.

---

## 1. The one-sentence pitch
> **Newton said he could not predict "the madness of people"; Feynman said you cannot understand
> the world with classical intuition. Both are the same statement: human financial
> decision-making does not obey classical probability — it obeys *quantum* probability. We model
> investor "irrationality" as a quantum circuit and reproduce real behavioral effects that
> classical models provably cannot.**

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
Field = **quantum cognition / quantum decision theory** (peer-reviewed). Key anchors:
- **Wang, Solloway, Shiffrin & Busemeyer, PNAS 111:9431 (2014)** — the *parameter-free*
  **QQ-equality** for question-order effects, confirmed across **70 national surveys**.
- **Pothos & Busemeyer, Proc. R. Soc. B 276:2171 (2009)** — quantum explains sure-thing-principle
  violations; an equivalent classical model **provably cannot**.
- **Pothos & Busemeyer, Annual Review of Psychology 73:749 (2022)** — authoritative review.
- **Yukalov & Sornette, Theory and Decision 70:283 (2011)** — quantum decision theory for risky
  financial *prospects* (Sornette = crash-modeling econophysicist → our finance bridge).
- **Widdows, Rani & Pothos, Entropy 25:548 (2023)** — these models built as **qubit circuits**
  (our buildability proof).

## 4. ⚠️ THE GUARDRAIL — every team member must say this, proactively
**We do NOT claim the brain is a quantum computer.** That is the fringe "quantum consciousness /
Orch-OR" zone — never go there. We claim: *human judgment violates classical probability exactly
as quantum probability predicts and quantitatively fits.* **Quantum-*like* math, not quantum
neurons.** Saying this first is a credibility win with the physics-literate judges.

## 5. What we are building
A quantum-circuit "investor" that reproduces a **question-order effect** in an investment
decision — something a classical (Bayesian) model **cannot** represent — and that satisfies the
parameter-free QQ-equality. Then (stretch) scale to a crowd where **interference of intentions**
produces herding/bubbles.

**Finance framing (the two questions):**
- A = "Do you trust the market right now?" (yes/no)
- B = "Will you invest your savings today?" (yes/no)
Asking order changes the yes-rates — the order effect.

## 6. Current status (as of this writing)
- ✅ **Core demo BUILT and VERIFIED** in [`/quantum_investor`](../quantum_investor). It runs:
  - quantum model fits the data **~131× better** than classical,
  - classical order effect = **0** (structurally impossible),
  - **QQ-equality = 0** (parameter-free), all self-checks pass, `figure.png` renders.
- ⚠️ **Data is currently illustrative** (synthetic, in `data.py`) so the pipeline runs out of the
  box. **Highest-value next task: replace with REAL data** (see §8).

## 7. How to run the code
```bash
cd quantum_investor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # autoray pinned <0.7 for pennylane 0.38
python main.py        # prints the story + numbers, saves figure.png
python test_model.py  # self-checks (QQ-equality, order effect, normalisation)
```
Files: `quantum_model.py` (the science), `classical_model.py` (baseline that fails),
`data.py` (swap in real data here), `fit.py`, `plot.py`, `main.py`, `test_model.py`.

## 8. What's next — task board (claim a task, push to main)
**MUST-HAVE (protect these):**
- [ ] **Real data** — run the two framed yes/no questions on ~30+ attendees, randomise which is
      asked first, paste counts into `data.py`. (Or use a published question-order dataset.)
      *This is the knockout: QQ-equality holding on real human data.* — _owner: ___
- [ ] **Deck (5 beats):** lament → classical fails → quantum fits → resolution → finance payoff,
      plus the guardrail slide. — _owner: ___
- [ ] **Notebook that runs top-to-bottom** for the live demo; record a 2-min fallback video. — _owner: ___

**NICE-TO-HAVE (only if must-haves are safe; freeze new features by Hr 15 of the plan):**
- [ ] **Crowd/bubble module** — N coupled investor qubits; sweep coupling → constructive
      interference of intentions forms a "bubble" (Yukalov–Sornette). — _owner: ___
- [ ] **Quantum vs classical "sentiment"** overlaid on a real historical bubble window. — _owner: ___
- [ ] **Second anomaly** — the disjunction effect (a two-stage gamble). — _owner: ___

See [`quantum-investor.html`](./quantum-investor.html) for the full hour-by-hour 24h plan and
team swimlanes.

## 9. How this scores (keep all four alive)
- **Novelty (25%):** Newton + Feynman + quantum cognition in finance — unique, maximally
  quantum-native (the *probability itself* is quantum).
- **Problem formulation (25%):** "classical probability can't model investor irrationality;
  quantum can" — crisp and falsifiable.
- **Technical depth (25%):** real qubit circuits + a falsifiable classical-vs-quantum comparison
  + the parameter-free QQ-equality, grounded in PNAS/RSPB.
- **Business case (25%):** OP cares about investor/customer behaviour — herding, bank-run panic,
  framing in pension/product choices, mispriced risk appetite. A quantum behavioural model
  predicts irrational moves classical models miss.

## 10. Conventions for agents working here
- This file is the source of truth; if the plan changes, **update this file** and the task board.
- Keep the **guardrail (§4)** intact in any external-facing material.
- Code lives in `/quantum_investor`; keep it runnable (`python main.py` + `python test_model.py`
  must pass) before pushing. Don't commit `.venv/` or `__pycache__/`.
- Small, focused commits to `main`; end commit messages with the Co-Authored-By line if an agent.
- Honesty: results on synthetic data must be labelled as such; the win is REAL data.
