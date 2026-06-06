# "The Madness of People Is Quantum" — resolving Newton & Feynman scientifically

> The single best narrative hook we have. Two famous quotes that sound unrelated turn out to be
> the *same statement*, and the resolution is a real, peer-reviewed science (quantum cognition /
> quantum decision theory) that is quantum-native, buildable on a few qubits, and ties straight
> to behavioral finance. This is our hero candidate.

## The two quotes
- **Newton (1720, after losing a fortune in the South Sea Bubble):**
  *"I can calculate the motion of the heavenly bodies, but not the madness of people."*
  → Classical, deterministic mechanics predicts planets but **fails on human markets**. The
  "madness" = irrational, herding, unpredictable crowd behaviour.
- **Feynman:**
  *"If you think you understand quantum mechanics, you don't understand quantum mechanics."*
  → Quantum mechanics is **irreducibly non-classical**; classical intuition is the wrong tool.
  The deeper lesson: some systems simply do not obey classical (commutative, deterministic)
  logic — you need a different probability structure.

## The resolution (the "whoa" moment)
> **Newton couldn't predict the madness of people for the *exact same reason* Feynman says you
> can't grasp quantum mechanics classically: human decision-making does not obey classical
> probability — it obeys *quantum* probability.**

Newton's mistake was applying *heavenly* (classical, deterministic) mechanics to a system that
is **quantum-like**. The "madness" isn't random noise to be averaged away — it's *structured
non-classicality*. Human judgment is:
- **non-commutative** — ask A-then-B ≠ B-then-A (order effects),
- **context-dependent** — the question constructs the answer (measurement collapses the state),
- **interfering** — choices under uncertainty show interference, not a weighted average.

Those are the *defining signatures of quantum probability*. So the two quotes collapse into one:
**markets are classically unpredictable because the agents in them run on quantum probability.**
Feynman tells us *why classical intuition was doomed*; quantum cognition tells us *what to use
instead*. Newton lamented the problem; quantum mechanics is the answer he was missing.

## Why this is SCIENCE, not poetry (verified literature)
This is a real field — "quantum cognition" — and it makes **parameter-free predictions that
have been confirmed on real data**:
- **Pothos & Busemeyer 2009, Proc. R. Soc. B 276:2171** — people violate the "sure-thing
  principle" (disjunction effect); a quantum model explains it and an equivalent classical
  Markov model **provably cannot**.
- **Wang, Solloway, Shiffrin & Busemeyer 2014, PNAS 111:9431** — the **QQ-equality**, a
  *parameter-free* quantum prediction for question-order effects, **confirmed across 70 national
  surveys**. (A metaphor can't make parameter-free predictions; this did.)
- **Pothos & Busemeyer 2022, Annual Review of Psychology 73:749** — the field's authoritative
  review (Annual Reviews only commissions established programs).
- **Yukalov & Sornette, "Quantum Decision Theory"** (Adv. Complex Syst. 2010; *Theory and
  Decision* 2011) — applies quantum probability directly to **risky prospects/lotteries**, the
  objects of prospect theory. *Sornette is a famous econophysicist (crash models)* → our direct
  bridge to finance.
- **Widdows, Rani & Pothos 2023, Entropy 25:548** — implements these cognitive decision models
  as **actual qubit circuits** (states in qubits, decisions as gates + measurement). This is our
  buildability proof: order effects = non-commuting single-qubit rotations; interference =
  Hadamard + relative phase.

## ⚠️ The one line that keeps us out of pseudoscience
**We do NOT claim the brain is a quantum computer.** That's the fringe "quantum consciousness /
Orch-OR" stuff — avoid it entirely. We claim: *human judgment violates classical probability in
exactly the ways quantum probability (a non-commutative generalization of probability) predicts
and quantitatively fits.* Quantum-**like** math, not quantum neurons. Say this proactively —
it's the question a sharp judge will ask, and answering it first is a credibility win.

Pitch-ready line:
> *"We don't claim the brain is quantum. We claim human judgment breaks classical probability —
> order effects, interference, context — precisely as quantum probability predicts (PNAS 2014,
> Annual Review of Psych 2022). The same Hilbert-space math runs natively on a qubit. So Newton's
> 'madness of people' and Feynman's 'you don't understand quantum mechanics' are the same
> sentence: the madness IS quantum."*

---

## The hackathon build — "Quantum Investor: pricing the madness of crowds"

**Concept:** a quantum-circuit model of investor decision-making that **reproduces a
behavioral-finance anomaly a classical model cannot**, then show what that means for markets.

### Demo arc (the story on stage)
1. **The lament:** show Newton's South Sea Bubble loss + quote. "The father of classical physics
   was wrecked by the madness of crowds."
2. **The classical model fails:** a standard rational/Bayesian model of an investor choice;
   feed it a real behavioral anomaly (e.g. framing/order effect in a risk decision) — it
   **can't fit** without contradiction (violates the law of total probability).
3. **The quantum model fits:** the *same* decision as a qubit circuit — superposition of
   intentions, interference term, measurement = the choice. It reproduces the anomaly **with the
   parameter-free QQ-equality holding**. Feynman's quote drops here.
4. **Resolution slide:** the two quotes side by side → one equation. "The madness of people is
   quantum."
5. **Finance payoff:** scale to a crowd — interfering investor states → herding / panic /
   bubble dynamics; show a toy "quantum sentiment" signal vs a classical one around a known
   bubble. Business case for OP below.

### What to build (1–3 qubits, very feasible)
- **Core:** reproduce a **question-order / disjunction effect** as a quantum circuit (per the
  Entropy 2023 paper): belief state on 1–2 qubits, two "questions" as non-commuting rotations,
  measure → show order-dependence + the QQ-equality. Classical baseline = a Markov/Bayesian
  model that demonstrably can't match it.
- **Finance framing:** make the two "questions" an **investment decision under framing/uncertainty**
  (e.g. "invest given good news first vs. bad news first," or a Tversky–Shafir-style two-stage
  gamble). Show the interference term predicting the behavioral gap.
- **Stretch:** an **interference-driven sentiment / herding** toy — N investor qubits with a
  coupling; sweep it and watch a "bubble" form from constructive interference of intentions
  (Yukalov–Sornette "interference of intentions"). Overlay a quantum vs classical sentiment
  index on a historical bubble window.
- **Stack:** PennyLane or Qiskit; tiny circuits, statevector sim; real survey/framing data or a
  classic behavioral-finance dataset for the anomaly.

### Why it scores on all four axes
- **Novelty/creativity (25%):** off the charts — nobody else will fuse Newton, Feynman, and
  quantum cognition into a finance demo. Maximally quantum-native (the *probability itself* is
  quantum).
- **Problem formulation (25%):** crisp — "classical probability can't model investor
  irrationality; quantum probability can," with a parameter-free testable prediction.
- **Technical depth (25%):** real circuits + a falsifiable classical-vs-quantum comparison +
  the QQ-equality; grounded in PNAS/RSPB papers.
- **Business case (25%):** OP cares about **customer/investor behaviour** — herding, bank-run
  panic, framing in product/pension choice, mis-pricing of risk appetite. A quantum behavioural
  model = better prediction of irrational market moves and better product/communication design.

### Honest guardrails
- Quantum-**like** math, not quantum neurons (say it first).
- The *finance-specific* quantum-behaviour literature is thin (Yukalov–Sornette is the bridge) —
  frame as "behavioral anomalies are decision anomalies, and that's quantum probability's
  strongest turf," not "mature quantum behavioural asset pricing exists."
- Lead with **order effects + disjunction effect** (strongest evidence); treat framing as color.
- The herding/bubble part is a *creative toy* — present it as illustrative, keep the core anomaly
  demo rigorous.

## Fusion options with our other ideas
- **× "Entanglement of Markets":** investor *entanglement* → herding; entanglement entropy of the
  crowd state spikes into a bubble. Behavioural micro-story + emergent macro crash signal.
- **× "Noise Is the Market":** decoherence = the crowd "making up its mind" (collapse of
  superposed intentions) under information pressure.

## Verdict
**Strongest narrative + genuinely quantum-native + buildable on 1–3 qubits + real papers behind
it.** This is the idea most likely to be *repeated by a judge to the next team* — which is how
you win the room. Recommend as **hero**, with "Entanglement of Markets" as the macro-scale
sibling if we want a two-act demo (micro: quantum investor → macro: quantum crash).

## Key references (verified)
- Pothos & Busemeyer 2009, Proc. R. Soc. B 276:2171 — https://royalsocietypublishing.org/rspb/article/276/1665/2171
- Wang et al. 2014, PNAS 111:9431 (QQ-equality, 70 surveys) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4084470/
- Pothos & Busemeyer 2022, Annu. Rev. Psychol. 73:749 — https://www.annualreviews.org/content/journals/10.1146/annurev-psych-033020-123501
- Yukalov & Sornette 2011, Theory and Decision 70:283 (prospects) — https://arxiv.org/abs/1102.2738
- Busemeyer & Bruza 2012, *Quantum Models of Cognition and Decision*, CUP.
- Widdows, Rani & Pothos 2023, Entropy 25:548 (qubit circuits) — https://arxiv.org/abs/2302.03012
- Khrennikov et al. 2025, Phil. Trans. R. Soc. A ("quantum-like", not neurons) — https://arxiv.org/abs/2503.05859
- Avoid: "Quantum mind"/Orch-OR — https://en.wikipedia.org/wiki/Quantum_mind
