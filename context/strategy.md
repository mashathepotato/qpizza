# Strategy — how we win

## Scoring model (internal)
All four criteria are weighted 25%. Treat them as a checklist we must each *defend* in the
demo. A judge should be able to point to a slide/cell for each.

| Criterion | What proves it | Our hook |
|---|---|---|
| Novelty (25%) | The quantum part is *essential*, not bolted on | TBD — see ideas.md |
| Problem formulation (25%) | Crisp problem, real data, well-posed objective | Pick a problem a quant actually cares about |
| Technical depth (25%) | Working sim, sound circuits, honest benchmarks | Real Qiskit/PennyLane run + classical baseline |
| Business case (25%) | OP could use this; € impact is articulated | Tie to OP products (wealth mgmt, insurance, banking) |

## The originality trap (most important)
Judges explicitly penalize "classical methods with a quantum label attached." Litmus test
for any idea: **"If I delete the quantum part, does the problem still make sense as stated?"**
- If the classical version is the *natural* formulation and quantum is a swap-in → weak.
- If the formulation is *born quantum* (the state space, entanglement, or amplitude encoding
  is what makes it tractable/interesting) → strong.

## How to use a mixed team (2 CS + 2 physics)
- **Physics**: own the quantum formulation, circuit design, why-it's-quantum-native argument,
  and the theoretical advantage story. They are our credibility with OP's researchers.
- **CS**: own the implementation, data pipeline, classical baseline, benchmarking harness,
  visualization, and the demo/repo polish.
- **Pairing**: 1 physics + 1 CS on the core algorithm; 1 physics + 1 CS on
  business-case + demo + baseline. Re-sync every few hours.

## Demo must-haves
1. Clear problem statement on one slide (with the € / risk stakes).
2. A working quantum simulation we can run live (or recorded fallback).
3. A **classical baseline** to compare against — shows we understand the advantage claim and
   are honest about NISQ limits.
4. An honest "where this goes on real hardware / at scale" slide.
5. A business-case slide mapping to an OP product line.

## Honesty wins with this audience
OP brings real quantum researchers. Overclaiming "quantum speedup" on a toy problem will be
seen through. Frame advantage carefully: asymptotic potential vs. NISQ-era reality. Being
rigorous about limitations is itself a differentiator.

## De-risking
- Scope to something we can *finish*. A polished narrow PoC beats an unfinished grand one.
- Have a simulator fallback; don't depend on live hardware access.
- Decide the idea fast (see ideas.md) — formulation time is more valuable than coding time.
