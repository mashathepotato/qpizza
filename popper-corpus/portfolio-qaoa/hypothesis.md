---
slug: portfolio-qaoa
created: 2026-06-06
status: active
falsifiability_gate: passed
literature_pass: completed
---

# QAOA yields a meaningful, quantum-native advantage over classical heuristics for cardinality-constrained portfolio selection

## Original framing
> Quantum computers can optimise portfolios better than classical computers.

## Operational restatement
QAOA (depth p=1 to p=3) applied to cardinality-constrained portfolio selection
(choose exactly k assets from N to maximise return-minus-risk objective) achieves
approximation ratio `approx_ratio` (solution value / brute-force optimum) >= that
of classical exact or greedy heuristic baselines at the same problem size.
Operationally: N=10 assets, k=3, expected-return and covariance drawn from a
synthetic Markowitz model; brute-force optimum computed exactly; `approx_ratio`
= QAOA objective / optimal objective. Claim: QAOA `approx_ratio` >= 0.95 and
>= classical greedy ratio, for at least one depth p <= 3. Problem is Q50-runnable
(shallow, <= 10 qubits, low depth). NOTE: this is the *combinatorial*
cardinality-constrained version; continuous mean-variance optimisation is a
convex QP solvable classically in poly time and is NOT the target problem.

## Falsifier(s)
- Classical exact solver (brute force, feasible at N <= 20) or greedy heuristic
  achieves `approx_ratio` >= QAOA `approx_ratio` at simulable scale (N <= 15),
  with no asymptotic argument for a QAOA crossover at larger N.
- QAOA variational landscape exhibits barren plateaus or local minima at p >= 2
  that prevent convergence to `approx_ratio` > 0.9 on the test instance.

## Test design
- Methods: `approx_ratio` = QAOA_objective / brute_force_optimum, measured by
  triage harness (`triage/harnesses/portfolio_opt.py`); QAOA optimised via
  COBYLA (classical outer loop).
- Design: N=10 assets, k=3; QAOA circuit on `aer_simulator` (statevector);
  brute force exact baseline; greedy (pick top-k by Sharpe ratio) baseline.
- Comparison: `approx_ratio` for QAOA p=1,2,3 vs exact and greedy.
- Q50-runnable: <= 10 qubits, shallow circuits, suitable for on-site hardware demo.

## Auxiliary assumptions
- The penalty-term encoding of the cardinality constraint (exactly k assets)
  does not dominate the objective landscape and allows COBYLA to find good angles.
- N=10 is representative enough of the combinatorial structure that results
  generalise directionally to larger N.
- Brute-force optimum is achievable in reasonable time (C(10,3)=120 combinations).

## Distinctiveness
QAOA is quantum-native for binary combinatorial problems: the binary
include/exclude decision for each asset maps directly to qubit states, and
the quantum interference across superpositions explores the 2^N solution space
simultaneously. This is categorically different from classical convex portfolio
optimisation. Forbids: QAOA matching classical performance only on problem sizes
where brute-force is also trivial (N <= 5); the claim requires N >= 10.

## References
- path: refs/farhi-2014-qaoa.md
  contribution: introduces QAOA and establishes that binary combinatorial
                problems (not convex QP) are the natural target; motivates
                choice of cardinality-constrained portfolio (binary) over
                continuous mean-variance (convex, classical poly-time)

## Intake log
2026-06-06 — Hypothesis instantiated from triage-lab quantum-finance template.
Critical distinction drawn: continuous mean-variance portfolio optimisation is
convex QP (classical poly-time, wrong target); the cardinality constraint makes
the problem genuinely combinatorial (NP-hard in general). Q50-runnable framing
noted — shallow circuits, <= 10 qubits.
