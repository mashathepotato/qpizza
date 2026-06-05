---
slug: option-pricing-qae
created: 2026-06-06
status: active
falsifiability_gate: passed
literature_pass: completed
---

# Quantum Amplitude Estimation yields a meaningful, quantum-native advantage over classical Monte Carlo for European option pricing

## Original framing
> Quantum computers can price options faster than classical computers.

## Operational restatement
Iterative Amplitude Estimation (IAE) applied to European option expected payoff
estimation achieves a lower number of oracle/sample calls to reach accuracy eps
than classical Monte Carlo, with the quantum-native complexity O(1/eps) versus
MC O(1/eps^2). Operationally: at eps = 1e-3, IAE requires ~1,000 oracle calls
while MC requires ~1,000,000 samples. We measure this as `samples_to_eps`
(the oracle/sample call count to first crossing a fixed eps threshold) on a
simulated lognormal payoff distribution. Advantage is claimed if the ratio
IAE_samples / MC_samples < 1 at eps <= 1e-2.

## Falsifier(s)
- A classical Monte Carlo baseline matches or beats IAE on `samples_to_eps` at
  any simulable scale (N <= 30 qubits, shallow circuits), with no asymptotic
  justification for a crossover at deeper/wider circuits.
- The distribution loading cost (state-preparation circuit depth) erases the
  oracle-call advantage: total gate count for IAE exceeds MC wall-clock at
  classically-simulable precision targets.

## Test design
- Methods: `samples_to_eps` is recorded by the triage harness
  (`triage/harnesses/option_pricing.py`) on a lognormal payoff, varying eps
  from 1e-1 down to 1e-3.
- Design: IAE circuit run on `aer_simulator` (statevector, noiseless) to isolate
  oracle-call complexity from hardware noise; MC run with numpy RNG at the same
  eps targets.
- Comparison: log-log slope of samples vs 1/eps — quantum slope ~1, MC slope ~2.
- LUMI-sim story: deep IAE circuits (> 50 qubits) require HPC simulation;
  overnight runs use Q50-equivalent statevector sim on LUMI.

## Auxiliary assumptions
- The lognormal distribution can be loaded into the quantum circuit with
  overhead that does not dominate the oracle-call count at the tested eps range.
- `aer_simulator` faithfully represents noiseless gate complexity (oracle call
  count is hardware-agnostic).
- Asymptotic O(1/eps) vs O(1/eps^2) scaling extends to the simulable regime
  at eps ~ 1e-2 to 1e-3.

## Distinctiveness
Predicts a *scaling advantage* (slope change on log-log) that classical variance
reduction (antithetic variates, control variates) cannot fully close: those
improve constants, not the quadratic-to-linear exponent. Forbids: equal log-log
slope between IAE and naive MC on `samples_to_eps`.

## References
- path: refs/brassard-2002-quantum-amplitude-estimation.md
  contribution: establishes the O(1/eps) vs O(1/eps^2) oracle complexity result
                that is the foundation of the quantum-native advantage claim;
                also identifies state-preparation cost as the key caveat included
                in the falsifier list

## Intake log
2026-06-06 — Hypothesis instantiated from triage-lab quantum-finance template.
Operational restatement sharpened to use `samples_to_eps` metric emitted by
the triage harness. Falsifier 2 added to capture the distribution-loading cost
caveat raised during literature review of Brassard 2002. LUMI-sim story noted
for deep-circuit regime.
