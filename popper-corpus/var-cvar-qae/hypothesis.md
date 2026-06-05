---
slug: var-cvar-qae
created: 2026-06-06
status: active
falsifiability_gate: passed
literature_pass: completed
---

# Quantum Amplitude Estimation yields a meaningful, quantum-native advantage over classical Monte Carlo for tail-risk (VaR/CVaR) estimation

## Original framing
> Quantum computers can compute financial tail risk faster than classical computers.

## Operational restatement
Iterative Amplitude Estimation (IAE) applied to Value at Risk (VaR) and
Conditional Value at Risk (CVaR) estimation for a loss distribution achieves
lower `samples_to_eps` than classical Monte Carlo, inheriting the O(1/eps)
vs O(1/eps^2) oracle-call complexity. Operationally: for a discrete loss
distribution over N assets, IAE encodes the cumulative distribution function
into a quantum state and reads out tail probabilities via amplitude estimation.
Advantage is claimed if IAE `samples_to_eps` < MC `samples_to_eps` at eps <= 1e-2.
Primary business context: capital/insurance reserving under Basel III / Solvency II
where tail-risk accuracy drives regulatory capital requirements.

## Falsifier(s)
- A classical Monte Carlo baseline (or classical importance-sampling variant)
  matches or beats IAE on `samples_to_eps` at any simulable scale, with no
  asymptotic backing for a crossover at larger problem size.
- The circuit depth required to discretise the loss distribution with sufficient
  resolution grows to exceed the depth budget before the oracle-call advantage
  materialises at practical eps targets.

## Test design
- Methods: `samples_to_eps` recorded by the triage harness
  (`triage/harnesses/var_cvar.py`) on a synthetic multivariate normal loss
  distribution; eps swept from 1e-1 to 1e-3.
- Design: IAE run on `aer_simulator` (statevector, noiseless); MC baseline uses
  numpy with 95th-percentile (VaR) and expected shortfall (CVaR) estimators.
- Comparison: oracle-call count for IAE vs sample count for MC at matched eps
  targets; log-log slope test for quadratic vs linear scaling.
- Business tie: report capital impact proxy (basis points of reserve change
  per unit eps improvement) alongside raw `samples_to_eps`.

## Auxiliary assumptions
- The loss distribution can be discretised at sufficient resolution without
  circuit depth overwhelming the oracle-call advantage.
- VaR/CVaR estimation reduces cleanly to amplitude estimation on the tail
  probability (i.e., the threshold-encoding circuit is tractable).
- Classical importance sampling is not included as a baseline (making it
  a partial competitor); the comparison is strictly against naive MC.

## Distinctiveness
Same O(1/eps) vs O(1/eps^2) asymptotic as option pricing QAE, but applied to
tail risk rather than expectation: the tail indicator function introduces a
hard threshold in the payoff, making the quantum encoding different from the
smooth payoff case. Forbids: equal `samples_to_eps` log-log slope between
IAE and naive MC on tail-risk targets.

## References
- path: refs/woerner-egger-2019-quantum-risk-analysis.md
  contribution: first demonstration of QAE applied to VaR/CVaR; establishes
                O(1/eps) query complexity for tail-risk estimation and identifies
                discretisation depth as the key bottleneck included in falsifier 2;
                capital/insurance reserving as primary business motivation

## Intake log
2026-06-06 — Hypothesis instantiated from triage-lab quantum-finance template.
Shares QAE machinery with option-pricing-qae but targets tail risk specifically.
Falsifier 2 distinguishes this from the option-pricing case by focusing on
the discretisation depth for tail-threshold encoding. Business motivation tied
to regulatory capital frameworks per Woerner & Egger 2019.
