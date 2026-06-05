# Quantum risk analysis

**Authors:** Woerner, S.; Egger, D. J.

## Abstract
We show how quantum amplitude estimation can be applied to risk analysis in
finance, specifically to compute Value at Risk (VaR) and Conditional Value at
Risk (CVaR) of loss distributions. The algorithm achieves O(1/eps) query
complexity for tail-risk estimation, a quadratic speedup over classical Monte
Carlo which requires O(1/eps^2) samples. The core technique encodes the
cumulative distribution function into a quantum state and uses amplitude
estimation to read out the tail probability. Key bottleneck: the depth of
the circuit scales with the discretisation resolution of the loss distribution.
Capital and insurance reserving applications are the primary business motivation,
as even small improvements in tail-risk accuracy can translate to significant
capital relief under Basel III/Solvency II frameworks.
