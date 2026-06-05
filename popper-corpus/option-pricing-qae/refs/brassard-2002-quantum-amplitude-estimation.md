# Quantum amplitude estimation and quantum Monte Carlo

**Authors:** Brassard, G.; Hoyer, P.; Mosca, M.; Tapp, A.

## Abstract
The canonical quantum amplitude estimation algorithm achieves a quadratic
speedup over classical Monte Carlo sampling: O(1/eps) oracle calls versus
O(1/eps^2) for classical Monte Carlo to estimate an expectation to accuracy eps.
Applied to financial derivatives pricing, this yields a provable asymptotic
advantage for European option payoff estimation when the circuit depth is
manageable. The key bottleneck is state preparation for the underlying
distribution, which may require additional gates proportional to the number of
discretisation points.
