# A Quantum Approximate Optimization Algorithm

**Authors:** Farhi, E.; Goldstone, J.; Gutmann, S.

## Abstract
We introduce the Quantum Approximate Optimization Algorithm (QAOA), a
variational quantum algorithm for combinatorial optimisation. At depth p,
QAOA applies p alternating layers of problem Hamiltonian and mixing Hamiltonian
evolution, producing a state whose energy expectation approximates the optimum.
For MaxCut on 3-regular graphs, QAOA at p=1 achieves an approximation ratio
of at least 0.6924. QAOA is natively suited to binary-variable combinatorial
problems where the solution space is exponential and classical exact solvers
require exponential time. For cardinality-constrained portfolio selection
(choose exactly k assets from N to maximise expected return minus risk), the
binary nature of asset inclusion/exclusion makes QAOA a natural candidate,
unlike continuous mean-variance optimisation which is a convex QP solvable
classically in polynomial time.
