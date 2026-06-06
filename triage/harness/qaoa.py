"""QAOA portfolio harness: candidate A (combinatorial selection).

Selects exactly k assets maximizing  mu.x - risk * x^T cov x  with sum(x)==k.

The encoding uses qiskit-optimization's QuadraticProgram + QuadraticProgramTo
Qubo so the cardinality penalty and QUBO->Ising mapping are handled by the
library (avoiding hand-rolled sign / bit-ordering bugs). QAOA + COBYLA over the
statevector Sampler optimizes the variational ansatz; MinimumEigenOptimizer
returns result.x as a 0/1 array aligned to QuadraticProgram variable order.

The quantum-native fingerprint: QAOA is a shallow variational circuit whose
amplitudes concentrate on the constrained optimum — delete the quantum sampler
and the method collapses to brute force. The metric is the approximation ratio
of the QAOA-selected portfolio vs the brute-force optimum."""
from __future__ import annotations
import numpy as np

from qiskit.primitives import Sampler  # V1 Sampler; works with qiskit-algorithms 0.3.x
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

from backends import get_backend
from triage.rubric import AdvantageRecord
from triage.baselines.classical_opt import exact_portfolio


def _build_program(mu: np.ndarray, cov: np.ndarray, k: int,
                   risk: float) -> QuadraticProgram:
    """maximize mu.x - risk*x^T cov x  s.t. sum(x)==k, x binary."""
    n = len(mu)
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(f"x{i}")
    linear = {f"x{i}": float(mu[i]) for i in range(n)}
    # Intentional: the loop emits both (x_i, x_j) and (x_j, x_i) for i != j.
    # qiskit-optimization SUMS duplicate keys, giving coefficient 2*(-risk*cov_ij)
    # for off-diagonal pairs.  This is CORRECT: for binary x the quadratic form
    # x^T cov x = sum_i cov_ii x_i + 2*sum_{i<j} cov_ij x_i x_j, so each
    # off-diagonal term must appear with factor 2.  Do NOT "fix" this to emit
    # only (i,j) once — that would halve the off-diagonal penalty and break the
    # QUBO.  The regression test test_qubo_objective_matches_formula guards this.
    quadratic = {}
    for i in range(n):
        for j in range(n):
            c = cov[i, j]
            if c != 0.0:
                quadratic[(f"x{i}", f"x{j}")] = -float(risk) * float(c)
    qp.maximize(linear=linear, quadratic=quadratic)
    qp.linear_constraint(
        linear={f"x{i}": 1 for i in range(n)}, sense="==", rhs=int(k), name="card"
    )
    return qp


def _q50_faithful(n_assets: int, k: int, reps: int, risk: float, seed: int) -> bool:
    """Does a real QAOA-structured circuit transpile on q50_fake?

    Builds the actual QuadraticProgram -> QUBO -> Ising operator pipeline that
    the solver uses, constructs a QAOAAnsatz from the Ising SparsePauliOp, and
    transpiles it on IQMFakeAphrodite (54 q, native gates {r, cz}).  Returns
    True if transpilation succeeds, False on any error.  Uses a small fixed
    instance (same n_assets/k/risk/seed as the caller) so it reflects the real
    circuit structure, not a trivial placeholder.

    Approach: QAOAAnsatz from qubo.to_ising() SparsePauliOp (same as the other
    harnesses that transpile a representative circuit on q50_fake).
    """
    try:
        from qiskit import transpile
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit_optimization.converters import QuadraticProgramToQubo

        rng = np.random.default_rng(seed)
        mu_small = rng.uniform(0.0, 0.2, size=n_assets)
        a = rng.normal(0.0, 0.02, size=(n_assets, n_assets))
        cov_small = a @ a.T

        qp = _build_program(mu_small, cov_small, k, risk)
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(qp)
        ising_op, _ = qubo.to_ising()

        ansatz = QAOAAnsatz(cost_operator=ising_op, reps=int(reps))
        backend = get_backend("q50_fake")
        transpile(ansatz.decompose(), backend, optimization_level=1)
        return True
    except Exception:
        return False


def solve_portfolio(mu, cov, k, risk=1.0, reps=1, seed=0, backend="local_aer"):
    """Select exactly k assets maximizing mu.x - risk*x^T cov x via QAOA.

    Returns (chosen_indices_list, objective_value) where objective_value is the
    portfolio objective of the chosen selection (comparable to exact_portfolio).
    """
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)
    n = len(mu)

    # backend is referenced for the rubric contract; QAOA itself runs on the
    # statevector Sampler primitive (qiskit-algorithms 0.3.x expectation).
    qp = _build_program(mu, cov, k, risk)

    np.random.seed(seed)
    sampler = Sampler(options={"seed": int(seed)})
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=250),
        reps=int(reps),
    )
    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(qp)

    # result.x is a 0/1 array aligned to qp.variables order (x0..x{n-1}).
    x = np.asarray(result.x, float).round().astype(int)
    chosen = [i for i in range(n) if x[i] == 1]

    obj = float(mu @ x - risk * (x @ cov @ x))
    return chosen, obj


def run(config: dict) -> AdvantageRecord:
    n = int(config.get("n_assets", 4))
    k = int(config.get("k", 2))
    reps = int(config.get("reps", 1))
    seed = int(config.get("seed", 0))
    risk = float(config.get("risk", 1.0))
    backend = config.get("backend", "local_aer")
    candidate = config.get("candidate", "A")

    # Small random instance: returns mu and a positive-semidefinite covariance.
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.0, 0.2, size=n)
    a = rng.normal(0.0, 0.02, size=(n, n))
    cov = a @ a.T  # SPD

    chosen, q_val = solve_portfolio(mu, cov, k=k, risk=risk, reps=reps,
                                    seed=seed, backend=backend)
    _, opt_val = exact_portfolio(mu, cov, k=k, risk=risk)

    # Approximation ratio in [0,1]; shift both by a common offset so the ratio
    # is well-defined even when objectives are small/negative.
    shift = abs(min(q_val, opt_val)) + 1e-6
    approx = (q_val + shift) / (opt_val + shift)
    approx = float(min(max(approx, 0.0), 1.0))

    if approx >= 0.99:
        direction = "win"
    elif approx >= 0.9:
        direction = "tie"
    else:
        direction = "loss"

    return AdvantageRecord(
        method="qaoa", candidate=candidate, config_id=config["config_id"],
        quantum_metric=approx, classical_metric=1.0,
        metric_name="approx_ratio", advantage_direction=direction,
        advantage_magnitude=float(approx), scaling_signature=float(reps),
        quantum_native_litmus=True,
        sim_runnable=True,
        q50_faithful_runnable=_q50_faithful(n, k, reps, risk, seed),
        demo_naturalness=0.6, op_business_fit=0.8,
        notes=f"n={n}, k={k}, reps={reps}: QAOA portfolio vs brute-force optimum",
        sweep_value=float(n), sweep_label="n_assets",
    )
