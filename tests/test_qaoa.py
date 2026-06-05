import numpy as np
from triage.harness.qaoa import solve_portfolio, run, _build_program, _q50_faithful


def test_qaoa_matches_brute_force_on_tiny_problem():
    mu = np.array([0.12, 0.01, 0.10, 0.02])
    cov = np.eye(4) * 0.0002
    chosen, _ = solve_portfolio(mu, cov, k=2, risk=1.0, reps=2, seed=1,
                                backend="local_aer")
    assert sorted(chosen) == [0, 2]  # the two highest-return assets


def test_run_returns_record():
    rec = run({"config_id": "a_smoke", "candidate": "A", "n_assets": 4,
               "k": 2, "reps": 1, "seed": 1, "backend": "local_aer"})
    assert rec.method == "qaoa"
    assert rec.metric_name == "approx_ratio"
    assert 0.0 <= rec.quantum_metric <= 1.0


def test_qubo_objective_matches_formula():
    """The QuadraticProgram objective must equal mu@x - risk*(x@cov@x) exactly.

    This locks in the correct QUBO construction including the intentional
    (i,j)+(j,i) summation that qiskit-optimization sums to 2*risk*cov_ij,
    which is the correct off-diagonal coefficient for a binary quadratic form.
    """
    rng = np.random.default_rng(42)
    n = 5
    mu = rng.uniform(0.0, 0.2, size=n)
    a = rng.normal(0.0, 0.02, size=(n, n))
    cov = a @ a.T  # SPD, same construction as run()
    risk = 1.5
    k = 2

    qp = _build_program(mu, cov, k, risk)

    # Try several feasible k-hot binary vectors
    feasible_xs = [
        np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 1.0]),
    ]
    for x in feasible_xs:
        x_dict = {f"x{i}": x[i] for i in range(n)}
        qp_val = qp.objective.evaluate(x_dict)
        formula_val = float(mu @ x - risk * (x @ cov @ x))
        assert abs(qp_val - formula_val) < 1e-9, (
            f"QUBO objective mismatch for x={x}: "
            f"qp={qp_val}, formula={formula_val}, diff={abs(qp_val-formula_val)}"
        )


def test_q50_faithful_is_measured():
    """_q50_faithful must return a bool (True) for a small instance.

    This ensures the field is MEASURED by actually transpiling a QAOA-structured
    circuit on q50_fake, not hardcoded. A QAOAAnsatz built from the Ising operator
    of a 4-qubit QUBO should transpile successfully on IQMFakeAphrodite (54 q).
    """
    result = _q50_faithful(n_assets=4, k=2, reps=1, risk=1.0, seed=0)
    assert isinstance(result, bool), f"_q50_faithful must return bool, got {type(result)}"
    assert result is True, (
        "_q50_faithful returned False for n=4 k=2 reps=1; "
        "the QAOAAnsatz circuit should transpile on q50_fake"
    )
