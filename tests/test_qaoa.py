import numpy as np
from triage.harness.qaoa import solve_portfolio, run


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
