import math
from backends import get_backend
from triage.harness.qae import estimate_bernoulli, run


def test_qae_estimates_known_bernoulli_probability():
    backend = get_backend("local_aer")
    p = 0.2
    est = estimate_bernoulli(p, backend=backend, epsilon=0.02)
    assert abs(est - p) < 0.05


def test_run_returns_record_with_scaling_advantage():
    rec = run({"config_id": "b_smoke", "candidate": "B", "p": 0.3,
               "epsilon": 0.05, "backend": "local_aer"})
    assert rec.method == "qae"
    assert rec.metric_name == "samples_to_eps"
    assert rec.quantum_metric < rec.classical_metric
    assert rec.advantage_direction in {"win", "tie", "loss"}
