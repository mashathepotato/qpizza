import math
from backends import get_backend
from triage.harness.qae import (
    estimate_bernoulli,
    run,
    _qae_oracle_calls,
    scaling_curve,
    price_european_call,
)


def test_qae_estimates_known_bernoulli_probability():
    backend = get_backend("local_aer")
    p = 0.2
    # shot-based IQAE: use enough shots that the 0.05 tolerance is robust to noise
    est = estimate_bernoulli(p, backend=backend, epsilon=0.02, shots=4096)
    assert abs(est - p) < 0.05


def test_run_returns_record_with_scaling_advantage():
    # Use eps=0.01: the MEASURED oracle-query advantage is asymptotic, so it
    # shows up at small eps (where IQAE's ~1/eps cost falls below MC's ~1/eps^2).
    # At coarse eps the per-round shot constant dominates and QAE need not win.
    rec = run({"config_id": "b_smoke", "candidate": "B", "p": 0.3,
               "epsilon": 0.01, "backend": "local_aer", "shots": 2048})
    assert rec.method == "qae"
    assert rec.metric_name == "samples_to_eps"
    assert rec.quantum_metric < rec.classical_metric
    assert rec.advantage_direction in {"win", "tie", "loss"}


def test_oracle_count_is_measured_not_analytic():
    # The measured oracle-query count comes from a real shot-based IQAE run.
    # It must NOT equal the analytic fallback ceil(1/0.05) == 20, and be > 0.
    calls = _qae_oracle_calls(0.3, 0.05, shots=2048)
    assert calls > 0
    assert calls != int(math.ceil(1.0 / 0.05))  # != 20


def test_scaling_quantum_beats_classical_slope():
    c = scaling_curve(shots=2048)
    # measured quantum scaling is gentler than MC's O(1/eps^2)
    assert abs(c["q_slope"]) < abs(c["mc_slope"])
    # at the smallest eps the quantum cost is far below MC's sample cost
    assert c["q_queries"][-1] < c["mc_samples"][-1]


def test_price_european_call_matches_discretized_expectation():
    # Small uncertainty register + modest shots so this stays fast (<60s).
    res = price_european_call(
        num_uncertainty_qubits=3, strike=1.9, s0=2.0, vol=0.4,
        r=0.05, t_maturity=0.1, epsilon=0.01, shots=4096,
    )
    assert res["price"] > 0.0
    assert res["oracle_queries"] > 0
    assert res["n_qubits"] > 3  # uncertainty register + payoff/ancilla qubits
    exact = res["exact_payoff"]
    # QAE price is a real shot-based amplitude estimate; confirm it tracks the
    # exact discretized expected payoff the model loads.
    assert abs(res["price"] - exact) < 0.05, (res["price"], exact)


def test_run_european_call_mode_returns_record():
    rec = run({"config_id": "opt0", "candidate": "B",
               "mode": "european_call", "epsilon": 0.01})
    assert rec.method == "qae"
    assert rec.metric_name == "samples_to_eps"
    assert rec.quantum_metric > 0
    assert "European" in rec.notes


def test_qae_oracle_count_is_reproducible():
    """Same seed must yield the EXACT same oracle-query count."""
    count1 = _qae_oracle_calls(0.3, 0.02, seed=7)
    count2 = _qae_oracle_calls(0.3, 0.02, seed=7)
    assert count1 == count2, (
        f"Oracle counts differ across runs with seed=7: {count1} vs {count2}"
    )
