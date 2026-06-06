import numpy as np
from quantum_pricer import tree


def test_risk_neutral_prob_in_unit_interval(base_params):
    u, d, q = tree.crr_params(M=4, **base_params)
    assert 0.0 < q < 1.0
    assert u > 1.0 > d > 0.0


def test_angles_recover_q():
    # theta = 2 arcsin(sqrt(q))  =>  sin^2(theta/2) = q
    q = 0.37
    theta = tree.angle_from_q(q)
    assert np.isclose(np.sin(theta / 2) ** 2, q)


def test_payoff_values_european_indexing(base_params):
    # M=1: index 0 (down) -> S0*d, index 1 (up) -> S0*u
    u, d, _ = tree.crr_params(M=1, **base_params)
    vals = tree.payoff_variable_values(M=1, option="european", **base_params)
    assert np.isclose(vals[0], base_params["S0"] * d)
    assert np.isclose(vals[1], base_params["S0"] * u)


def test_exact_tree_price_converges_to_black_scholes(base_params):
    # The enumerated tree price must approach the BS price as M grows.
    from quantum_pricer.classical import black_scholes_call
    bs = black_scholes_call(**base_params)
    price_M12 = tree.exact_tree_price(M=12, option="european", kind="call", **base_params)
    assert abs(price_M12 - bs) < 0.5  # loose; tighter convergence checked in classical tests


def test_exact_tree_price_recombining_matches_full_enumeration(base_params):
    # The O(M) binomial-sum European price must match the 2^M full enumeration
    # to machine precision (they compute the same expectation, different routes).
    for M in (6, 10):
        full = tree.exact_tree_price(M=M, option="european", kind="call", **base_params)
        fast = tree.exact_tree_price_recombining(M=M, kind="call", **base_params)
        assert abs(full - fast) < 1e-9
