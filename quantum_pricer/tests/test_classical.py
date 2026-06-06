import numpy as np
from quantum_pricer import classical, tree


def test_black_scholes_atm_known_value(base_params):
    # ATM call, S0=K=100, r=0.05, sigma=0.20, T=1 -> ~10.4506 (standard reference)
    price = classical.black_scholes_call(**base_params)
    assert np.isclose(price, 10.4506, atol=1e-3)


def test_mc_matches_exact_tree_price(base_params):
    exact = tree.exact_tree_price(M=6, option="european", kind="call", **base_params)
    price, stderr = classical.monte_carlo_price(
        M=6, option="european", kind="call", n_paths=200_000, seed=0, **base_params)
    assert abs(price - exact) < 4 * stderr + 0.05


def test_mc_error_shrinks_like_sqrt_n(base_params):
    _, se_small = classical.monte_carlo_price(M=6, n_paths=10_000, seed=1, **base_params)
    _, se_big = classical.monte_carlo_price(M=6, n_paths=160_000, seed=1, **base_params)
    # 16x samples -> ~4x smaller stderr
    assert se_big < se_small / 3.0
