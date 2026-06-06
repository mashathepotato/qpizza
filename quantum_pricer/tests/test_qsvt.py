import numpy as np
from quantum_pricer import qsvt, tree


def test_relu_poly_approximates_rescaled_payoff():
    # poly(theta) should approximate sqrt(ReLU((theta/c + K) - K)/Cmax) on the oracle grid.
    poly = qsvt.relu_sqrt_poly(degree=20, delta=0.1)
    xs = np.linspace(-1, 1, 200)
    target = np.sqrt(np.maximum(xs, 0.0))
    approx = np.polynomial.chebyshev.chebval(xs, poly) if False else qsvt.eval_poly(poly, xs)
    # smoothed ReLU: allow error away from the kink
    mask = np.abs(xs) > 0.15
    assert np.max(np.abs(approx[mask] - target[mask])) < 0.15


def test_qsvt_block_encoding_matches_target_on_sim(base_params):
    M = 2
    exact = tree.exact_tree_price(M=M, option="european", kind="call", **base_params)
    price = qsvt.price(M=M, option="european", kind="call",
                       degree=30, delta=0.1, use_qae=False, **base_params)
    assert abs(price - exact) < 0.1   # limited by polynomial degree near the kink


def test_qsvt_single_run_qae(base_params):
    M = 2
    exact = tree.exact_tree_price(M=M, option="european", kind="call", **base_params)
    res = qsvt.price(M=M, degree=30, delta=0.1, use_qae=True,
                     epsilon_target=0.02, return_meta=True, **base_params)
    assert abs(res["price"] - exact) < 0.15
    assert res["num_oracle_queries"] > 0
