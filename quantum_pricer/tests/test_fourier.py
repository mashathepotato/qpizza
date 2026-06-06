import numpy as np
from quantum_pricer import fourier, tree


def test_fourier_price_matches_exact_tree_statevector(base_params):
    M = 3
    exact = tree.exact_tree_price(M=M, option="european", kind="call", **base_params)
    price = fourier.price(M=M, option="european", kind="call",
                          n_lambda=24, shots=None, **base_params)  # shots=None -> exact SV
    assert abs(price - exact) < 1e-3


def test_fourier_price_matches_exact_tree_with_shots(base_params):
    M = 2
    exact = tree.exact_tree_price(M=M, option="european", kind="call", **base_params)
    price = fourier.price(M=M, option="european", kind="call",
                          n_lambda=16, shots=200_000, seed=0, **base_params)
    assert abs(price - exact) < 0.15
