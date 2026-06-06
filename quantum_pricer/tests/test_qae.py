import numpy as np
from quantum_pricer import qae, tree


def test_qae_price_matches_exact_tree(base_params):
    M = 3
    exact = tree.exact_tree_price(M=M, option="european", kind="call", **base_params)
    result = qae.price(M=M, option="european", kind="call",
                       epsilon_target=0.01, **base_params)
    assert abs(result["price"] - exact) < 0.05
    assert result["num_oracle_queries"] > 0


def test_qae_more_precision_costs_more_queries(base_params):
    coarse = qae.price(M=3, epsilon_target=0.05, **base_params)
    fine = qae.price(M=3, epsilon_target=0.005, **base_params)
    assert fine["num_oracle_queries"] > coarse["num_oracle_queries"]
