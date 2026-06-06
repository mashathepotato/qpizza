"""Task 11 (nice-to-have): Asian (path-dependent) option via the same circuits.

The Asian option uses the arithmetic-average path value instead of the terminal
price; everything else — the quantum oracles, QAE, QSVT — is unchanged because
`payoff_variable_values(option="asian")` already computes the path averages.

Tests assert:
  1. Asian exact tree price agrees with Asian Monte Carlo (within 4*stderr + 0.05).
  2. QAE prices an Asian call (M=3) within 0.05 of the exact Asian tree price.
"""
import numpy as np
import pytest
from quantum_pricer import tree, classical, qae


@pytest.fixture
def asian_params():
    """ATM parameters for a fast Asian-call test."""
    return dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)


def test_asian_mc_vs_tree(asian_params):
    """Asian exact tree price must agree with Asian Monte Carlo within 4*stderr + 0.05."""
    M = 3
    exact = tree.exact_tree_price(M=M, option="asian", kind="call", **asian_params)
    mc_price, stderr = classical.monte_carlo_price(
        M=M, n_paths=50_000, option="asian", kind="call", seed=0, **asian_params
    )
    tol = 4.0 * stderr + 0.05
    assert abs(mc_price - exact) <= tol, (
        f"Asian MC {mc_price:.6f} deviates from tree {exact:.6f} by "
        f"{abs(mc_price - exact):.6f} > tol {tol:.6f} (stderr={stderr:.6f})"
    )


def test_asian_qae_vs_tree(asian_params):
    """QAE with option='asian' must match Asian exact tree price within 0.05."""
    M = 3
    exact = tree.exact_tree_price(M=M, option="asian", kind="call", **asian_params)
    result = qae.price(
        M=M, option="asian", kind="call",
        epsilon_target=0.01, **asian_params
    )
    assert result["num_oracle_queries"] > 0, "IAE should perform at least one Grover query"
    assert abs(result["price"] - exact) < 0.05, (
        f"Asian QAE {result['price']:.6f} deviates from tree {exact:.6f} by "
        f"{abs(result['price'] - exact):.6f} >= 0.05"
    )
