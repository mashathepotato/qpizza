"""Exact small-portfolio solver: brute force cardinality-constrained selection."""
import itertools
import numpy as np


def exact_portfolio(mu, cov, k: int, risk: float):
    """Maximize  mu.x - risk * x^T cov x  over x in {0,1}^n with sum(x)==k.
    Returns (chosen_indices, objective_value). Brute force — small n only."""
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)
    n = len(mu)
    best, best_val = None, -np.inf
    for combo in itertools.combinations(range(n), k):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        val = mu @ x - risk * (x @ cov @ x)
        if val > best_val:
            best, best_val = combo, val
    return list(best), float(best_val)
