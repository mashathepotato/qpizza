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


def sa_portfolio(mu, cov, k: int, risk: float, seed: int = 0, iters: int = 2000):
    """Simulated-annealing heuristic for cardinality-k portfolio selection.

    Maximises  mu.x - risk * x^T cov x  with sum(x)==k, x in {0,1}^n.
    Each step proposes a swap (remove one chosen asset, add one unchosen asset);
    accepts improvements unconditionally and worse moves with Metropolis
    probability exp(delta/T) under a geometric cooling schedule.

    Returns (chosen_indices_list, objective_value). Deterministic for a given seed.
    """
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)
    n = len(mu)
    rng = np.random.default_rng(seed)

    # Initialise: random k-subset
    perm = rng.permutation(n)
    chosen = set(perm[:k].tolist())
    unchosen = set(perm[k:].tolist())

    def _obj(sel):
        x = np.zeros(n)
        x[list(sel)] = 1.0
        return float(mu @ x - risk * (x @ cov @ x))

    current_val = _obj(chosen)
    best_chosen = set(chosen)
    best_val = current_val

    # Geometric cooling: T_0 chosen so that a typical bad swap (~0.01 objective
    # unit) is accepted ~50 % of the time at the start, cooled to near-zero.
    T0 = 0.1
    T_min = 1e-10
    alpha = (T_min / T0) ** (1.0 / max(iters - 1, 1))
    T = T0

    for _ in range(iters):
        # Propose a random swap
        remove = int(rng.choice(list(chosen)))
        add = int(rng.choice(list(unchosen)))

        new_chosen = (chosen - {remove}) | {add}
        new_val = _obj(new_chosen)
        delta = new_val - current_val

        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-300)):
            chosen = new_chosen
            unchosen = (unchosen - {add}) | {remove}
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_chosen = set(chosen)

        T *= alpha

    return sorted(best_chosen), float(best_val)
