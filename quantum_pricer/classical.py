"""Classical baselines: Black-Scholes closed form (continuum check) and a
Monte-Carlo pricer over the SAME binomial tree the quantum routes use."""
import numpy as np
from scipy.stats import norm
from quantum_pricer.tree import crr_params


def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def monte_carlo_price(S0, K, r, sigma, T, M, n_paths, option="european",
                      kind="call", seed=None):
    """Sample n_paths Bernoulli(q) trajectories on the tree; return (price, stderr)."""
    rng = np.random.default_rng(seed)
    u, d, q = crr_params(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    ups = rng.random((n_paths, M)) < q          # bool up-moves
    factors = np.where(ups, u, d)
    prices = S0 * np.cumprod(factors, axis=1)   # shape (n_paths, M)
    if option == "european":
        f = prices[:, -1]
    elif option == "asian":
        f = prices.mean(axis=1)
    else:
        raise ValueError(f"unknown option {option!r}")
    payoff = np.maximum(f - K, 0.0) if kind == "call" else np.maximum(K - f, 0.0)
    disc = np.exp(-r * T)
    samples = disc * payoff
    price = float(samples.mean())
    stderr = float(samples.std(ddof=1) / np.sqrt(n_paths))
    return price, stderr
