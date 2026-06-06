"""Cox-Ross-Rubinstein binomial tree: risk-neutral probabilities, loading angles,
and the EXACT tree price by full path enumeration (the quantum ground truth).

Bit convention: path qubit i (0-indexed) is time step i+1; integer path index
x = sum_i bit_i * 2**i  (qubit 0 = LSB, matching Qiskit Statevector ordering).
"""
import numpy as np


def crr_params(S0, K, r, sigma, T, M):
    """Return (u, d, q): up/down factors and risk-neutral up-probability."""
    dt = T / M
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    u = np.exp(drift + vol)
    d = np.exp(drift - vol)
    q = (np.exp(r * dt) - d) / (u - d)
    return u, d, q


def angle_from_q(q):
    """Loading angle theta_i = 2 arcsin(sqrt(q)) for the R_Y path loader."""
    return 2.0 * np.arcsin(np.sqrt(q))


def loading_angles(S0, K, r, sigma, T, M):
    """Per-step R_Y angles (constant across steps for time-independent params)."""
    _, _, q = crr_params(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    return [angle_from_q(q)] * M


def _bits(x, M):
    """bit_i of integer path index x, i = qubit/step index."""
    return [(x >> i) & 1 for i in range(M)]


def payoff_variable_values(S0, K, r, sigma, T, M, option="european"):
    """Array of length 2**M of f(x): terminal price (European) or path average (Asian),
    indexed by integer path index x (see bit convention)."""
    u, d, _ = crr_params(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    vals = np.empty(2 ** M)
    for x in range(2 ** M):
        bits = _bits(x, M)
        prices = []
        s = S0
        for b in bits:
            s = s * (u if b else d)
            prices.append(s)
        if option == "european":
            vals[x] = prices[-1]            # terminal price S_T
        elif option == "asian":
            vals[x] = float(np.mean(prices))  # arithmetic average S_bar
        else:
            raise ValueError(f"unknown option {option!r}")
    return vals


def path_probabilities(S0, K, r, sigma, T, M):
    """Array of length 2**M of p(x) = prod q^{x_i}(1-q)^{1-x_i}."""
    _, _, q = crr_params(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    p = np.empty(2 ** M)
    for x in range(2 ** M):
        bits = _bits(x, M)
        prob = 1.0
        for b in bits:
            prob *= q if b else (1 - q)
        p[x] = prob
    return p


def exact_tree_price(S0, K, r, sigma, T, M, option="european", kind="call"):
    """Discounted expected payoff over ALL 2**M paths — the quantum ground truth."""
    vals = payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, option=option)
    p = path_probabilities(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    if kind == "call":
        payoff = np.maximum(vals - K, 0.0)
    elif kind == "put":
        payoff = np.maximum(K - vals, 0.0)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return float(np.exp(-r * T) * np.sum(p * payoff))
