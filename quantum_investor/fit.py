"""Fit the quantum model's parameters (alpha, beta) to the human data."""

import numpy as np
from scipy.optimize import minimize

import quantum_model as qm


def _sse(params, data):
    alpha, beta = params
    pred = qm.predict(alpha, beta)
    err = 0.0
    for order in ("AB", "BA"):
        for cell in ("yy", "yn", "ny", "nn"):
            err += (pred[order][cell] - data[order][cell]) ** 2
    return err


def fit_quantum(data, restarts=12, seed=0):
    """
    Least-squares fit of (alpha, beta) to the data over BOTH measurement orders.

    Multi-start to avoid local minima (the landscape is periodic). Returns
    (alpha, beta, sse).
    """
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = rng.uniform(0.0, np.pi, size=2)
        res = minimize(_sse, x0, args=(data,), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    alpha, beta = best.x
    return float(alpha), float(beta), float(best.fun)
