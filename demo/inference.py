"""Backend-agnostic inference for the demo. Trains the quantum-kernel fraud model
once and scores individual transactions — the frontrunner candidate, ready to dress.
Also provides thin wrappers for QAE option pricing and QAOA portfolio optimization
so that app logic is testable independently of the Streamlit UI."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.svm import SVC

from triage.data.fraud import load_ulb, make_synthetic_fraud
from triage.harness.fraud_qml import _kernel_matrix


@dataclass
class FraudModel:
    clf: SVC
    X_train: np.ndarray
    n_features: int
    backend: str


def train_fraud_model(backend="local_aer", n=200, n_features=4, seed=0) -> FraudModel:
    try:
        X, y = load_ulb(n=n, n_features=n_features, seed=seed)
    except Exception:
        X, y = make_synthetic_fraud(n=n, n_features=n_features, seed=seed)
    K = _kernel_matrix(X, X, n_features)
    clf = SVC(kernel="precomputed", probability=True, random_state=seed).fit(K, y)
    return FraudModel(clf=clf, X_train=X, n_features=n_features, backend=backend)


def score_transaction(model: FraudModel, x) -> float:
    x = np.asarray(x, float).reshape(1, -1)
    k = _kernel_matrix(x, model.X_train, model.n_features)
    return float(model.clf.predict_proba(k)[0, 1])


def price_option(
    strike: float,
    vol: float,
    maturity: float,
    eps: float,
    n_qubits: int,
    backend: str = "local_aer",
    seed: int = 7,
) -> dict:
    """Price a European call via QAE and compute the classical MC speedup.

    Delegates to ``price_european_call`` from the QAE harness, then enriches
    the result with:
      - ``mc_samples``: classical Monte-Carlo sample count needed for the same
        accuracy (conservative p=0.5 Bernoulli variance proxy via
        ``mc_samples_to_eps``).
      - ``speedup``: mc_samples / oracle_queries — how many fewer resources QAE
        uses compared to MC at the given eps.

    Parameters mirror ``price_european_call`` with ``r`` fixed at 0.05 and
    ``s0`` at 2.0 (sensible defaults for a demo; easily overridden by callers
    who need full control).
    """
    from triage.harness.qae import price_european_call
    from triage.baselines.mc import mc_samples_to_eps

    result = price_european_call(
        num_uncertainty_qubits=int(n_qubits),
        strike=float(strike),
        s0=2.0,
        vol=float(vol),
        r=0.05,
        t_maturity=float(maturity),
        epsilon=float(eps),
        shots=4096,
        seed=int(seed),
    )
    mc_samples = mc_samples_to_eps(p=0.5, eps=float(eps))
    oracle_queries = result["oracle_queries"]
    speedup = float(mc_samples) / float(max(oracle_queries, 1))
    return {
        "price": result["price"],
        "oracle_queries": oracle_queries,
        "n_qubits": result["n_qubits"],
        "exact_payoff": result["exact_payoff"],
        "mc_samples": mc_samples,
        "speedup": speedup,
    }


def optimize_portfolio(
    n_assets: int,
    k: int,
    risk: float,
    reps: int,
    seed: int,
    backend: str = "local_aer",
) -> dict:
    """Select a k-asset portfolio via QAOA and compare to the exact optimum.

    Generates a small random (seeded) mean-return vector and covariance matrix,
    runs ``solve_portfolio`` (QAOA), runs ``exact_portfolio`` (brute force), and
    returns a dict with:
      - ``chosen``: list of selected asset indices
      - ``objective``: QAOA portfolio objective value
      - ``optimum``: exact best objective value
      - ``approx_ratio``: QAOA objective as a fraction of the exact optimum,
        clipped to [0, 1] using a shift to handle near-zero/negative objectives.
    """
    from triage.harness.qaoa import solve_portfolio
    from triage.baselines.classical_opt import exact_portfolio

    rng = np.random.default_rng(int(seed))
    mu = rng.uniform(0.0, 0.2, size=int(n_assets))
    a = rng.normal(0.0, 0.02, size=(int(n_assets), int(n_assets)))
    cov = a @ a.T

    chosen, q_obj = solve_portfolio(
        mu, cov, k=int(k), risk=float(risk),
        reps=int(reps), seed=int(seed), backend=backend,
    )
    _, opt_obj = exact_portfolio(mu, cov, k=int(k), risk=float(risk))

    # Approx ratio: shift both by |min| + epsilon so ratio is always in [0, 1].
    shift = abs(min(q_obj, opt_obj)) + 1e-6
    approx_ratio = float(
        min(max((q_obj + shift) / (opt_obj + shift), 0.0), 1.0)
    )
    return {
        "chosen": chosen,
        "objective": float(q_obj),
        "optimum": float(opt_obj),
        "approx_ratio": approx_ratio,
    }
