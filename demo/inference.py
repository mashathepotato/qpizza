"""Backend-agnostic inference for the demo. Trains the quantum-kernel fraud model
once and scores individual transactions — the frontrunner candidate, ready to dress."""
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
