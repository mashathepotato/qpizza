"""Fraud data: real ULB credit-card set if present, else a synthetic stand-in.

The ULB Kaggle file (creditcard.csv) has PCA features V1..V28 + Amount + Class.
Tonight we run on synthetic data so the build never blocks on a download; swap in
the real CSV via load_ulb() when available."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def make_synthetic_fraud(n: int = 200, n_features: int = 8, seed: int = 0):
    """Imbalanced two-class blob set resembling fraud (rare positive class)."""
    rng = np.random.default_rng(seed)
    n_pos = max(4, n // 10)
    n_neg = n - n_pos
    neg = rng.normal(0.0, 1.0, (n_neg, n_features))
    pos = rng.normal(1.8, 1.0, (n_pos, n_features))  # separable-ish shift
    X = np.vstack([neg, pos])
    y = np.array([0] * n_neg + [1] * n_pos)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def prepare_features(X, n_features: int = 8):
    """Standardize then PCA-reduce to n_features (<=10 for <=10 qubits)."""
    Xs = StandardScaler().fit_transform(np.asarray(X, float))
    if Xs.shape[1] <= n_features:
        return Xs
    return PCA(n_components=n_features, random_state=0).fit_transform(Xs)


def load_ulb(path: str = "data/raw/creditcard.csv", n: int = 400,
             n_features: int = 8, seed: int = 0):
    """Load + subsample the real ULB set. Falls back to synthetic if missing."""
    if not os.path.exists(path):
        return make_synthetic_fraud(n=n, n_features=n_features, seed=seed)
    df = pd.read_csv(path)
    pos = df[df["Class"] == 1]
    neg = df[df["Class"] == 0].sample(n=min(len(df) - len(pos), n - len(pos)),
                                      random_state=seed)
    sub = pd.concat([pos.sample(n=min(len(pos), n // 10), random_state=seed), neg])
    y = sub["Class"].to_numpy()
    X = sub.drop(columns=["Class", "Time"], errors="ignore").to_numpy()
    return prepare_features(X, n_features), y
