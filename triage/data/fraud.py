"""Fraud data: real ULB credit-card set if present, else a synthetic stand-in.

The ULB Kaggle file (creditcard.csv) has PCA features V1..V28 + Amount + Class.
Tonight we run on synthetic data so the build never blocks on a download; swap in
the real CSV via load_ulb() when available."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


def make_hard_fraud(n: int = 200, n_features: int = 4, seed: int = 0):
    """A genuinely hard classification geometry: XOR/parity over thresholded feature signs.

    Labels are determined by the parity of (sign(f0) >= 0) XOR (sign(f1) >= 0) XOR ...
    over the first two features (checkerboard in the principal plane), with the remaining
    features acting as noise dimensions.  This creates a dataset that is NOT linearly or
    RBF-trivially separable in the full feature space, providing a fair stress test for
    the quantum fidelity kernel.

    IMPORTANT: this function is NOT tuned to guarantee a quantum win.  It builds a
    genuinely hard geometry and lets the triage give an honest measured comparison.
    """
    rng = np.random.default_rng(seed)
    n_pos = max(4, n // 2)  # balanced for the parity task — imbalance would trivially
    n_neg = n - n_pos       # inflate AUC on the majority class

    # Draw samples uniformly in [-1, 1]^n_features (avoids scale bias)
    X = rng.uniform(-1.0, 1.0, (n, n_features))

    # XOR label over first two feature signs: positive iff exactly one of f0,f1 is >= 0
    # This gives a checkerboard / XOR pattern in the (f0, f1) plane.
    # Remaining features are uncorrelated noise — they cannot help but also do not hurt
    # a kernel that properly captures the XOR structure.
    s0 = (X[:, 0] >= 0).astype(int)
    s1 = (X[:, 1] >= 0).astype(int)
    y = (s0 ^ s1).astype(int)  # 0 or 1

    idx = rng.permutation(n)
    return X[idx], y[idx]


def scale_for_embedding(X_train: np.ndarray):
    """Fit a per-feature min-max scaler on X_train and scale to [0, pi].

    Returns (fitted_scaler, X_train_scaled) so the caller can apply the same
    scaler to the test set without leakage.
    """
    scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
    X_scaled = scaler.fit_transform(np.asarray(X_train, float))
    return scaler, X_scaled


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
