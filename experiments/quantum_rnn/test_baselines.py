"""Tests for the floor baselines (baselines.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as bl
import classical_rnn as cr


def test_persistence_scores_are_last_abs_return():
    X_raw = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, -0.5]])
    s = bl.persistence_scores(X_raw)
    np.testing.assert_allclose(s, [0.3, 0.5])


def test_logistic_forward_is_probability():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (5, 6))
    model = bl.LogisticClassifier(n_features=6, seed=0)
    s = cr.scores(model, X)
    assert s.shape == (5,)
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_logistic_training_reduces_loss():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (60, 6)); y = (X.mean(1) > 0).astype(int)
    model = bl.LogisticClassifier(n_features=6, seed=0)
    hist = cr.train(model, X, y, epochs=80, lr=0.1, seed=0)
    assert hist[-1] < hist[0]
