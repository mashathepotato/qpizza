"""Tests for the classical PyTorch recurrent classifiers (classical_rnn.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classical_rnn as cr

CLASSES = [cr.RNNClassifier, cr.GRUClassifier, cr.LSTMClassifier]


@pytest.mark.parametrize("Cls", CLASSES)
def test_forward_outputs_probabilities(Cls):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (5, 6))
    model = Cls(hidden=8, seed=0)
    s = cr.scores(model, X)
    assert s.shape == (5,)
    assert np.all((s >= 0.0) & (s <= 1.0))


@pytest.mark.parametrize("Cls", CLASSES)
def test_deterministic_under_seed(Cls):
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (30, 6)); y = (X.mean(1) > 0).astype(int)
    a = Cls(hidden=8, seed=3); cr.train(a, X, y, epochs=20, seed=3)
    b = Cls(hidden=8, seed=3); cr.train(b, X, y, epochs=20, seed=3)
    np.testing.assert_allclose(cr.scores(a, X), cr.scores(b, X))


@pytest.mark.parametrize("Cls", CLASSES)
def test_training_reduces_loss_on_separable_signal(Cls):
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (60, 6)); y = (X.mean(1) > 0).astype(int)
    model = Cls(hidden=8, seed=0)
    hist = cr.train(model, X, y, epochs=80, lr=0.05, seed=0)
    assert hist[-1] < hist[0]


@pytest.mark.parametrize("Cls", CLASSES)
def test_param_count_positive(Cls):
    assert cr.count_params(Cls(hidden=8, seed=0)) > 0


def test_train_accepts_weight_decay_and_still_learns():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (60, 6)); y = (X.mean(1) > 0).astype(int)
    model = cr.LSTMClassifier(hidden=8, seed=0)
    hist = cr.train(model, X, y, epochs=80, lr=0.05, seed=0, weight_decay=1e-2)
    assert hist[-1] < hist[0]
