"""Tests for the classical and quantum transformer classifiers."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classical_tf as ct
import quantum_tf as qt
import classical_rnn as cr   # shared train/scores/count_params

MODELS = [ct.TransformerClassifier, qt.QuantumTransformer]


def _Cls(M, seed):
    return M(seed=seed)


def test_forward_outputs_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (5, 10))
    for M in MODELS:
        s = cr.scores(_Cls(M, 0), X)
        assert s.shape == (5,)
        assert np.all((s >= 0.0) & (s <= 1.0)), M.__name__


def test_deterministic_under_seed():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (20, 10)); y = (X.mean(1) > 0).astype(int)
    for M in MODELS:
        a = _Cls(M, 3); cr.train(a, X, y, epochs=6, seed=3)
        b = _Cls(M, 3); cr.train(b, X, y, epochs=6, seed=3)
        np.testing.assert_allclose(cr.scores(a, X), cr.scores(b, X),
                                   rtol=1e-5, atol=1e-6, err_msg=M.__name__)


def test_training_reduces_loss_on_separable_signal():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (50, 10)); y = (X.mean(1) > 0).astype(int)
    for M in MODELS:
        model = _Cls(M, 0)
        hist = cr.train(model, X, y, epochs=60, lr=0.05, seed=0)
        assert hist[-1] < hist[0], M.__name__


def test_param_counts_positive_and_quantum_is_lean():
    n_classical = cr.count_params(ct.TransformerClassifier(seed=0))
    n_quantum = cr.count_params(qt.QuantumTransformer(seed=0))
    assert n_classical > 0 and n_quantum > 0
    # the quantum self-attention uses far fewer params than the classical encoder
    assert n_quantum < n_classical
