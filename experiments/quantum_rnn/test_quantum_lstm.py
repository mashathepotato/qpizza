"""Tests for the quantum LSTM classifier QLSTM (quantum_lstm.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantum_lstm as ql
import classical_rnn as cr   # shared train/scores/count_params


def test_forward_outputs_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (5, 6))
    model = ql.QLSTM(n_qubits=4, n_layers=2, seed=0)
    s = cr.scores(model, X)
    assert s.shape == (5,)
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_deterministic_under_seed():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (20, 5)); y = (X.mean(1) > 0).astype(int)
    a = ql.QLSTM(seed=3); cr.train(a, X, y, epochs=8, seed=3)
    b = ql.QLSTM(seed=3); cr.train(b, X, y, epochs=8, seed=3)
    np.testing.assert_allclose(cr.scores(a, X), cr.scores(b, X), rtol=1e-5, atol=1e-6)


def test_training_reduces_loss_on_separable_signal():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (40, 5)); y = (X.mean(1) > 0).astype(int)
    model = ql.QLSTM(n_qubits=4, n_layers=2, seed=0)
    hist = cr.train(model, X, y, epochs=40, lr=0.08, seed=0)
    assert hist[-1] < hist[0]


def test_param_count_small():
    model = ql.QLSTM(n_qubits=4, n_layers=2, seed=0)
    n = cr.count_params(model)
    assert n > 0
    # four gate VQCs + small linear head: sanity upper bound
    assert n < 400
