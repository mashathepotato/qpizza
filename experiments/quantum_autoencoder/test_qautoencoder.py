"""Tests for the quantum autoencoder anomaly detector (qautoencoder.py).

The detector trains a Romero-style quantum autoencoder on 'calm' NOKIA.HE
return windows so the trash qubits disentangle to |0>; reconstruction
infidelity 1 - P(trash=|00>) is the anomaly score. A numpy-SVD PCA
reconstructor is the classical baseline.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qautoencoder as qa


def test_make_windows_shape_and_content():
    returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    w = qa.make_windows(returns, k=4)
    assert w.shape == (2, 4)
    np.testing.assert_allclose(w[0], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(w[1], [0.2, 0.3, 0.4, 0.5])


def test_zscore_uses_training_stats_only():
    train = np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    mean, std = qa.zscore_fit(train)
    np.testing.assert_allclose(mean, [1.0, 1.0, 1.0, 1.0])
    # applying to the training data yields zero mean per column
    z = qa.zscore_apply(train, mean, std)
    np.testing.assert_allclose(z.mean(axis=0), [0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_qae_param_count_matches_formula():
    model = qa.QuantumAutoencoder(n_qubits=4, n_latent=2, n_layers=2, seed=0)
    # 2 rotations (RY, RZ) per wire per layer
    assert model.n_params == 2 * 4 * 2 == 16


def test_score_is_a_probability():
    model = qa.QuantumAutoencoder(n_qubits=4, n_latent=2, n_layers=2, seed=0)
    s = model.score(np.array([0.1, -0.2, 0.3, -0.1]))
    assert 0.0 <= s <= 1.0


def test_training_reduces_loss_on_calm_signal():
    rng = np.random.default_rng(0)
    # a 'calm' regime: small, similar return windows
    windows = rng.normal(0.0, 0.3, size=(40, 4))
    model = qa.QuantumAutoencoder(n_qubits=4, n_latent=2, n_layers=2, seed=1)
    before = model.loss(model.params, windows)
    model.train(windows, steps=60, lr=0.1)
    after = model.loss(model.params, windows)
    assert after < before


def test_score_is_deterministic_under_seed():
    model = qa.QuantumAutoencoder(n_qubits=4, n_latent=2, n_layers=2, seed=2)
    w = np.array([0.4, -0.3, 0.2, 0.1])
    assert model.score(w) == model.score(w)


def test_injected_spike_scores_higher_than_calm():
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0, 0.5, size=(50, 4))
    model = qa.QuantumAutoencoder(n_qubits=4, n_latent=2, n_layers=2, seed=4)
    model.train(calm, steps=80, lr=0.1)
    calm_scores = np.array([model.score(w) for w in calm])
    spike = np.array([6.0, -6.0, 5.0, -5.0])  # far outside the calm regime
    assert model.score(spike) > calm_scores.mean()


def test_pca_reconstruction_error_nonnegative():
    rng = np.random.default_rng(5)
    windows = rng.normal(0.0, 0.4, size=(30, 4))
    pca = qa.PCAReconstructor(n_components=2)
    pca.fit(windows)
    assert all(pca.score(w) >= 0.0 for w in windows)


def test_pca_spike_scores_higher_than_calm():
    rng = np.random.default_rng(6)
    calm = rng.normal(0.0, 0.4, size=(40, 4))
    pca = qa.PCAReconstructor(n_components=2)
    pca.fit(calm)
    calm_scores = np.array([pca.score(w) for w in calm])
    spike = np.array([8.0, -7.0, 6.0, -8.0])
    assert pca.score(spike) > calm_scores.mean()


def test_roc_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert qa.roc_auc(scores, labels) == pytest.approx(1.0)
