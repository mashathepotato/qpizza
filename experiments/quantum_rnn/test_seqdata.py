"""Tests for the shared sequence-task harness (seqdata.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seqdata as sd


def test_make_sequences_shape_and_content():
    r = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    s = sd.make_sequences(r, L=3)
    assert s.shape == (3, 3)
    np.testing.assert_allclose(s[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(s[-1], [0.3, 0.4, 0.5])


def test_zscore_uses_training_stats_only():
    train = np.array([[0.0, 2.0], [2.0, 4.0]])  # train mean=2, std=...
    mean, std = sd.zscore_fit(train)
    assert mean == pytest.approx(2.0)
    z = sd.zscore_apply(train, mean, std)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)


def test_build_task_has_no_label_leakage():
    # target for sequence i must be return[i+L], which is NOT inside sequence i
    r = np.arange(1, 21, dtype=float) / 100.0
    task = sd.build_task_from_returns(r, L=5, train_frac=0.5, proxy_q=0.5)
    X_raw, nextabs = task["X_raw"], task["next_abs"]
    # sequence 0 = r[0:5]; its target return is r[5] (NOT inside the sequence)
    np.testing.assert_allclose(X_raw[0], r[0:5])
    np.testing.assert_allclose(nextabs[0], abs(r[5]))
    assert r[5] not in X_raw[0]
    assert len(X_raw) == len(task["y"]) == len(nextabs)


def test_build_task_split_and_label_fraction():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, 300)
    task = sd.build_task_from_returns(r, L=10, train_frac=0.5, proxy_q=0.9)
    n = len(task["X"])
    assert task["n_train"] == n // 2
    # proxy threshold from TRAIN portion; ~10% of train flagged
    y_train = task["y"][: task["n_train"]]
    assert 0.05 <= y_train.mean() <= 0.15


def test_roc_auc_perfect_and_random():
    assert sd.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert sd.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == pytest.approx(0.0)
