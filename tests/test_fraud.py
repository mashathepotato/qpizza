import numpy as np
import pytest
from triage.data.fraud import make_synthetic_fraud, prepare_features, make_hard_fraud, scale_for_embedding
from triage.harness.fraud_qml import quantum_kernel_auc, run

def test_prepare_features_caps_dimension():
    X = np.random.default_rng(0).normal(size=(200, 28))
    Xr = prepare_features(X, n_features=8)
    assert Xr.shape == (200, 8)

def test_quantum_kernel_separates_synthetic_fraud():
    X, y = make_synthetic_fraud(n=80, n_features=4, seed=0)
    auc = quantum_kernel_auc(X, y, backend="local_aer", seed=0)
    assert auc > 0.8

def test_run_returns_record_with_auc():
    rec = run({"config_id": "d_smoke", "candidate": "D", "n": 60,
               "n_features": 4, "backend": "local_aer", "seed": 0})
    assert rec.method == "fraud_qml"
    assert rec.metric_name == "auc"
    assert 0.0 <= rec.quantum_metric <= 1.0

def test_scale_for_embedding_bounds_angles():
    """scale_for_embedding must map every feature into [0, pi] (per-feature min-max)."""
    eps = 1e-9
    rng = np.random.default_rng(42)
    # Intentionally wild-range input: values well outside [-3, 3]
    X_train = rng.normal(0, 5, (100, 6))
    X_test = rng.normal(0, 5, (30, 6))
    scaler, X_train_scaled = scale_for_embedding(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Training data must be strictly within [0, pi]
    assert X_train_scaled.min() >= 0.0 - eps, f"min={X_train_scaled.min()}"
    assert X_train_scaled.max() <= np.pi + eps, f"max={X_train_scaled.max()}"
    # Test data may extrapolate slightly, but scaler output should still be ~[0,pi]
    # (we test that the scaler was fitted on train; test values can be checked loosely)
    assert X_test_scaled.shape == X_test.shape

def test_hard_dataset_runs_and_reports_honestly():
    """The hard dataset run must produce a valid AdvantageRecord with measured AUCs in [0,1]
    and an advantage_direction in {win, tie, loss}.  Any direction is acceptable — this
    test is explicitly NOT checking for a quantum win."""
    rec = run({
        "config_id": "h",
        "candidate": "D",
        "n": 120,
        "n_features": 4,
        "dataset": "hard",
        "backend": "local_aer",
        "seed": 0,
    })
    assert rec.method == "fraud_qml"
    assert rec.metric_name == "auc"
    assert 0.0 <= rec.quantum_metric <= 1.0, f"q_auc={rec.quantum_metric}"
    assert 0.0 <= rec.classical_metric <= 1.0, f"c_auc={rec.classical_metric}"
    assert rec.advantage_direction in {"win", "tie", "loss"}, \
        f"unexpected direction: {rec.advantage_direction}"
    # Report what actually happened (will appear in pytest -v output)
    print(
        f"\n[HARD] q_auc={rec.quantum_metric:.4f}  c_auc={rec.classical_metric:.4f}"
        f"  direction={rec.advantage_direction}"
    )
