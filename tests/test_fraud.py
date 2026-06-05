import numpy as np
from triage.data.fraud import make_synthetic_fraud, prepare_features
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
