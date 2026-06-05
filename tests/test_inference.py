import numpy as np
from demo.inference import train_fraud_model, score_transaction


def test_score_transaction_returns_probability():
    model = train_fraud_model(backend="local_aer", n=80, n_features=4, seed=0)
    x = np.zeros(4)
    p = score_transaction(model, x)
    assert 0.0 <= p <= 1.0
