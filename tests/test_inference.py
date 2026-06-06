import numpy as np
from demo.inference import (
    train_fraud_model,
    score_transaction,
    price_option,
    optimize_portfolio,
)


def test_score_transaction_returns_probability():
    model = train_fraud_model(backend="local_aer", n=80, n_features=4, seed=0)
    x = np.zeros(4)
    p = score_transaction(model, x)
    assert 0.0 <= p <= 1.0


def test_price_option_reports_speedup():
    r = price_option(
        strike=1.9, vol=0.4, maturity=0.1, eps=0.01,
        n_qubits=3, backend="local_aer"
    )
    assert r["price"] > 0
    assert r["oracle_queries"] > 0
    assert r["mc_samples"] > 0
    assert "speedup" in r


def test_optimize_portfolio_returns_valid_selection():
    r = optimize_portfolio(n_assets=4, k=2, risk=1.0, reps=2, seed=1,
                           backend="local_aer")
    assert len(r["chosen"]) == 2
    assert 0 <= r["approx_ratio"] <= 1
