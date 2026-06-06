import numpy as np
from quantum_pricer import data


def test_realized_vol_from_returns():
    # constant 1% daily up/down alternating -> known annualized vol
    rng = np.random.default_rng(0)
    daily = rng.normal(0, 0.02, size=252)
    sigma = data.annualized_vol(daily, periods_per_year=252)
    assert 0.2 < sigma < 0.45  # ~0.02*sqrt(252) ≈ 0.317


def test_fallback_params_are_labelled_synthetic():
    params, meta = data.nokia_params(allow_network=False)
    assert meta["source"] == "synthetic"
    assert params["S0"] > 0 and 0 < params["sigma"] < 2 and "r" in params
