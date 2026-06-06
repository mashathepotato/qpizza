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


def test_nokia_price_series_offline_is_labelled_synthetic_and_well_formed():
    from quantum_pricer import data
    series, meta = data.nokia_price_series(allow_network=False, n_synth=120, seed=0)
    # provenance is explicit
    assert meta["source"] == "synthetic"
    # series is internally consistent and plottable
    assert len(series["dates"]) == len(series["closes"]) == 120
    assert all(c > 0 for c in series["closes"])
    assert series["S0"] == series["closes"][-1]
    assert series["sigma"] > 0
    # deterministic given the seed
    series2, _ = data.nokia_price_series(allow_network=False, n_synth=120, seed=0)
    assert series2["closes"] == series["closes"]
