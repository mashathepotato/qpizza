import numpy as np
import pytest

from quantum_pricer import data


def _write_fake_cache(path, start="2024-06-01", n=520, s0=4.0):
    """Deterministic synthetic close series spanning calibration + future windows."""
    from datetime import date, timedelta
    rng = np.random.default_rng(0)
    closes = s0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=n)))
    d0 = date.fromisoformat(start)
    with open(path, "w") as fh:
        fh.write("date,close\n")
        for i, c in enumerate(closes):
            fh.write(f"{d0 + timedelta(days=i)},{c:.6f}\n")
    return path


def test_realized_vol_from_returns():
    rng = np.random.default_rng(0)
    daily = rng.normal(0, 0.02, size=252)
    sigma = data.annualized_vol(daily, periods_per_year=252)
    assert 0.2 < sigma < 0.45  # ~0.02*sqrt(252) ≈ 0.317


def test_no_cache_offline_is_labelled_synthetic(tmp_path):
    params, meta = data.nokia_params(allow_network=False,
                                     cache_path=str(tmp_path / "none.csv"))
    assert meta["source"] == "synthetic"
    assert params["S0"] > 0 and 0 < params["sigma"] < 2 and "r" in params


def test_calibration_never_sees_beyond_asof(tmp_path):
    """Look-ahead guard: calibration window must end on or before t0."""
    cache = _write_fake_cache(str(tmp_path / "fake.csv"))
    asof = "2025-06-05"
    params, meta = data.nokia_params(asof=asof, cache_path=cache,
                                     allow_network=False)
    assert meta["source"] == "yfinance-cache"
    assert meta["calib_end"] <= asof
    assert meta["calib_start"] >= "2024-06-01"
    # S0 is the close AT the calibration end, not a later one
    dates, closes = data.load_history(cache_path=cache, allow_network=False)
    s0_expected = [c for d, c in zip(dates, closes) if d <= asof][-1]
    assert params["S0"] == pytest.approx(s0_expected)


def test_realized_outcome_is_strictly_future(tmp_path):
    """Evaluation window must start after t0 and end at t0 + T."""
    cache = _write_fake_cache(str(tmp_path / "fake.csv"))
    asof = "2025-06-05"
    out = data.realized_outcome(asof=asof, T=1.0, cache_path=cache,
                                allow_network=False)
    assert out["window_start"] > asof
    assert out["S_T_date"] <= out["window_end"]
    assert out["n_obs"] > 30 and out["realized_vol"] > 0


def test_realized_outcome_refuses_thin_window(tmp_path):
    """asof too close to the end of history -> raise, never fabricate."""
    cache = _write_fake_cache(str(tmp_path / "fake.csv"), n=400)
    with pytest.raises(RuntimeError):
        data.realized_outcome(asof="2025-06-20", T=1.0, cache_path=cache,
                              allow_network=False)


def test_committed_cache_is_loadable_offline():
    """The repo-committed CSV must keep results reproducible with no network."""
    import os
    if not os.path.exists(data._default_cache_path("NOKIA.HE")):
        pytest.skip("market cache not present")
    params, meta = data.nokia_params(allow_network=False)
    assert meta["source"] == "yfinance-cache"
    assert meta["calib_end"] <= meta["asof"]
