"""Tests for the autoregressive statistical baselines (ar_baseline.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ar_baseline as ar


def _clustered_returns(seed=0):
    # two volatility bursts -- one INSIDE the training window (so the models can
    # learn persistence), one in the test region (so we can check elevation).
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, 250)
    r[60:90] = rng.normal(0, 0.06, 30)        # burst 1 (train, fit_end=120)
    r[160:200] = rng.normal(0, 0.06, 40)      # burst 2 (test)
    return r


FIT_END = 120
CALM = slice(120, 155)                         # post-fit, pre-burst-2 calm region
BURST = slice(160, 200)                        # test-region burst


def test_garch_forecast_positive_and_full_length():
    r = _clustered_returns()
    h = ar.garch_forecast(r, fit_end=FIT_END)
    assert h.shape == r.shape
    assert np.all(h > 0.0)


def test_garch_forecasts_higher_vol_in_burst():
    r = _clustered_returns()
    h = ar.garch_forecast(r, fit_end=FIT_END)
    assert h[BURST].mean() > h[CALM].mean()


def test_ar_abs_forecast_finite_after_lags():
    r = _clustered_returns()
    fc = ar.ar_abs_forecast(r, fit_end=FIT_END, p=5)
    assert fc.shape == r.shape
    assert np.all(np.isfinite(fc[5:]))


def test_ar_abs_forecast_elevated_in_burst():
    r = _clustered_returns()
    fc = ar.ar_abs_forecast(r, fit_end=FIT_END, p=5)
    assert fc[BURST].mean() > fc[CALM].mean()
