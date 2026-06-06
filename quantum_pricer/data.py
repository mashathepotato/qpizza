"""Real Nokia (NOKIA.HE) market parameters with a PINNED as-of date t0 and a
LOOK-AHEAD-FREE calibration protocol:

  * calibration uses ONLY closes in [t0 - lookback, t0]  ->  S0, sigma
  * the option's life (t0, t0 + T] is reserved for OUT-OF-SAMPLE evaluation
    (realized vol, realized payoff) and never touches calibration.

History is cached to a CSV committed to the repo (market_data/) so every rebuild
is byte-reproducible offline; yfinance is hit only when the cache is absent.
The offline no-cache fallback is explicitly labelled synthetic.
"""
import os
from datetime import date, timedelta

import numpy as np

ASOF = "2025-06-05"   # t0: pinned pricing date — nothing after this enters calibration
# 12M Euribor stood near 2.1% in June 2025; this is a FIXED PROXY chosen as of t0,
# not a calibrated quantity (recorded as such in meta["r_source"]).
DEFAULT_R = 0.021
_SYNTHETIC = dict(S0=4.20, sigma=0.30, r=DEFAULT_R)  # plausible NOKIA.HE stand-in
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_data")


def annualized_vol(daily_log_returns, periods_per_year=252):
    return float(np.std(daily_log_returns, ddof=1) * np.sqrt(periods_per_year))


def _default_cache_path(ticker):
    return os.path.join(_CACHE_DIR, ticker.replace(".", "_") + ".csv")


def _read_cache(path):
    dates, closes = [], []
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            d, c = line.strip().split(",")
            dates.append(d)
            closes.append(float(c))
    return dates, np.asarray(closes, dtype=float)


def _write_cache(path, dates, closes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("date,close\n")
        for d, c in zip(dates, closes):
            fh.write(f"{d},{c:.6f}\n")


def load_history(ticker="NOKIA.HE", allow_network=True, cache_path=None,
                 start="2024-05-01", end=None):
    """Return (dates, closes) covering calibration + evaluation windows.

    Reads the committed CSV cache when present (deterministic, offline);
    otherwise fetches via yfinance once and writes the cache. Raises if neither
    is possible (callers fall back to the labelled synthetic stand-in).
    """
    path = cache_path or _default_cache_path(ticker)
    if os.path.exists(path):
        return _read_cache(path)
    if not allow_network:
        raise RuntimeError("no cached history at %s and network disabled" % path)
    import yfinance as yf
    hist = yf.Ticker(ticker).history(start=start, end=end)["Close"].dropna()
    if len(hist) <= 30:
        raise RuntimeError("yfinance returned too little history (%d rows)" % len(hist))
    dates = [str(ix.date()) for ix in hist.index]
    closes = hist.values.astype(float)
    _write_cache(path, dates, closes)
    return dates, closes


def _window(dates, closes, lo, hi):
    """Slice (dates, closes) to lo <= date <= hi (ISO strings sort correctly)."""
    idx = [i for i, d in enumerate(dates) if lo <= d <= hi]
    return [dates[i] for i in idx], closes[idx]


def nokia_params(ticker="NOKIA.HE", asof=ASOF, lookback_days=365, r=DEFAULT_R,
                 allow_network=True, cache_path=None):
    """Calibrated (params, meta) using ONLY data on or before `asof`.

    params = {S0, sigma, r}: S0 = last close <= asof; sigma = annualized realized
    vol of log-returns in [asof - lookback_days, asof]; r = fixed proxy (NOT
    calibrated — see meta["r_source"]). meta records the exact calibration window
    so look-ahead-freeness is checkable (calib_end <= asof always).
    Falls back to a LABELLED synthetic stand-in when no cache and no network.
    """
    try:
        dates, closes = load_history(ticker, allow_network=allow_network,
                                     cache_path=cache_path)
    except Exception as exc:  # offline with no cache / rate-limited / delisted
        return dict(_SYNTHETIC, r=r), dict(source="synthetic", reason=str(exc))
    lo = str(date.fromisoformat(asof) - timedelta(days=lookback_days))
    cal_dates, cal_closes = _window(dates, closes, lo, asof)
    if len(cal_dates) <= 30:
        return dict(_SYNTHETIC, r=r), dict(
            source="synthetic", reason="calibration window has %d obs" % len(cal_dates))
    logret = np.diff(np.log(cal_closes))
    meta = dict(source="yfinance-cache", ticker=ticker, asof=asof,
                calib_start=cal_dates[0], calib_end=cal_dates[-1],
                n_obs=len(cal_dates), lookback_days=lookback_days,
                r_source="12M Euribor proxy as of t0 (fixed, not calibrated)")
    return dict(S0=float(cal_closes[-1]), sigma=annualized_vol(logret), r=r), meta


def realized_outcome(ticker="NOKIA.HE", asof=ASOF, T=1.0, allow_network=True,
                     cache_path=None):
    """OUT-OF-SAMPLE outcome over the option's life (asof, asof + T] — strictly
    disjoint from the calibration window.

    Returns dict(S_T, S_T_date, realized_vol, n_obs, window_start, window_end):
    S_T = last available close <= asof + T*365d; realized_vol = annualized vol of
    log-returns inside the future window. Raises when the window has too little
    data (e.g. asof + T is still in the future) — callers must skip, not fake.
    """
    dates, closes = load_history(ticker, allow_network=allow_network,
                                 cache_path=cache_path)
    t0 = date.fromisoformat(asof)
    hi = str(t0 + timedelta(days=round(T * 365)))
    fut_dates, fut_closes = _window(dates, closes, _next_day(asof), hi)
    if len(fut_dates) <= 30:
        raise RuntimeError("future window (%s, %s] has only %d obs"
                           % (asof, hi, len(fut_dates)))
    logret = np.diff(np.log(fut_closes))
    return dict(S_T=float(fut_closes[-1]), S_T_date=fut_dates[-1],
                realized_vol=annualized_vol(logret), n_obs=len(fut_dates),
                window_start=fut_dates[0], window_end=hi)


def _next_day(iso):
    return str(date.fromisoformat(iso) + timedelta(days=1))
