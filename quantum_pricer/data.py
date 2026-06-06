"""Fetch real Nokia (NOKIA.HE) market parameters: spot S0 and realized volatility
sigma from daily log-returns. Offline fallback is explicitly labelled synthetic."""
import numpy as np

DEFAULT_R = 0.03  # EUR risk-free proxy; override per scenario
_SYNTHETIC = dict(S0=4.20, sigma=0.30, r=DEFAULT_R)  # plausible NOKIA.HE stand-in


def annualized_vol(daily_log_returns, periods_per_year=252):
    return float(np.std(daily_log_returns, ddof=1) * np.sqrt(periods_per_year))


def nokia_params(ticker="NOKIA.HE", lookback="1y", r=DEFAULT_R, allow_network=True):
    """Return (params, meta). params = {S0, sigma, r}. meta records provenance.
    Falls back to a LABELLED synthetic stand-in if the network/yfinance is unavailable."""
    if allow_network:
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period=lookback)["Close"].dropna()
            if len(hist) > 30:
                logret = np.diff(np.log(hist.values))
                sigma = annualized_vol(logret)
                S0 = float(hist.values[-1])
                meta = dict(source="yfinance", ticker=ticker, lookback=lookback,
                            n_obs=len(hist), start=str(hist.index[0].date()),
                            end=str(hist.index[-1].date()))
                return dict(S0=S0, sigma=sigma, r=r), meta
        except Exception as exc:  # offline / rate-limited / delisted
            meta = dict(source="synthetic", reason=str(exc))
            return dict(**_SYNTHETIC), meta
    return dict(**_SYNTHETIC), dict(source="synthetic", reason="network disabled")


def _synthetic_series(n, r, seed):
    """Labelled GBM stand-in path from the synthetic params (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    S0, sigma = _SYNTHETIC["S0"], _SYNTHETIC["sigma"]
    dt = 1.0 / 252
    shocks = rng.normal((r - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), size=n - 1)
    closes = [S0]
    for s in shocks:
        closes.append(closes[-1] * float(np.exp(s)))
    closes = [float(c) for c in closes]
    logret = np.diff(np.log(closes))
    return dict(dates=["d%03d" % i for i in range(n)], closes=closes,
                S0=closes[-1], sigma=annualized_vol(logret), r=r)


def nokia_price_series(ticker="NOKIA.HE", lookback="1y", r=DEFAULT_R,
                       allow_network=True, n_synth=252, seed=0):
    """Return (series, meta). series = {dates, closes, S0, sigma, r} — the real daily
    closing path for act 1. Falls back to a LABELLED synthetic GBM path offline."""
    if allow_network:
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period=lookback)["Close"].dropna()
            if len(hist) > 30:
                closes = [float(x) for x in hist.values]
                sigma = annualized_vol(np.diff(np.log(hist.values)))
                series = dict(dates=[str(d.date()) for d in hist.index], closes=closes,
                              S0=closes[-1], sigma=sigma, r=r)
                meta = dict(source="yfinance", ticker=ticker, lookback=lookback,
                            n_obs=len(closes), start=str(hist.index[0].date()),
                            end=str(hist.index[-1].date()))
                return series, meta
        except Exception as exc:  # offline / rate-limited / delisted
            return _synthetic_series(n_synth, r, seed), dict(source="synthetic", reason=str(exc))
    return _synthetic_series(n_synth, r, seed), dict(source="synthetic", reason="network disabled")
