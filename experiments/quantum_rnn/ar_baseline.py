"""Autoregressive statistical baselines for the next-day-volatility task.

These are the principled classical models for volatility clustering -- the only
real signal in daily returns. Both are fit on the TRAINING returns only, then
their one-step-ahead forecasts are rolled across the full series using realized
(lagged) returns -- standard out-of-sample filtering, no look-ahead leakage. The
day-t+1 forecast is the anomaly score for the sequence ending day t.

  garch_forecast  -- GARCH(1,1) conditional-variance forecast (the headline model
                     for vol clustering), via the `arch` package.
  ar_abs_forecast -- AR(p) on |returns| (the plain ARIMA-flavored baseline), via
                     statsmodels.
"""
import numpy as np


def garch_forecast(returns, fit_end):
    """One-step-ahead conditional variance for every day; GARCH(1,1) params fit
    on returns[:fit_end], then filtered forward over the whole series with
    realized returns. h[d] uses information through day d-1 only."""
    from arch import arch_model

    r = np.asarray(returns, dtype=float) * 100.0          # arch likes percent-scale
    res = arch_model(r[:fit_end], vol="Garch", p=1, q=1,
                     mean="Constant", dist="normal").fit(disp="off")
    mu = res.params["mu"]
    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]

    eps = r - mu
    h = np.empty(len(r))
    h[0] = float(np.var(r[:fit_end]))
    for t in range(1, len(r)):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    return h


def ar_abs_forecast(returns, fit_end, p=5):
    """One-step-ahead AR(p) forecast of |return| for every day; params fit on the
    training |returns|, applied forward with realized lagged |returns|."""
    from statsmodels.tsa.ar_model import AutoReg

    a = np.abs(np.asarray(returns, dtype=float))
    res = AutoReg(a[:fit_end], lags=p, old_names=False).fit()
    const = res.params[0]
    phi = np.asarray(res.params[1:])                       # phi[k] multiplies lag k+1
    fc = np.full(len(a), np.nan)
    for t in range(p, len(a)):
        lags = a[t - p:t][::-1]                            # [lag1, lag2, ..., lagp]
        fc[t] = const + float(np.dot(phi, lags))
    return fc
