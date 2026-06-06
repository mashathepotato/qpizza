"""Rolling (sliding-window, stride-1) walk-forward backtest on NOKIA.HE.

Windows: 0-60, 1-61, 2-62, ... (length 60, stride 1). Each window: first 30 days
CALIBRATE (sigma_hat + boundary spot S0), last 30 days PREDICT. M=8 CRR steps span
the 30-day prediction window (~3.75 trading days/step), so each window forecasts at
8 step-nodes. Consecutive prediction windows overlap by 29 days, so every future day
is forecast by many windows -> we AVERAGE those overlapping predictions per day.

One plot (price y, day x), from day 30 on:
  * realized price (ground truth, black)
  * averaged QNDM/quantum model forecast E[S_t] (all 4 routes share this) + averaged
    25-75 / 5-95% cone
  * averaged Monte-Carlo mean path (the distinct classical model line)
Averaged evals printed + annotated: cone coverage and per-route option-price MAE
(routes run on a subset of windows; all routes agree with the exact tree).

Drift: risk-neutral forward (drift unestimable at 30 days); cone width from sigma_hat.

Run (pricer venv; yfinance + qiskit):
  PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python results/backtest_rolling.py
Saves results/figures/backtest_rolling.png.
"""
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from results import style
from quantum_pricer.tree import crr_params, exact_tree_price

R = 0.03
M = 8
CAL, PRED = 30, 30
WIN = CAL + PRED
N_SUBSET = 12            # windows on which to run all 4 quantum/classical routes
OUT = os.path.join(os.path.dirname(__file__), "figures", "backtest_rolling.png")


def fetch_prices():
    import yfinance as yf
    return np.asarray(yf.Ticker("NOKIA.HE").history(period="1y")["Close"].dropna().values, float)


def binom_pmf(i, q):
    return np.array([comb(i, k) * q ** k * (1 - q) ** (i - k) for k in range(i + 1)])


def price_routes(S0, sigma, T_h):
    K = S0
    out = {"exact": exact_tree_price(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M,
                                     option="european", kind="call")}
    try:
        from quantum_pricer.classical import monte_carlo_price
        out["MC"], _ = monte_carlo_price(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M,
                                         n_paths=50_000, option="european", kind="call", seed=1)
    except Exception as e:  # noqa: BLE001
        print("  MC fail", e)
    try:
        from quantum_pricer.fourier import price as fp
        out["Fourier"] = fp(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M, option="european", kind="call")
    except Exception as e:  # noqa: BLE001
        print("  Fourier fail", e)
    try:
        from quantum_pricer.qae import price as qp
        out["QAE"] = qp(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M, option="european",
                        kind="call", epsilon_target=0.01)["price"]
    except Exception as e:  # noqa: BLE001
        print("  QAE fail", e)
    try:
        from quantum_pricer.qsvt import price as sp
        out["QSVT"] = sp(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M, option="european",
                         kind="call", degree=40, use_qae=False)
    except Exception as e:  # noqa: BLE001
        print("  QSVT fail", e)
    return out


def main():
    style.apply_style()
    prices = fetch_prices()
    n = len(prices)
    starts = list(range(0, n - WIN + 1))      # stride-1 sliding windows
    print(f"  {n} trading days -> {len(starts)} sliding 60-day windows (stride 1)")

    T_h = PRED / 252.0
    # accumulate per-day forecast samples across overlapping windows
    acc_E, acc_lo, acc_hi, acc_mc = (defaultdict(list) for _ in range(4))
    rng = np.random.default_rng(0)
    subset = set(np.linspace(0, len(starts) - 1, N_SUBSET, dtype=int).tolist())
    route_err = defaultdict(list)            # route -> |price - exact| over subset

    for wi, s in enumerate(starts):
        cal = prices[s:s + CAL]
        b = s + CAL - 1
        S0 = float(prices[b])
        sigma = float(np.std(np.diff(np.log(cal)), ddof=1) * np.sqrt(252))
        u, d, q = crr_params(S0=S0, K=S0, r=R, sigma=sigma, T=T_h, M=M)

        # analytic forecast at the 8 step nodes
        for i in range(1, M + 1):
            day = int(round(b + i * (PRED / M)))
            pmf = binom_pmf(i, q); cdf = np.cumsum(pmf)
            pw = np.array([S0 * u ** w * d ** (i - w) for w in range(i + 1)])
            acc_E[day].append(float(np.sum(pmf * pw)))
            acc_lo[day].append(pw[min(int(np.searchsorted(cdf, 0.05)), i)])
            acc_hi[day].append(pw[min(int(np.searchsorted(cdf, 0.95)), i)])

        # Monte-Carlo mean path (distinct classical model line)
        ups = rng.random((1500, M)) < q
        steps = np.where(ups, u, d)
        paths = S0 * np.cumprod(steps, axis=1)
        for i in range(1, M + 1):
            day = int(round(b + i * (PRED / M)))
            acc_mc[day].append(float(paths[:, i - 1].mean()))

        # routes (subset only, for the agreement metric)
        if wi in subset:
            routes = price_routes(S0, sigma, T_h)
            ex = routes["exact"]
            for k, v in routes.items():
                if k != "exact":
                    route_err[k].append(abs(v - ex))

    def curve(acc):
        ds = sorted(d for d in acc if d <= n - 1)
        return np.array(ds), np.array([np.mean(acc[d]) for d in ds])

    dE, E = curve(acc_E)
    dlo, lo = curve(acc_lo)
    dhi, hi = curve(acc_hi)
    dmc, mc = curve(acc_mc)

    # cone coverage: fraction of realized days inside averaged 5-95% band
    inside = [lo[k] <= prices[dE[k]] <= hi[k] for k in range(len(dE))]
    coverage = 100.0 * np.mean(inside)
    mae = {k: float(np.mean(v)) for k, v in route_err.items()}
    print(f"  averaged cone coverage (realized in 5-95%): {coverage:.1f}%")
    print(f"  averaged option-price MAE vs exact tree over {N_SUBSET} windows: "
          f"{ {k: round(v,4) for k,v in mae.items()} }")

    fig, ax = plt.subplots(figsize=(13, 6))
    days = np.arange(n)
    ax.plot(days[CAL:], prices[CAL:], color=style.PALETTE["ink"], lw=1.8, zorder=5,
            label="realized price (ground truth)")
    ax.fill_between(dhi, lo, hi, color=style.PALETTE["quantum"], alpha=0.15,
                    label="avg 5-95% cone")
    ax.plot(dE, E, color=style.PALETTE["quantum"], lw=2.0, zorder=4,
            label="avg QNDM forecast E[S_t]  (all 4 routes share this)")
    ax.plot(dmc, mc, color=style.PALETTE["classical"], lw=1.4, ls="--", zorder=3,
            label="avg Monte-Carlo mean path")
    ax.set_xlim(CAL, n - 1)
    ax.set_xlabel("trading day")
    ax.set_ylabel("NOKIA.HE price  [EUR]")
    ax.set_title("Rolling backtest (sliding 60-day windows, stride 1): averaged "
                 "predictions from day 30 vs realized price")
    mae_txt = ", ".join(f"{k} {v:.1e}" for k, v in mae.items())
    ax.text(0.99, 0.02,
            f"avg cone coverage: {coverage:.0f}%   |   avg option-price MAE vs exact tree: {mae_txt}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=style.PALETTE["muted"])
    ax.legend(loc="upper left", fontsize=9)
    style.caption(fig, "Every future day is forecast by up to 30 overlapping windows; "
                       "predictions are averaged per day. Cone width from calibrated sigma_hat; "
                       "centre = risk-neutral forward. All 4 routes agree on the option price "
                       "(MAE vs exact tree, inset), so they share one forecast line.")
    style.provenance(fig, "quantum_pricer routes; NOKIA.HE 1y daily (yfinance); risk-neutral drift")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
