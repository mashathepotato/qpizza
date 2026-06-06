"""Side-by-side rolling price-prediction backtest: risk-neutral drift (r) vs
real-world drift (mu_hat estimated per window) on NOKIA.HE.

Same sliding-window setup as backtest_rolling.py (60-day windows, stride 1, 30
calibrate / 30 predict, M=8). The underlying-price forecast is route-independent,
so this is analytic (tree math) -- no quantum circuits needed.

Figure: 2x2 grid. Columns = {risk-neutral r, real-world mu_hat}; rows = {price
prediction vs realized, forecast error over time}.

Run:
  PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python results/backtest_drift_compare.py
Saves results/figures/backtest_drift_compare.png.
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
from quantum_pricer.tree import crr_params

R, M, CAL, PRED = 0.03, 8, 30, 30
WIN = CAL + PRED
OUT = os.path.join(os.path.dirname(__file__), "figures", "backtest_drift_compare.png")


def fetch():
    """Return calibration-window closes from pinned CSV (2024-06-05 to 2025-06-05).
    Strictly look-ahead-free: no data after asof enters the backtest."""
    import csv as _csv
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "quantum_pricer", "market_data", "NOKIA_HE.csv")
    rows = [r for r in _csv.DictReader(open(path))
            if "2024-06-05" <= r["date"] <= "2025-06-05"]
    return np.asarray([float(r["close"]) for r in rows], float)


def binom_pmf(i, q):
    return np.array([comb(i, k) * q ** k * (1 - q) ** (i - k) for k in range(i + 1)])


def run_mode(prices, realworld):
    n = len(prices)
    T_h = PRED / 252.0
    accE, accLo, accHi, accMc = (defaultdict(list) for _ in range(4))
    rng = np.random.default_rng(0)
    for s in range(0, n - WIN + 1):
        cal = prices[s:s + CAL]
        b = s + CAL - 1
        S0 = float(prices[b])
        logret = np.diff(np.log(cal))
        sigma = float(np.std(logret, ddof=1) * np.sqrt(252))
        u, d, q = crr_params(S0=S0, K=S0, r=R, sigma=sigma, T=T_h, M=M)
        if realworld:
            mu = float(np.mean(logret) * 252)
            dt = T_h / M
            q = float(np.clip((np.exp(mu * dt) - d) / (u - d), 1e-6, 1 - 1e-6))
        for i in range(1, M + 1):
            day = int(round(b + i * (PRED / M)))
            pmf = binom_pmf(i, q); cdf = np.cumsum(pmf)
            pw = np.array([S0 * u ** w * d ** (i - w) for w in range(i + 1)])
            accE[day].append(float(np.sum(pmf * pw)))
            accLo[day].append(pw[min(int(np.searchsorted(cdf, 0.05)), i)])
            accHi[day].append(pw[min(int(np.searchsorted(cdf, 0.95)), i)])
        ups = rng.random((1500, M)) < q
        paths = S0 * np.cumprod(np.where(ups, u, d), axis=1)
        for i in range(1, M + 1):
            day = int(round(b + i * (PRED / M)))
            accMc[day].append(float(paths[:, i - 1].mean()))

    def curve(acc):
        ds = sorted(x for x in acc if x <= n - 1)
        return np.array(ds), np.array([np.mean(acc[x]) for x in ds])

    dE, E = curve(accE)
    _, lo = curve(accLo)
    _, hi = curve(accHi)
    dmc, mc = curve(accMc)
    realized = np.array([prices[x] for x in dE])
    err = E - realized
    coverage = 100.0 * np.mean([lo[k] <= realized[k] <= hi[k] for k in range(len(dE))])
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return dict(dE=dE, E=E, lo=lo, hi=hi, dmc=dmc, mc=mc, realized=realized,
                err=err, coverage=coverage, mae=mae, rmse=rmse)


def main():
    style.apply_style()
    prices = fetch()
    n = len(prices)
    rn = run_mode(prices, realworld=False)
    rw = run_mode(prices, realworld=True)
    print(f"  risk-neutral: MAE={rn['mae']:.3f} RMSE={rn['rmse']:.3f} coverage={rn['coverage']:.0f}%")
    print(f"  real-world  : MAE={rw['mae']:.3f} RMSE={rw['rmse']:.3f} coverage={rw['coverage']:.0f}%")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    days = np.arange(n)
    pmax = max(prices[CAL:].max(), rn["hi"].max(), rw["hi"].max()) * 1.05
    emax = max(np.abs(rn["err"]).max(), np.abs(rw["err"]).max(),
               rn["hi"][0] - rn["lo"][0]) * 1.1

    for col, (mode, lab) in enumerate([(rn, "Risk-neutral drift (r = 3%)  [Q-measure, NOT a forecast]"),
                                       (rw, "Real-world drift (mu_hat per window)  [P-measure, noisy]")]):
        axP, axE = axes[0][col], axes[1][col]
        axP.plot(days[CAL:], prices[CAL:], color=style.PALETTE["ink"], lw=1.7, zorder=5,
                 label="realized (ground truth)")
        axP.fill_between(mode["dE"], mode["lo"], mode["hi"], color=style.PALETTE["quantum"],
                         alpha=0.15, label="predicted 5-95%")
        axP.plot(mode["dE"], mode["E"], color=style.PALETTE["quantum"], lw=2.0,
                 label="predicted E[S_t]")
        axP.plot(mode["dmc"], mode["mc"], color=style.PALETTE["classical"], lw=1.2, ls="--",
                 label="MC mean")
        axP.set_title(lab, fontsize=11)
        axP.set_ylim(0, pmax)
        axP.set_ylabel("NOKIA.HE price [EUR]")
        axP.legend(loc="upper left", fontsize=8.5)
        axP.text(0.99, 0.03, f"price-forecast MAE={mode['mae']:.2f} EUR\ncone coverage {mode['coverage']:.0f}%",
                 transform=axP.transAxes, ha="right", va="bottom", fontsize=8.5,
                 color=style.PALETTE["muted"])

        half = (mode["hi"] - mode["lo"]) / 2.0
        axE.fill_between(mode["dE"], -half, half, color=style.PALETTE["muted"], alpha=0.20,
                         label="+/- cone half-width")
        axE.axhline(0, color=style.PALETTE["ink"], lw=1.0)
        axE.plot(mode["dE"], mode["err"], color=style.PALETTE["accent"], lw=1.5,
                 label="forecast error = predicted - realized")
        axE.set_ylim(-emax, emax)
        axE.set_xlim(CAL, n - 1)
        axE.set_xlabel("trading day")
        axE.set_ylabel("price error [EUR]")
        axE.set_title("Forecast error over time", fontsize=10)
        axE.legend(loc="lower left", fontsize=8)

    fig.suptitle("Q-measure vs P-measure pricing cone - NOKIA.HE "
                 "(pinned 2024-06-05→2025-06-05, M=8)", fontsize=13, fontweight="bold")
    style.caption(fig, "Left (risk-neutral Q, the pricing measure): E_Q[S_t]=S0·e^{rt} ≈ S0. "
                       "Not a forecast — drift r is imposed by no-arbitrage. Right (real-world P, "
                       "mu_hat per window): better tracks trends but mu_hat is noisy and overshoots "
                       "on reversals. Neither is a stock-price prediction; both are correct pricing "
                       "measures for their respective purposes. Cone error is a property of the "
                       "measure, not of the quantum algorithms.")
    style.provenance(fig, "quantum_pricer/tree.py (CRR); NOKIA.HE pinned CSV 2024-06-05→2025-06-05")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
