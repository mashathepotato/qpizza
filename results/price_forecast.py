"""Forecast plot: starting from today's NOKIA.HE spot S0, show the model's
risk-neutral prediction of the price over the one-year option horizon.

Faithful to the pricer's CRR binomial tree (quantum_pricer/tree.py):
  Left  - prediction fan over time: 5-95% and 25-75% risk-neutral quantile bands
          of S_t, the expected (forward) path E[S_t]=S0 e^{rt}, a few sampled
          paths, the start S0, and the strike K.
  Right - terminal price distribution at t=T (what the option is priced on), with
          the in-the-money region (S_T > K) shaded and E[S_T] marked.

Needs only numpy (tree math is pure Python). Run:
  PYTHONPATH=<repo-root> quantum_investor/.venv/bin/python results/price_forecast.py
Saves results/figures/price_forecast.png.
"""
import os
import sys
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from results import style
from quantum_pricer.tree import crr_params

# NOKIA.HE ATM call - fixed for reproducibility (latest close / realized vol), no network.
S0, K, r, sigma, T = 13.09, 13.09, 0.03, 0.479, 1.0
M = 52                       # weekly steps over the 1-year horizon (viz resolution)
N_PATHS = 14                 # illustrative sampled trajectories
OUT = os.path.join(os.path.dirname(__file__), "figures", "price_forecast.png")


def binom_pmf(i, q):
    w = np.arange(i + 1)
    return np.array([comb(i, int(k)) * q ** k * (1 - q) ** (i - k) for k in w])


def main():
    style.apply_style()
    u, d, q = crr_params(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    dt = T / M
    times = np.arange(M + 1) * dt

    # Risk-neutral quantile bands of S_t (price is monotincreasing in up-count w).
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    bands = {p: np.empty(M + 1) for p in qs}
    for i in range(M + 1):
        pmf = binom_pmf(i, q)
        cdf = np.cumsum(pmf)
        prices_w = np.array([S0 * u ** w * d ** (i - w) for w in range(i + 1)])
        for p in qs:
            w_idx = int(np.searchsorted(cdf, p))
            bands[p][i] = prices_w[min(w_idx, i)]

    exp_path = S0 * np.exp(r * times)            # E[S_t] = S0 e^{rt} (forward)

    # A few sampled risk-neutral trajectories (up with prob q each step).
    rng = np.random.default_rng(7)
    paths = np.empty((N_PATHS, M + 1)); paths[:, 0] = S0
    ups = rng.random((N_PATHS, M)) < q
    for k in range(N_PATHS):
        s = S0
        for i in range(M):
            s *= u if ups[k, i] else d
            paths[k, i + 1] = s

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [2.1, 1]})

    # ---- Left: prediction fan over the year ----
    axL.fill_between(times, bands[0.05], bands[0.95], color=style.PALETTE["quantum"],
                     alpha=0.15, label="5-95% range")
    axL.fill_between(times, bands[0.25], bands[0.75], color=style.PALETTE["quantum"],
                     alpha=0.30, label="25-75% range")
    for k in range(N_PATHS):
        axL.plot(times, paths[k], color=style.PALETTE["muted"], lw=0.7, alpha=0.5)
    axL.plot(times, exp_path, color=style.PALETTE["quantum"], lw=2.2,
             label=r"expected (forward) $E[S_t]=S_0 e^{rt}$")
    axL.axhline(K, ls="--", color=style.PALETTE["classical"], lw=1.4,
                label=f"strike K = {K:.2f}")
    axL.scatter([0], [S0], s=90, color=style.PALETTE["accent"], zorder=5,
                edgecolor=style.PALETTE["ink"])
    axL.annotate(f"start: today's spot\n$S_0$ = {S0:.2f} EUR",
                 xy=(0, S0), xytext=(0.12, S0 * 1.32), fontsize=9,
                 color=style.PALETTE["ink"],
                 arrowprops=dict(arrowstyle="->", color=style.PALETTE["ink"]))
    axL.set_xlabel("time from today  t  [years]")
    axL.set_ylabel("NOKIA.HE price  [EUR]")
    axL.set_title(f"Risk-neutral price prediction over the 1-year horizon  (M={M} weekly steps)")
    axL.set_xlim(0, T)
    axL.legend(loc="upper left", fontsize=8.5)

    # ---- Right: terminal distribution at t = T ----
    pmf_T = binom_pmf(M, q)
    prices_T = np.array([S0 * u ** w * d ** (M - w) for w in range(M + 1)])
    order = np.argsort(prices_T)
    pT, wT = prices_T[order], pmf_T[order]
    axR.fill_betweenx(pT, 0, wT, where=(pT > K), color=style.PALETTE["accent"],
                      alpha=0.5, label="in-the-money ($S_T>K$)")
    axR.fill_betweenx(pT, 0, wT, where=(pT <= K), color=style.PALETTE["muted"],
                      alpha=0.5, label="out-of-the-money")
    axR.axhline(K, ls="--", color=style.PALETTE["classical"], lw=1.4)
    eST = float(np.sum(pmf_T * prices_T))
    axR.axhline(eST, ls="-", color=style.PALETTE["quantum"], lw=1.6,
                label=f"$E[S_T]$ = {eST:.2f}")
    axR.set_xlabel("risk-neutral probability")
    axR.set_ylabel("terminal price $S_T$  [EUR]")
    axR.set_title("Predicted price at T = 1y")
    axR.set_ylim(axL.get_ylim())
    axR.legend(loc="upper right", fontsize=8)

    fig.suptitle("Quantum option pricer - NOKIA.HE price forecast over the option horizon",
                 fontsize=13, fontweight="bold")
    style.caption(fig, f"Start from today's spot S0={S0:.2f} EUR; the CRR risk-neutral "
                       f"tree predicts the price distribution out to T={T:g}y. The "
                       f"terminal distribution (right) is what the call (K={K:.2f}) is priced on.")
    style.provenance(fig, "quantum_pricer/tree.py (CRR risk-neutral); NOKIA.HE S0/sigma")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)

    print(f"  start S0 = {S0:.2f} EUR (today's spot)   K = {K:.2f}   T = {T:g}y   M = {M}")
    print(f"  E[S_T]   = {eST:.4f} EUR  (= S0 e^(rT) = {S0 * np.exp(r * T):.4f})")
    print(f"  5-95% terminal range: [{pT[wT.cumsum() >= 0.05][0]:.2f}, "
          f"{pT[wT.cumsum() >= 0.95][0]:.2f}] EUR")
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
