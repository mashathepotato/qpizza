"""FULL verification: price an ATM call on EVERY sliding window with ALL routes
(exact tree, MC, Fourier, QAE, QSVT) -- real circuit executions, no analytic
shortcut for the routes. Prints per-window QAE error (unrounded), saves a full
CSV, and a per-route option-price time-series figure (every point a genuine run).

Run (pricer venv):
  PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python results/verify_backtest.py
"""
import os
import sys
import csv
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from results import style
from quantum_pricer.tree import crr_params, exact_tree_price
from quantum_pricer.classical import monte_carlo_price
from quantum_pricer.fourier import price as fourier_price
from quantum_pricer.qae import price as qae_price
from quantum_pricer.qsvt import price as qsvt_price

R, M, CAL, PRED = 0.03, 8, 30, 30
WIN = CAL + PRED
CSV_OUT = os.path.join(os.path.dirname(__file__), "verify_backtest.csv")
FIG_OUT = os.path.join(os.path.dirname(__file__), "figures", "backtest_routes_timeseries.png")


def fetch():
    """Return calibration-window closes from pinned CSV (2024-06-05 to 2025-06-05).
    Strictly look-ahead-free: no data after asof enters the backtest."""
    import csv as _csv
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "quantum_pricer", "market_data", "NOKIA_HE.csv")
    rows = [r for r in _csv.DictReader(open(path))
            if "2024-06-05" <= r["date"] <= "2025-06-05"]
    return np.asarray([float(r["close"]) for r in rows], float)


def main():
    style.apply_style()
    prices = fetch()
    n = len(prices)
    starts = list(range(0, n - WIN + 1))
    T_h = PRED / 252.0
    print(f"  {n} days -> {len(starts)} windows; pricing ALL routes on EVERY window")

    rows = []
    t0 = time.time()
    for wi, s in enumerate(starts):
        cal = prices[s:s + CAL]
        b = s + CAL - 1
        S0 = float(prices[b]); K = S0
        sigma = float(np.std(np.diff(np.log(cal)), ddof=1) * np.sqrt(252))

        ex = exact_tree_price(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M,
                              option="european", kind="call")
        mc, _ = monte_carlo_price(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M,
                                  n_paths=50_000, option="european", kind="call", seed=1)
        fo = fourier_price(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M,
                           option="european", kind="call")
        qa = qae_price(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M,
                       option="european", kind="call", epsilon_target=0.01)
        qs = qsvt_price(S0=S0, K=K, r=R, sigma=sigma, T=T_h, M=M,
                        option="european", kind="call", degree=40, use_qae=False)
        qa_price, qa_q = qa["price"], qa["num_oracle_queries"]

        rows.append(dict(window=wi, day=b, S0=S0, sigma=sigma, exact=ex,
                         MC=mc, Fourier=fo, QAE=qa_price, QSVT=qs, qae_queries=qa_q,
                         qae_err=qa_price - ex))
        # per-window QAE error, UNROUNDED
        print(f"  w{wi:>3} day{b:>3} S0={S0:6.2f} sig={sigma:5.3f} | exact={ex:.6f} "
              f"QAE={qa_price:.6f} qae_err={qa_price - ex:+.3e} q={qa_q} "
              f"| MC_err={mc - ex:+.3e} Fou_err={fo - ex:+.3e} QSVT_err={qs - ex:+.3e}",
              flush=True)

    dt = time.time() - t0
    print(f"  priced {len(rows)} windows x 4 routes in {dt:.1f}s")

    # write full CSV
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("[saved]", CSV_OUT)

    # summary stats per route (abs error vs exact)
    arr = {k: np.array([r[k] for r in rows]) for k in
           ("day", "exact", "MC", "Fourier", "QAE", "QSVT")}
    print("\n  === per-route error vs exact tree over ALL windows ===")
    for k in ("MC", "Fourier", "QAE", "QSVT"):
        e = np.abs(arr[k] - arr["exact"])
        print(f"    {k:8s}  mean|err|={e.mean():.3e}  max|err|={e.max():.3e}  "
              f"mean rel={np.mean(e / arr['exact']):.2%}")

    # ---- figure: option price per route (top) + abs error (bottom), real runs ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                   gridspec_kw={"height_ratios": [1.6, 1]})
    cols = {"exact": style.PALETTE["ink"], "MC": style.PALETTE["classical"],
            "Fourier": style.PALETTE["muted"], "QAE": style.PALETTE["quantum"],
            "QSVT": style.PALETTE["accent"]}
    for k in ("exact", "MC", "Fourier", "QAE", "QSVT"):
        ax1.plot(arr["day"], arr[k], color=cols[k], lw=2.0 if k == "exact" else 1.2,
                 ls="-" if k == "exact" else "--", label=k, alpha=0.9)
    ax1.set_ylabel("ATM call price")
    ax1.set_title("Verification: ATM call price per route on every sliding window "
                  "(real runs) vs exact tree")
    ax1.legend(ncol=5, fontsize=9, loc="upper left")
    ax1.set_xlabel("boundary trading day")

    for k in ("MC", "Fourier", "QAE", "QSVT"):
        e = np.abs(arr[k] - arr["exact"])
        ax2.semilogy(arr["day"], np.maximum(e, 1e-12), color=cols[k], lw=1.1, label=k)
    ax2.set_ylabel("|price - exact|  (log)")
    ax2.set_xlabel("boundary trading day")
    ax2.set_title("Absolute error vs exact tree per route (every window, real runs)")
    ax2.legend(ncol=4, fontsize=8.5)

    fig.suptitle("Full backtest verification - all 4 routes priced on all "
                 f"{len(rows)} windows (M=8)", fontsize=13, fontweight="bold")
    style.caption(fig, "Every point is a genuine model execution (no analytic shortcut). "
                       "Fourier is statevector-exact (~1e-8); QAE ~1e-5 (eps target); "
                       "QSVT ~1-3% (straddle polynomial floor, degree-40 design constant); "
                       "MC ~1e-3 (sampling noise). Pinned CSV: 2024-06-05→2025-06-05.")
    style.provenance(fig, "quantum_pricer routes on NOKIA.HE pinned CSV; full sliding-window sweep")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT)
    plt.close(fig)
    print("[saved]", FIG_OUT)


if __name__ == "__main__":
    main()
