"""Reproduce Stamatopoulos et al. (arXiv:1905.02666) Figure 11 convention on OUR
pricer: estimation error (y) vs number of samples / oracle queries M (x), log-log,
Amplitude Estimation (~1/M, slope -1) vs Monte Carlo (~1/sqrt(M), slope -1/2).

Run (needs the pricer venv with qiskit):
  PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python results/fig11_error_vs_queries.py

Saves results/figures/error_vs_queries.png. Standalone so it does not touch
quantum_pricer/benchmark.py (owned by concurrent work).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from results import style
from quantum_pricer.benchmark import error_vs_queries_rms

# Nokia (NOKIA.HE) ATM European call -- pinned asof 2025-06-05 (look-ahead-free).
from quantum_pricer.data import nokia_params as _np
_p, _ = _np(allow_network=False)
_S0 = _p["S0"]
PARAMS = dict(S0=_S0, K=round(_S0, 2), r=_p["r"], sigma=_p["sigma"], T=1.0, M=4)
OUT = os.path.join(os.path.dirname(__file__), "figures", "error_vs_queries.png")


def _fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0]) if len(xs) >= 2 else float("nan")


def _qsvt_floor_from_csv():
    """Read QSVT mean |error| from the already-computed verify_backtest.csv.
    No simulation needed — this is the polynomial-approximation floor measured
    across 191 real-circuit windows."""
    import csv as _csv
    path = os.path.join(os.path.dirname(__file__), "verify_backtest.csv")
    rows = list(_csv.DictReader(open(path)))
    errs = [abs(float(r["QSVT"]) - float(r["exact"])) for r in rows]
    return float(sum(errs) / len(errs))


def main():
    style.apply_style()
    rows = error_vs_queries_rms(**PARAMS, seeds=8)

    def pts(method, seeds_gt0=None):
        out = []
        for r in rows:
            if r["method"] != method:
                continue
            if seeds_gt0 is True and not r["n_seeds"] > 0:
                continue
            if seeds_gt0 is False and not r["n_seeds"] == 0:
                continue
            if r["budget_x"] > 0 and np.isfinite(r["rms_error"]) and r["rms_error"] > 0:
                out.append((r["budget_x"], r["rms_error"]))
        return tuple(zip(*sorted(out))) if out else ([], [])

    mc_x, mc_y = pts("classical_mc")
    qae_emp_x, qae_emp_y = pts("qae", seeds_gt0=True)
    qae_th_x, qae_th_y = pts("qae", seeds_gt0=False)  # theory line iff saturated

    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    # Monte Carlo: empirical points + dashed linear fit (Fig 11 convention)
    if mc_x:
        s = _fit(mc_x, mc_y)
        ax.loglog(mc_x, mc_y, "s", color=style.PALETTE["classical"], markersize=7,
                  label=f"Monte Carlo (slope {s:+.2f}, ideal -0.50)")
        xx = np.array([min(mc_x), max(mc_x)], float)
        c = np.polyfit(np.log(mc_x), np.log(mc_y), 1)
        ax.loglog(xx, np.exp(np.polyval(c, np.log(xx))), "--",
                  color=style.PALETTE["classical"], linewidth=1.3, alpha=0.8)

    # Amplitude Estimation: empirical points + fit; overlay theory line if saturated
    if qae_emp_x:
        s = _fit(qae_emp_x, qae_emp_y)
        ax.loglog(qae_emp_x, qae_emp_y, "o", color=style.PALETTE["quantum"], markersize=8,
                  label=f"Amplitude Estimation (slope {s:+.2f}, ideal -1.00)")
        if len(qae_emp_x) >= 2:
            xx = np.array([min(qae_emp_x), max(qae_emp_x)], float)
            c = np.polyfit(np.log(qae_emp_x), np.log(qae_emp_y), 1)
            ax.loglog(xx, np.exp(np.polyval(c, np.log(xx))), "-",
                      color=style.PALETTE["quantum"], linewidth=1.3, alpha=0.8)
    if qae_th_x:
        ax.loglog(qae_th_x, qae_th_y, ":", color=style.PALETTE["accent"], linewidth=1.6,
                  label=r"AE ideal $\varepsilon=\pi/2M$ (slope -1.00)")

    # QSVT: polynomial-approximation floor — not query-scalable.
    # No simulation needed; value read from verify_backtest.csv (191 real windows).
    qsvt_floor = _qsvt_floor_from_csv()

    # Reference slope guides (-1 and -1/2) anchored to the data span
    all_x = list(mc_x) + list(qae_emp_x) + list(qae_th_x)
    if all_x:
        x0, x1 = min(all_x), max(all_x)
        ymax = max(list(mc_y) + list(qae_emp_y) + list(qae_th_y))
        for slope, lab in ((-1.0, "1/M"), (-0.5, r"1/$\sqrt{M}$")):
            xref = np.array([x0, x1], float)
            yref = ymax * (xref / x0) ** slope
            ax.loglog(xref, yref, color=style.PALETTE["muted"], linewidth=0.8,
                      alpha=0.5, zorder=0)
        # QSVT horizontal floor line
        ax.axhline(qsvt_floor, color=style.PALETTE["accent"], linewidth=1.4,
                   linestyle="--", alpha=0.85, zorder=2,
                   label=f"QSVT poly-approx floor  {qsvt_floor:.2e}  (degree 40, not query-scalable)")
        ax.annotate("QSVT floor\n(poly-approx, deg 40)",
                    xy=(x1 * 0.6, qsvt_floor), xytext=(x1 * 0.15, qsvt_floor * 2.5),
                    fontsize=8, color=style.PALETTE["accent"],
                    arrowprops=dict(arrowstyle="->", color=style.PALETTE["accent"], lw=0.9))

    ax.set_xlabel("Number of samples / oracle queries  M")
    ax.set_ylabel(r"Estimation error  $\varepsilon$")
    ax.set_title("Estimation error vs queries (cf. Stamatopoulos et al. 2020, Fig. 11)")
    ax.legend(loc="lower left", fontsize=8.5)
    style.provenance(fig, "quantum_pricer (QNDM+QAE+QSVT) on NOKIA.HE ATM call; "
                          "cf. arXiv:1905.02666 Fig. 11")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)

    print("MC   slope:", round(_fit(mc_x, mc_y), 3) if mc_x else "n/a")
    if qae_emp_x:
        print("QAE  slope:", round(_fit(qae_emp_x, qae_emp_y), 3),
              "(empirical, finite-shot)")
    if qae_th_x:
        print("QAE  saturated -> theory line pi/2M overlaid")
    print(f"QSVT floor: {qsvt_floor:.3e}  (from verify_backtest.csv, 191 windows)")
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
