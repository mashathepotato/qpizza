"""Export the real numbers the demo_animation.html frontend renders.

Every moving element in the frontend traces to a value produced here:
  prices.json       -- the real Nokia price path (act 1) + the 2^M tree (act 2)
  convergence.json  -- seed-averaged error-vs-QUERIES descent (act 4): MC slope ~-1/2,
                       QAE slope ~-1; honesty notes preserved. Ground truth = exact tree.
  hardware.json     -- a 'pending' Q50 placeholder a teammate overwrites with real counts.
results/results.json (already produced by make_results.py) supplies the final prices/queries.

Run:  python -m results.export_demo_data
"""
import json
import os

import numpy as np

from quantum_pricer import benchmark, tree
from quantum_pricer.data import nokia_price_series

_HERE = os.path.dirname(os.path.abspath(__file__))

AXIS_LABEL_CONVERGENCE = "oracle queries / samples (resource spent, NOT wall-clock)"


def build_convergence(S0, K, r, sigma, T, M=5, seeds=6):
    """error_vs_queries_rms -> {ground_truth, M, axis_label, series, slopes, notes}."""
    rows = benchmark.error_vs_queries_rms(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                          seeds=seeds)
    # ground truth recomputed here (error_vs_queries_rms uses it internally but does not return it)
    gt = tree.exact_tree_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    series, notes, slopes = {}, {}, {}
    # only these two methods feed the frontend convergence plot; QSVT is intentionally excluded (it carries a query-independent approximation floor, not a descent)
    for method in ("classical_mc", "qae"):
        pts = [dict(x=float(row["budget_x"]), y=float(row["rms_error"]))
               for row in rows if row["method"] == method and row["budget_x"] > 0
               and np.isfinite(row["rms_error"]) and row["rms_error"] > 0]
        pts.sort(key=lambda p: p["x"])
        series[method] = pts
        notes[method] = sorted({row.get("note", "") for row in rows
                                if row["method"] == method} - {""})
        if len(pts) >= 2:
            xs = np.log([p["x"] for p in pts])
            ys = np.log([p["y"] for p in pts])
            slopes[method] = float(np.polyfit(xs, ys, 1)[0])
        else:
            slopes[method] = None
    return dict(ground_truth=gt, M=M,
                axis_label=AXIS_LABEL_CONVERGENCE,
                series=series, slopes=slopes, notes=notes)
