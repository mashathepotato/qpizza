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


def build_convergence(S0, K, r, sigma, T, M=5, seeds=6):
    """error_vs_queries_rms -> {ground_truth, M, axis_label, series, slopes, notes}."""
    rows = benchmark.error_vs_queries_rms(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                          seeds=seeds)
    gt = tree.exact_tree_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    series, notes, slopes = {}, {}, {}
    for method in ("classical_mc", "qae"):
        pts = [dict(x=float(rw["budget_x"]), y=float(rw["rms_error"]))
               for rw in rows if rw["method"] == method and rw["budget_x"] > 0
               and np.isfinite(rw["rms_error"]) and rw["rms_error"] > 0]
        pts.sort(key=lambda p: p["x"])
        series[method] = pts
        notes[method] = sorted({rw.get("note", "") for rw in rows
                                if rw["method"] == method} - {""})
        if len(pts) >= 2:
            xs = np.log([p["x"] for p in pts])
            ys = np.log([p["y"] for p in pts])
            slopes[method] = float(np.polyfit(xs, ys, 1)[0])
        else:
            slopes[method] = None
    return dict(ground_truth=gt, M=M,
                axis_label="oracle queries / samples (resource spent, NOT wall-clock)",
                series=series, slopes=slopes, notes=notes)
