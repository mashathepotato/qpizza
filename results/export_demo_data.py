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
T_MATURITY = 1.0


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


def build_prices(series_meta, K, T, M_paths=3):
    """Real price path (act 1) + the exact 2^M CRR tree (act 2 superposition)."""
    series, meta = series_meta
    S0, sigma, r = series["S0"], series["sigma"], series["r"]
    vals = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_paths)
    probs = tree.path_probabilities(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_paths)
    return dict(ticker=meta.get("ticker", "NOKIA.HE"), currency="EUR",
                source=meta.get("source", "synthetic"),
                window="%s..%s" % (series["dates"][0], series["dates"][-1]),
                dates=series["dates"], closes=series["closes"],
                S0=S0, sigma=sigma, r=r, K=K, T=T,
                tree=dict(M=M_paths,
                          terminal_values=[float(v) for v in vals],
                          path_probs=[float(p) for p in probs]))


def hardware_placeholder():
    """Q50 slot a teammate overwrites with real counts. Frontend shows 'pending'."""
    return dict(status="pending", backend="Q50", route="fourier",
                price=None, abs_error=None, shots=None, note="awaiting teammate Q50 run")


def main(out_dir=None, allow_network=True, M_paths=3, M_conv=5, seeds=6):
    out_dir = out_dir or _HERE
    os.makedirs(out_dir, exist_ok=True)
    series, meta = nokia_price_series(allow_network=allow_network)
    S0, sigma, r = series["S0"], series["sigma"], series["r"]
    K = round(S0, 2)

    prices = build_prices((series, meta), K=K, T=T_MATURITY, M_paths=M_paths)
    conv = build_convergence(S0=S0, K=K, r=r, sigma=sigma, T=T_MATURITY, M=M_conv, seeds=seeds)

    with open(os.path.join(out_dir, "prices.json"), "w") as fh:
        json.dump(prices, fh, indent=2)
    with open(os.path.join(out_dir, "convergence.json"), "w") as fh:
        json.dump(conv, fh, indent=2)
    hw_path = os.path.join(out_dir, "hardware.json")
    if not os.path.exists(hw_path):          # never clobber a real Q50 result
        with open(hw_path, "w") as fh:
            json.dump(hardware_placeholder(), fh, indent=2)
    return dict(prices=prices, convergence=conv)


if __name__ == "__main__":
    main()
    print("wrote prices.json + convergence.json to results/ (hardware.json only if it was absent)")
