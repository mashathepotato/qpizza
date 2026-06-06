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
from math import comb

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


def _binom_pmf(i, q):
    return np.array([comb(i, k) * q ** k * (1 - q) ** (i - k) for k in range(i + 1)])


def build_forecast(S0, K, r, sigma, T, M=52, n_paths=14, seed=7):
    """The risk-neutral price-prediction CONE the demo animates open (beat 1).

    Same CRR tree math as results/price_forecast.py, exported as JSON arrays over
    M weekly steps: quantile bands of S_t, the forward path E[S_t]=S0 e^{rt}, a few
    sample trajectories, and the terminal distribution at t=T (what the call is priced on).
    Pure numpy -- no QAE -- so it is fast and deterministic.
    """
    u, d, q = tree.crr_params(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    dt = T / M
    times = (np.arange(M + 1) * dt).tolist()

    qs = {"p05": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}
    bands = {name: [] for name in qs}
    for i in range(M + 1):
        cdf = np.cumsum(_binom_pmf(i, q))
        prices_w = np.array([S0 * u ** w * d ** (i - w) for w in range(i + 1)])
        for name, p in qs.items():
            w_idx = min(int(np.searchsorted(cdf, p)), i)
            bands[name].append(float(prices_w[w_idx]))

    exp_path = [float(S0 * np.exp(r * t)) for t in times]

    rng = np.random.default_rng(seed)
    ups = rng.random((n_paths, M)) < q
    paths = []
    for k in range(n_paths):
        s, row = S0, [float(S0)]
        for i in range(M):
            s *= u if ups[k, i] else d
            row.append(float(s))
        paths.append(row)

    pmf_T = _binom_pmf(M, q)
    prices_T = np.array([S0 * u ** w * d ** (M - w) for w in range(M + 1)])
    order = np.argsort(prices_T)
    terminal = dict(prices=[float(x) for x in prices_T[order]],
                    pmf=[float(x) for x in pmf_T[order]],
                    expected=float(np.sum(pmf_T * prices_T)))
    return dict(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                times=times, bands=bands, exp_path=exp_path, paths=paths,
                terminal=terminal)


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
    forecast = build_forecast(S0=S0, K=K, r=r, sigma=sigma, T=T_MATURITY)

    with open(os.path.join(out_dir, "prices.json"), "w") as fh:
        json.dump(prices, fh, indent=2)
    with open(os.path.join(out_dir, "convergence.json"), "w") as fh:
        json.dump(conv, fh, indent=2)
    with open(os.path.join(out_dir, "forecast.json"), "w") as fh:
        json.dump(forecast, fh, indent=2)
    hw_path = os.path.join(out_dir, "hardware.json")
    if not os.path.exists(hw_path):          # never clobber a real Q50 result
        with open(hw_path, "w") as fh:
            json.dump(hardware_placeholder(), fh, indent=2)
    return dict(prices=prices, convergence=conv, forecast=forecast)


if __name__ == "__main__":
    main()
    print("wrote prices.json + convergence.json + forecast.json to results/ "
          "(hardware.json only if it was absent)")
