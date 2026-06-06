"""Single source of truth for dashboard content + figure source locations.

Each track declares where its figure is, the key-number table to show, the prose
interpretation, and provenance. build_dashboard.py reads ONLY this for content.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(*parts):
    return os.path.join(ROOT, *parts)


# The triage worktree is archived; its measured figure is cached in results/figures/.
_TRIAGE_FIG = _p("results", "figures", "qae_scaling.png")

TRACKS = [
    {
        "key": "cognition",
        "title": "The Madness of People Is Quantum",
        "claim": "Real question-order data obeys the parameter-free QQ-equality (q = -0.003).",
        "figure": _p("quantum_investor", "figure.png"),
        "regenerate": "cognition",          # build_dashboard re-runs this one
        "table": {
            "header": ["quantity", "value", "meaning"],
            "rows": [
                ["order effect |delta|", "0.095", "asking order changes the answer (real, large)"],
                ["QQ-equality q", "-0.003", "parameter-free quantum prediction = 0"],
                ["PNAS chi^2(1)", "0.01 (p=0.91)", "data is consistent with q = 0"],
                ["classical order effect", "0.000", "structurally impossible to produce"],
            ],
        },
        "prose": ("Humans answer differently depending on question order; a classical "
                  "(Bayesian) joint distribution is order-blind and cannot represent this. "
                  "Yet the same data satisfies the QQ-equality that quantum probability "
                  "predicts with no fitted parameters. Honest caveat: a single-qubit model "
                  "reproduces the structure, not the full joint fit of this survey."),
        "provenance": "Wang et al., PNAS 111:9431 (2014); Gallup 1997 Clinton/Gore, ~1000 adults.",
    },
    {
        "key": "pricer",
        "title": "Quantum Option Pricer - quadratic Monte-Carlo speedup",
        "claim": "QNDM+QAE reaches accuracy eps in O(1/eps) oracle queries vs MC's O(1/eps^2).",
        "figure": _p("results", "figures", "benchmark_scoreboard.png"),
        "extra_figures": [
            _p("results", "figures", "price_forecast.png"),
            _p("results", "figures", "backtest_rolling.png"),
            _p("results", "figures", "backtest_drift_compare.png"),
            _p("results", "figures", "backtest_routes_timeseries.png"),
            _p("results", "figures", "backtest_walkforward.png"),
            _p("results", "figures", "model_results.png"),
            _p("results", "figures", "error_vs_queries.png"),
            _p("quantum_pricer", "complexity.png"),
            _p("quantum_pricer", "speedup.png"),
            _p("quantum_pricer", "depth_crossover.png"),
        ],
        # Per-figure captions (keyed by basename) shown under each image.
        "captions": {
            "benchmark_scoreboard.png": ("Summary scoreboard (all numbers from real runs). Our "
                                         "QNDM-QAE matches classical and quantum-SOTA prices "
                                         "(error 4e-5 vs exact tree) at ~6x shallower circuit depth "
                                         "than the SOTA lognormal-loader baseline (16 vs 100 CZ), "
                                         "with the same O(1/eps) quantum query advantage over MC's "
                                         "O(1/eps^2). No wall-clock speedup over MC is claimed; the "
                                         "win is loading depth + asymptotic query scaling."),
            "price_forecast.png": ("What we price: from today's NOKIA.HE spot "
                                   "S0=13.09 EUR, the CRR risk-neutral tree predicts the "
                                   "price distribution over the 1-year horizon (E[S_T]=13.49, "
                                   "forward). The terminal distribution is what the call is priced on."),
            "backtest_rolling.png": ("PREDICTION TARGET = the underlying NOKIA.HE price 30 trading "
                                     "days ahead (NOT the option price). Sliding 60-day windows "
                                     "(stride 1), each calibrated on its first 30 days; every future "
                                     "day is forecast by up to 30 overlapping windows, averaged. "
                                     "Top: predicted price + 5-95% range vs realized (black). Bottom: "
                                     "forecast error (predicted - realized) over time - negative "
                                     "because the risk-neutral forecast under-shoots the realized "
                                     "uptrend. The 4 routes differ only in OPTION price, not this "
                                     "underlying forecast."),
            "backtest_drift_compare.png": ("Drift choice, side by side (underlying-price prediction). "
                                           "Left = risk-neutral drift r (the pricing measure): flat-"
                                           "centred cone under-shoots the rally, error dives negative "
                                           "(MAE 0.93 EUR, 67% coverage). Right = real-world mu_hat "
                                           "estimated per window: the cone chases the trend, lower "
                                           "error (MAE 0.74) but noisier and overshoots reversals "
                                           "(57% coverage). Honest tradeoff; M=8 sliding windows."),
            "backtest_routes_timeseries.png": ("Full verification: an ATM call priced on EVERY "
                                               "one of the 192 sliding windows with all four routes "
                                               "(real circuit runs, no analytic shortcut). Top: price "
                                               "per route vs exact tree; bottom: abs error (log). "
                                               "Fourier is statevector-exact (~1e-8), QAE ~2e-5 "
                                               "(beats its 0.01 target), MC ~1.6e-3 (sampling), "
                                               "QSVT ~4.5e-3 (straddle floor) - all routes agree."),
            "backtest_walkforward.png": ("Walk-forward backtest: the year is cut into 60-day "
                                         "chunks; each forecast cone (M=8 steps, ~3.75 trading "
                                         "days/step) is calibrated on the prior 30 days (shaded) "
                                         "and tested on the next 30 vs the realized price (black). "
                                         "Honest: the risk-neutral cone is flat, so a trending "
                                         "stock rides its upper edge. Bottom: all four routes "
                                         "reproduce the exact-tree option price each chunk."),
            "model_results.png": ("Actual run of all four routes: each recovers the "
                                  "exact-tree price (M=3); right panel is the real "
                                  "transpiled IQM CZ depth + qubits. QAE is shallowest "
                                  "(~16 CZ); QSVT is the deep, honest straddle route."),
            "error_vs_queries.png": ("Estimation error vs oracle queries (cf. "
                                     "Stamatopoulos 2020, Fig. 11): AE slope -0.80 vs "
                                     "MC -0.58. Honest: AE is under the ideal -1.0 due "
                                     "to small-M finite-shot saturation; queries != samples."),
            "complexity.png": ("Query complexity to reach accuracy eps: MC O(1/eps^2) vs "
                               "QAE O(1/eps) - the textbook quadratic separation (analytic "
                               "theory curves; cite Montanaro 2015)."),
            "speedup.png": ("Seed-averaged empirical RMS error vs queries: QAE ~ -0.80, "
                            "MC ~ -0.58 - the measured counterpart to the complexity plot."),
            "depth_crossover.png": ("Novelty: Hamming-weight poly(M) phase oracle vs naive "
                                    "2^M CZ depth. Naive infeasible from M~14; Hamming prices "
                                    "M=14 to 3e-4 error on 19 qubits. The strongest result."),
        },
        "regenerate": None,                 # collected as built by quantum_pricer
        "table": {
            "header": ["route", "query complexity", "depth (IQM CZ)", "note"],
            "rows": [
                ["Classical MC", "O(1/eps^2)", "0", "baseline"],
                ["QNDM Fourier", "O(1/eps^2) shots", "~112 CZ (M=4)", "shallow, exact loading"],
                ["QNDM QAE", "O(1/eps)", "~16 CZ", "quadratic speedup"],
                ["QSVT (novel)", "O(1/eps)", "~2240 CZ", "honest straddle construction"],
            ],
        },
        "prose": ("Loads the full price-path tree into a superposition and reads the fair "
                  "price off it via QNDM phase encoding + amplitude estimation, removing the "
                  "expensive distribution-loading oracle that bottlenecks prior quantum pricers. "
                  "Ground truth = exact CRR binomial tree; Nokia (NOKIA.HE) European/Asian call. "
                  "Calibration is look-ahead-free (only closes on or before t0 = 2025-06-05); "
                  "out-of-sample checks live in results/dashboard.html section 6."),
        "provenance": "Stamatopoulos et al., Quantum 4:291 (2020); Montanaro, Proc. R. Soc. A (2015).",
    },
    {
        "key": "triage",
        "title": "Triage Lab - QAE scaling advantage",
        "claim": ("QAE cost grows as O(1/eps) (log-log slope -1 vs target accuracy eps) "
                  "vs Monte-Carlo's O(1/eps^2) (slope -2) - quadratic speedup at tight eps."),
        "figure": _TRIAGE_FIG,
        "regenerate": None,
        "table": {
            "header": ["method", "cost slope vs eps (measured)", "interpretation"],
            "rows": [
                ["QAE (quantum)", "~ -1", "queries ~ O(1/eps), i.e. error ~ O(1/queries)"],
                ["Monte-Carlo", "~ -2", "samples ~ O(1/eps^2), i.e. error ~ O(1/sqrt(samples))"],
            ],
        },
        "prose": ("Overnight triage of quantum-finance methods (QAE / QAOA / fraud). The QAE "
                  "scaling curve shows the quadratic edge - but only at tight accuracy; at coarse "
                  "eps the fixed per-round cost lets classical win. Honest, not hyped."),
        "provenance": ("Triage-lab worktree REPORT.md (worktree archived; measured figure "
                       "cached in results/figures/); same QAE math as the pricer track."),
    },
]

SUMMARY = {
    "framing": ("Newton could not predict 'the madness of people'; Feynman said classical "
                "intuition fails. Both say markets are non-classical - so we compute with it."),
    "headlines": [
        {"label": "Cognition", "value": "q = -0.003", "sub": "parameter-free QQ-equality holds"},
        {"label": "Pricing", "value": "O(1/eps) vs O(1/eps^2)", "sub": "quadratic MC speedup"},
        {"label": "Triage", "value": "cost slope -1 vs -2", "sub": "queries vs target eps (measured)"},
    ],
}
