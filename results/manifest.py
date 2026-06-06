"""Single source of truth for dashboard content + figure source locations.

Each track declares where its figure is, the key-number table to show, the prose
interpretation, and provenance. build_dashboard.py reads ONLY this for content.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(*parts):
    return os.path.join(ROOT, *parts)


# Triage still lives in a worktree; pricer + cognition are on main.
_TRIAGE = _p(".claude", "worktrees", "triage-lab", "triage")

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
        "figure": _p("results", "figures", "model_results.png"),
        "extra_figures": [
            _p("results", "figures", "error_vs_queries.png"),
            _p("quantum_pricer", "complexity.png"),
            _p("quantum_pricer", "speedup.png"),
            _p("quantum_pricer", "depth_crossover.png"),
        ],
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
                  "Ground truth = exact CRR binomial tree; Nokia (NOKIA.HE) European/Asian call."),
        "provenance": "Stamatopoulos et al., Quantum 4:291 (2020); Montanaro, Proc. R. Soc. A (2015).",
    },
    {
        "key": "triage",
        "title": "Triage Lab - QAE scaling advantage",
        "claim": "QAE error falls with slope ~ -1 vs classical Monte-Carlo ~ -2 in queries.",
        "figure": os.path.join(_TRIAGE, "plots", "qae_scaling.png"),
        "regenerate": None,
        "table": {
            "header": ["method", "scaling slope", "interpretation"],
            "rows": [
                ["QAE (quantum)", "~ -1", "error ~ O(1/queries): quadratic edge at tight eps"],
                ["Monte-Carlo", "~ -2", "error ~ O(1/sqrt(samples))"],
            ],
        },
        "prose": ("Overnight triage of quantum-finance methods (QAE / QAOA / fraud). The QAE "
                  "scaling curve shows the quadratic edge - but only at tight accuracy; at coarse "
                  "eps the fixed per-round cost lets classical win. Honest, not hyped."),
        "provenance": "Triage-lab worktree REPORT.md; same QAE math as the pricer track.",
    },
]

SUMMARY = {
    "framing": ("Newton could not predict 'the madness of people'; Feynman said classical "
                "intuition fails. Both say markets are non-classical - so we compute with it."),
    "headlines": [
        {"label": "Cognition", "value": "q = -0.003", "sub": "parameter-free QQ-equality holds"},
        {"label": "Pricing", "value": "O(1/eps) vs O(1/eps^2)", "sub": "quadratic MC speedup"},
        {"label": "Triage", "value": "slope -1 vs -2", "sub": "QAE scaling edge"},
    ],
}
