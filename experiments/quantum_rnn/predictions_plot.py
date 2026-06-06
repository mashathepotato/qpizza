"""Static v1: every model's per-day next-day-volatility prediction vs ground truth.

All 12 models (classical + quantum recurrent, transformers, GARCH/AR, floors) are
trained once on the calm-period split, then their held-out daily scores are
min-max normalized to [0,1] and overlaid as lines over the held-out trading days.
True top-decile next-day |return| days (the binary ground truth the models are
scored on) are shaded. Lines are colored by family with per-family linestyles to
keep 12 traces legible.

(v2, out of scope here: animate this as a left-to-right day-by-day sweep.)

Run:  ./quantum_pricer/.venv/bin/python experiments/quantum_rnn/predictions_plot.py
Saves predictions_per_day.png next to this script.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import seqdata as sd
import classical_rnn as cr
import baselines as bl
import ar_baseline as ar
from quantum_rnn import QRNN
from quantum_gru import QGRU
from quantum_lstm import QLSTM
from classical_tf import TransformerClassifier
from quantum_tf import QuantumTransformer
try:
    from results import style
    _HAVE_STYLE = True
except Exception:
    _HAVE_STYLE = False

L = 10
EPOCHS = 150
LR = 0.05
WD = 1e-2
SEED = 7
OUT = os.path.join(HERE, "predictions_per_day.png")

# (name, family, color, linestyle)
TORCH = [
    ("QRNN", "quantum", QRNN, "#9ecae1", "-"),
    ("QGRU", "quantum", QGRU, "#4292c6", "-"),
    ("QLSTM", "quantum", QLSTM, "#08519c", "-"),
    ("QTransformer", "quantum", QuantumTransformer, "#6baed6", "--"),
    ("RNN", "classical", cr.RNNClassifier, "#fdae6b", "-"),
    ("GRU", "classical", cr.GRUClassifier, "#e6550d", "-"),
    ("LSTM", "classical", cr.LSTMClassifier, "#a63603", "-"),
    ("Transformer", "classical", TransformerClassifier, "#d6616b", "--"),
]


def nrm(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    if _HAVE_STYLE:
        style.apply_style()
    prices, provenance = sd.fetch_prices()
    R = np.diff(np.log(prices))
    task = sd.build_task_from_returns(R, L=L, train_frac=0.5, proxy_q=0.9)
    X, y, X_raw = task["X"], task["y"], task["X_raw"]
    n, n_train = len(X), task["n_train"]
    tr, te = slice(0, n_train), slice(n_train, n)
    day = np.arange(L, L + n)                 # trading-day index per window's target day
    fit_end = L + n_train

    series = []  # (name, color, ls, normalized held-out score, auc)
    for name, fam, Cls, color, ls in TORCH:
        m = Cls(seed=SEED)
        cr.train(m, X[tr], y[tr], epochs=EPOCHS, lr=LR, seed=SEED, weight_decay=WD)
        s = cr.scores(m, X)
        series.append((name, color, ls, nrm(s[te]), sd.roc_auc(s[te], y[te])))
        print(f"  {name:12s} AUC={series[-1][4]:.3f}")

    h = ar.garch_forecast(R, fit_end=fit_end)[L:L + n]
    series.append(("GARCH(1,1)", "#238b45", ":", nrm(h[te]), sd.roc_auc(h[te], y[te])))
    fc = ar.ar_abs_forecast(R, fit_end=fit_end, p=5)[L:L + n]
    series.append(("AR(|r|,5)", "#74c476", ":", nrm(fc[te]), sd.roc_auc(fc[te], y[te])))
    s_pers = bl.persistence_scores(X_raw)
    series.append(("Persistence", "#969696", ":", nrm(s_pers[te]), sd.roc_auc(s_pers[te], y[te])))
    logit = bl.LogisticClassifier(n_features=L, seed=SEED)
    cr.train(logit, X[tr], y[tr], epochs=200, lr=0.1, seed=SEED, weight_decay=WD)
    s_log = cr.scores(logit, X)
    series.append(("Logistic", "#bdbdbd", ":", nrm(s_log[te]), sd.roc_auc(s_log[te], y[te])))

    _plot(day[te], y[te], series, provenance)
    print(f"saved {OUT}")


def _plot(days, y_true, series, provenance):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    # ground truth: shade true next-day vol-event days
    first = True
    for d, yy in zip(days, y_true):
        if yy == 1:
            ax.axvspan(d - 0.5, d + 0.5, color="#c44536", alpha=0.16,
                       label="true vol event" if first else None, lw=0)
            first = False
    for name, color, ls, s, auc in series:
        ax.plot(days, s, color=color, ls=ls, lw=1.5, alpha=0.9,
                label=f"{name} ({auc:.2f})")
    ax.set_xlim(days[0], days[-1])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("held-out trading day")
    ax.set_ylabel("predicted score  (min-max normalized per model)")
    ax.set_title("Per-day next-day-volatility predictions — all models vs ground truth "
                 "(NOKIA.HE, held-out)")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8.3,
              title="model (held-out AUC)", title_fontsize=8.8, ncol=1)
    cap = ("Each model trained once on the calm-period split; held-out daily scores "
           "min-max normalized to [0,1] for shape comparison. Shaded = true top-decile "
           "next-day |return| days. Blues = quantum, warm = classical (dashed = "
           "transformer), greens = autoregressive, greys = floor baselines. All models "
           "track the same volatility-clustering signal; none separates events cleanly.")
    if _HAVE_STYLE:
        style.caption(fig, cap)
        style.provenance(fig, provenance)
    else:
        fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=8.3, style="italic", wrap=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
