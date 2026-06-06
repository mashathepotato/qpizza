"""Frozen-expert ensembles: classical vs quantum, two averaged prediction lines.

Mixture-of-experts / frozen-model style: each member is trained (or fit)
independently then frozen; the ensemble prediction is the uniform mean of its
members' per-day scores (no learned gate). To mix members with different output
scales (neural sigmoids in [0,1] and GARCH's variance forecast), every member's
held-out score is min-max normalized to [0,1] before averaging.

  CLASSICAL line = RNN, GRU, LSTM, Transformer  +  GARCH(1,1)
  QUANTUM line   = QRNN, QGRU, QLSTM, QTransformer

Both overlaid against the true next-day vol-event days on the held-out split.

Run:  ./quantum_pricer/.venv/bin/python experiments/quantum_rnn/ensemble_plot.py
Saves ensemble_predictions.png next to this script.
"""
import os
import sys

import numpy as np
import torch
torch.set_num_threads(1)      # single-threaded -> deterministic BLAS reductions (reproducible AUCs)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import seqdata as sd
import classical_rnn as cr
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
SEEDS = range(5)          # each neural expert seed-averaged (deep ensemble) for stable lines
OUT = os.path.join(HERE, "ensemble_predictions.png")

CLASSICAL = [("RNN", cr.RNNClassifier), ("GRU", cr.GRUClassifier),
             ("LSTM", cr.LSTMClassifier), ("Transformer", TransformerClassifier)]
QUANTUM = [("QRNN", QRNN), ("QGRU", QGRU), ("QLSTM", QLSTM),
           ("QTransformer", QuantumTransformer)]


def nrm(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _neural_arrays(models, X, y, tr, te):
    """Return {name: seed-averaged held-out score} and {name: AUC} for each expert."""
    arrs, aucs = {}, {}
    for name, Cls in models:
        seed_scores = []
        for s in SEEDS:
            m = Cls(seed=s)
            cr.train(m, X[tr], y[tr], epochs=EPOCHS, lr=LR, seed=s, weight_decay=WD)
            seed_scores.append(cr.scores(m, X)[te])
        arrs[name] = np.mean(np.vstack(seed_scores), axis=0)
        aucs[name] = sd.roc_auc(arrs[name], y[te])
        print(f"  {name:12s} AUC={aucs[name]:.3f}")
    return arrs, aucs


def _ensemble(arrs, y_te):
    """Uniform mean of each member's min-max-normalized held-out score."""
    ens = np.mean(np.vstack([nrm(v) for v in arrs.values()]), axis=0)
    return ens, sd.roc_auc(ens, y_te)


def main():
    if _HAVE_STYLE:
        style.apply_style()
    prices, provenance = sd.fetch_prices()
    R = np.diff(np.log(prices))
    task = sd.build_task_from_returns(R, L=L, train_frac=0.5, proxy_q=0.9)
    X, y = task["X"], task["y"]
    n, n_train = len(X), task["n_train"]
    tr, te = slice(0, n_train), slice(n_train, n)
    day = np.arange(L, L + n)[te]
    y_te = y[te]

    print("classical experts:")
    c_arrs, c_aucs = _neural_arrays(CLASSICAL, X, y, tr, te)
    # fold GARCH(1,1) into the classical average (deterministic; not seed-dependent)
    garch = ar.garch_forecast(R, fit_end=L + n_train)[L:L + n][te]
    c_arrs["GARCH(1,1)"] = garch
    c_aucs["GARCH(1,1)"] = sd.roc_auc(garch, y_te)
    print(f"  {'GARCH(1,1)':12s} AUC={c_aucs['GARCH(1,1)']:.3f}")
    print("quantum experts:")
    q_arrs, q_aucs = _neural_arrays(QUANTUM, X, y, tr, te)

    c_ens, c_auc = _ensemble(c_arrs, y_te)
    q_ens, q_auc = _ensemble(q_arrs, y_te)
    print(f"\nprovenance        : {provenance}")
    print(f"held-out vol events: {int(y_te.sum())}/{len(y_te)}")
    print(f"CLASSICAL ensemble (4 NN + GARCH) AUC = {c_auc:.3f}  "
          f"(members mean {np.mean(list(c_aucs.values())):.3f})")
    print(f"QUANTUM   ensemble (4 NN)          AUC = {q_auc:.3f}  "
          f"(members mean {np.mean(list(q_aucs.values())):.3f})")

    _plot(day, y_te, c_ens, q_ens, c_auc, q_auc, provenance)
    print(f"saved {OUT}")


def _plot(days, y_true, c_ens, q_ens, c_auc, q_auc, provenance):
    c_col, q_col, ev_col = "#c44536", "#2a6f97", "#c44536"
    fig, ax = plt.subplots(figsize=(12, 6))
    first = True
    for d, yy in zip(days, y_true):
        if yy == 1:
            ax.axvspan(d - 0.5, d + 0.5, color=ev_col, alpha=0.16, lw=0,
                       label="true vol event" if first else None)
            first = False
    ax.plot(days, c_ens, color=c_col, lw=2.0,
            label=f"classical: 4 NN + GARCH  AUC {c_auc:.2f}")
    ax.plot(days, q_ens, color=q_col, lw=2.0,
            label=f"quantum: 4 NN  AUC {q_auc:.2f}")
    ax.set_xlim(days[0], days[-1])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("held-out trading day")
    ax.set_ylabel("ensemble score (mean of normalized member predictions)")
    ax.set_title("Frozen-expert ensembles — classical (incl. GARCH) vs quantum "
                 "(NOKIA.HE, held-out)")
    ax.legend(loc="upper left", fontsize=9.5)
    cap = ("Classical = RNN/GRU/LSTM/Transformer (each seed-averaged over 5 seeds) + "
           "GARCH(1,1); quantum = QRNN/QGRU/QLSTM/QTransformer. Each member's held-out "
           "score min-max normalized to [0,1], then averaged uniformly (no learned gate). "
           "Shaded = true top-decile next-day |return| days. The two ensembles track the "
           f"same volatility-clustering bumps with near-identical AUC ({c_auc:.2f} vs "
           f"{q_auc:.2f}) -- adding GARCH does not separate classical from quantum.")
    if _HAVE_STYLE:
        style.caption(fig, cap)
        style.provenance(fig, provenance)
    else:
        fig.text(0.5, -0.03, cap, ha="center", va="top", fontsize=8.4, style="italic", wrap=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
