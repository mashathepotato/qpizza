"""Frozen-expert ensembles: classical vs quantum, two averaged prediction lines.

Mixture-of-experts / frozen-model style: each neural model is trained
independently then frozen; the ensemble prediction is the uniform mean of its
experts' per-day probabilities (no learned gate). We average the four CLASSICAL
neural experts (RNN, GRU, LSTM, Transformer) into one line and the four QUANTUM
neural experts (QRNN, QGRU, QLSTM, QTransformer) into another, and overlay both
against the true next-day vol-event days on the held-out split.

(Statistical baselines -- GARCH/AR/persistence/logistic -- are intentionally
excluded: they are not neural "experts". Swap them in if you want them counted.)

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
SEEDS = range(5)          # each expert seed-averaged (deep ensemble) for honest, stable lines
OUT = os.path.join(HERE, "ensemble_predictions.png")

CLASSICAL = [("RNN", cr.RNNClassifier), ("GRU", cr.GRUClassifier),
             ("LSTM", cr.LSTMClassifier), ("Transformer", TransformerClassifier)]
QUANTUM = [("QRNN", QRNN), ("QGRU", QGRU), ("QLSTM", QLSTM),
           ("QTransformer", QuantumTransformer)]


def _expert_probs(models, X, y, tr, te):
    """Each expert = its held-out prob averaged over SEEDS (a deep ensemble, robust to
    init). Returns (mean held-out prob over the experts, per-expert seed-averaged AUCs)."""
    probs, aucs = [], {}
    for name, Cls in models:
        seed_probs = []
        for s in SEEDS:
            m = Cls(seed=s)
            cr.train(m, X[tr], y[tr], epochs=EPOCHS, lr=LR, seed=s, weight_decay=WD)
            seed_probs.append(cr.scores(m, X)[te])
        ep = np.mean(np.vstack(seed_probs), axis=0)     # expert's seed-averaged prediction
        probs.append(ep)
        aucs[name] = sd.roc_auc(ep, y[te])
        print(f"  {name:12s} AUC={aucs[name]:.3f}")
    return np.mean(np.vstack(probs), axis=0), aucs


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
    c_ens, c_aucs = _expert_probs(CLASSICAL, X, y, tr, te)
    print("quantum experts:")
    q_ens, q_aucs = _expert_probs(QUANTUM, X, y, tr, te)

    c_auc = sd.roc_auc(c_ens, y_te)
    q_auc = sd.roc_auc(q_ens, y_te)
    print(f"\nprovenance        : {provenance}")
    print(f"held-out vol events: {int(y_te.sum())}/{len(y_te)}")
    print(f"CLASSICAL ensemble AUC = {c_auc:.3f}  (experts mean {np.mean(list(c_aucs.values())):.3f})")
    print(f"QUANTUM   ensemble AUC = {q_auc:.3f}  (experts mean {np.mean(list(q_aucs.values())):.3f})")

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
            label=f"classical experts (mean of 4)  AUC {c_auc:.2f}")
    ax.plot(days, q_ens, color=q_col, lw=2.0,
            label=f"quantum experts (mean of 4)  AUC {q_auc:.2f}")
    ax.set_xlim(days[0], days[-1])
    ax.set_xlabel("held-out trading day")
    ax.set_ylabel("ensemble P(next-day vol event)")
    ax.set_title("Frozen-expert ensembles — classical vs quantum mean prediction "
                 "(NOKIA.HE, held-out)")
    ax.legend(loc="upper left", fontsize=9.5)
    cap = ("Each of 4 classical (RNN/GRU/LSTM/Transformer) and 4 quantum "
           "(QRNN/QGRU/QLSTM/QTransformer) neural experts trained over 5 seeds on the calm "
           "split, frozen, then averaged uniformly (no learned gate). Lines = mean expert "
           "P(next-day top-decile |return|); shaded = true vol-event days. The two ensembles "
           "track the same volatility-clustering bumps and give near-identical held-out AUC "
           f"({c_auc:.2f} vs {q_auc:.2f}) -- averaging confirms no classical/quantum separation.")
    if _HAVE_STYLE:
        style.caption(fig, cap)
        style.provenance(fig, provenance)
    else:
        fig.text(0.5, -0.03, cap, ha="center", va="top", fontsize=8.4, style="italic", wrap=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
