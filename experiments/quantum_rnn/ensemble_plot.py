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
os.environ.setdefault("OMP_NUM_THREADS", "1")     # pin BLAS threads BEFORE numpy/torch import
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import csv
import json

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
EXPORT_DIR = os.path.join(HERE, "ensemble_export")   # raw plot data for teammates

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
    next_abs_te = task["next_abs"][te]

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
    _export(day, prices, y_te, next_abs_te, c_arrs, q_arrs, c_ens, q_ens,
            c_aucs, q_aucs, c_auc, q_auc, task, provenance)
    print(f"exported raw data -> {EXPORT_DIR}")


def _export(days, prices, y_te, next_abs_te, c_arrs, q_arrs, c_ens, q_ens,
            c_aucs, q_aucs, c_auc, q_auc, task, provenance):
    """Write the exact arrays behind the figure as plain CSV/JSON for re-plotting."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    cnames, qnames = list(c_arrs), list(q_arrs)
    cn = {nm: nrm(c_arrs[nm]) for nm in cnames}      # normalized member scores (feed the means)
    qn = {nm: nrm(q_arrs[nm]) for nm in qnames}

    with open(os.path.join(EXPORT_DIR, "ensemble_timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trading_day", "price_eur", "true_vol_event", "next_abs_return",
                    "classical_ensemble_score", "quantum_ensemble_score"])
        for i, d in enumerate(days):
            w.writerow([int(d), f"{prices[d]:.6f}", int(y_te[i]), f"{next_abs_te[i]:.8f}",
                        f"{c_ens[i]:.6f}", f"{q_ens[i]:.6f}"])

    with open(os.path.join(EXPORT_DIR, "members_normalized.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trading_day"] + cnames + qnames)
        for i, d in enumerate(days):
            w.writerow([int(d)] + [f"{cn[nm][i]:.6f}" for nm in cnames]
                       + [f"{qn[nm][i]:.6f}" for nm in qnames])

    summary = dict(
        provenance=provenance, ticker="NOKIA.HE", task="next-day top-decile |return| (vol event)",
        sequence_length_L=L, train_frac=0.5, proxy_quantile=0.9,
        n_windows=int(len(task["X"])), n_train=int(task["n_train"]),
        n_test=int(len(days)), n_test_events=int(y_te.sum()),
        neural_seeds=list(SEEDS), epochs=EPOCHS, lr=LR, weight_decay=WD,
        classical_members=cnames, quantum_members=qnames,
        member_auc={**{k: round(v, 3) for k, v in c_aucs.items()},
                    **{k: round(v, 3) for k, v in q_aucs.items()}},
        classical_ensemble_auc=round(c_auc, 3), quantum_ensemble_auc=round(q_auc, 3),
        note=("Scores are min-max normalized per member over the test window, then "
              "averaged uniformly within each family. Models predict next-day VOLATILITY, "
              "not price. ~0.01 run-to-run AUC noise on retrain (BLAS); this file is the "
              "canonical data for the committed figure."),
    )
    with open(os.path.join(EXPORT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


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
            label=f"classical: 4 NN + GARCH  AUC {c_auc:.3f}")
    ax.plot(days, q_ens, color=q_col, lw=2.0,
            label=f"quantum: 4 NN  AUC {q_auc:.3f}")
    ax.set_xlim(days[0], days[-1])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("held-out trading day")
    ax.set_ylabel("ensemble score (normalized)")
    ax.set_title("Frozen-expert ensembles — classical (incl. GARCH) vs quantum "
                 "(NOKIA.HE, held-out)")
    ax.legend(loc="upper left", fontsize=9.5)

    cap = ("Classical = RNN/GRU/LSTM/Transformer (each seed-averaged over 5 seeds) + "
           "GARCH(1,1); quantum = QRNN/QGRU/QLSTM/QTransformer. Each member's held-out "
           "score min-max normalized to [0,1], then averaged uniformly (no learned gate). "
           "Shaded = true top-decile next-day |return| days. Models predict next-day "
           f"VOLATILITY, not price. Near-identical AUC ({c_auc:.3f} vs {q_auc:.3f}) -- "
           "no classical/quantum separation.")
    if _HAVE_STYLE:
        style.caption(fig, cap)
        style.provenance(fig, provenance)
    else:
        fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.4, style="italic", wrap=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
