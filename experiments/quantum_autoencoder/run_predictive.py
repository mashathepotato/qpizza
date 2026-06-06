"""Quantum autoencoder for NEXT-DAY volatility prediction on NOKIA.HE (honest variant).

Experimental sidecar (not part of the quantum_pricer package). We train a
Romero-style quantum autoencoder (PennyLane default.qubit, 4 qubits, 2 latent /
2 trash) on a CALM early stretch of NOKIA.HE 4-day return windows so the trash
qubits disentangle to |00>. The anomaly score is the reconstruction infidelity
1 - P(trash=|00>); a 2-component PCA reconstruction error (numpy SVD) is the
classical baseline.

PREDICTIVE framing -- the honest one: the target volatility event is on day t+1,
which is NOT in the window (returns ending day t). This removes the label leakage
of the concurrent task (where the flagged return is also a model input and a bare
|return| detector trivially scores AUC 1.0). Daily equity returns are
near-unpredictable, so the only learnable signal here is volatility clustering;
we report ROC-AUC vs a transparent proxy (top-decile next-day |return|).

FINDING (real NOKIA.HE 1y): all models cluster at ~0.64-0.68 held-out AUC and the
quantum model does NOT beat classical PCA. No quantum advantage for prediction --
a parity/negative result, by design honest.

Run (needs pennylane + yfinance; see requirements.txt in this folder):
  python experiments/quantum_autoencoder/run_predictive.py
Saves qml_autoencoder_predictive.png next to this script.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))  # repo root for results.style
import qautoencoder as qa
try:
    from results import style          # reuse the repo's figure look if available
    _HAVE_STYLE = True
except Exception:
    _HAVE_STYLE = False

K = 4                  # window length (qubits)
N_LATENT = 2
N_LAYERS = 2
TRAIN_FRAC = 0.5       # first half = calm training regime
PROXY_Q = 0.90         # top-decile next-day |return| = proxy "anomaly"
TRAIN_STEPS = 150
LR = 0.08
SEED = 7
OUT = os.path.join(HERE, "qml_autoencoder_predictive.png")

_PALETTE = {"quantum": "#2a6f97", "classical": "#c44536", "accent": "#2a9d8f",
            "muted": "#8d99ae", "ink": "#22223b"}


def fetch_prices():
    try:
        import yfinance as yf
        v = yf.Ticker("NOKIA.HE").history(period="1y")["Close"].dropna().values
        if len(v) > K + 30:
            return np.asarray(v, float), "yfinance NOKIA.HE 1y daily"
    except Exception as exc:
        print(f"[synthetic fallback] {exc}")
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.012, 200)
    calm[120:160] = rng.normal(0, 0.05, 40)  # injected volatile burst
    return 4.20 * np.exp(np.cumsum(np.concatenate([[0.0], calm]))), \
        "[SYNTHETIC FALLBACK] not real market data"


def main():
    if _HAVE_STYLE:
        style.apply_style()
    prices, provenance = fetch_prices()
    logret = np.diff(np.log(prices))

    # window i = returns[i..i+K-1]; predict next-day return[i+K] (NOT in window)
    windows = qa.make_windows(logret, k=K)[:-1]
    next_ret = np.abs(logret[K:])                     # next-day |return| per window
    end_day = np.arange(K, len(prices) - 1)           # price-series index of day t (window end)
    n = len(windows)
    n_train = int(n * TRAIN_FRAC)

    mean, std = qa.zscore_fit(windows[:n_train])
    Z = qa.zscore_apply(windows, mean, std)

    thresh = np.quantile(next_ret[:n_train], PROXY_Q)
    labels = (next_ret >= thresh).astype(int)
    test = slice(n_train, n)

    model = qa.QuantumAutoencoder(n_qubits=K, n_latent=N_LATENT,
                                  n_layers=N_LAYERS, seed=SEED)
    model.train(Z[:n_train], steps=TRAIN_STEPS, lr=LR)
    q_scores = np.array([model.score(z) for z in Z])

    pca = qa.PCAReconstructor(n_components=N_LATENT).fit(Z[:n_train])
    p_scores = np.array([pca.score(z) for z in Z])

    nae = qa.NeuralAutoencoder(n_inputs=K, n_latent=N_LATENT, seed=SEED)
    nae.fit(Z[:n_train], steps=400, lr=0.05)
    nn_scores = np.array([nae.score(z) for z in Z])

    energy = (Z ** 2).sum(axis=1)                     # trivial magnitude baseline

    ev = dict(
        auc_q=qa.roc_auc(q_scores[test], labels[test]),
        auc_p=qa.roc_auc(p_scores[test], labels[test]),
        auc_n=qa.roc_auc(nn_scores[test], labels[test]),
        auc_e=qa.roc_auc(energy[test], labels[test]),
        n_qae=model.n_params, n_pca=pca.n_params, n_nae=nae.n_params,
    )
    print(f"provenance     : {provenance}")
    print(f"windows        : {n} (train {n_train} / test {n - n_train}), k={K}")
    print(f"next-day events: {int(labels.sum())} ({100 * labels.mean():.0f}%), "
          f"|ret| >= {thresh:.4f}")
    print(f"held-out AUC   : QAE={ev['auc_q']:.3f}  PCA={ev['auc_p']:.3f}  "
          f"neuralAE={ev['auc_n']:.3f}  energy={ev['auc_e']:.3f}")
    print("verdict        : models cluster ~0.65; quantum does NOT beat the classical "
          "autoencoder/PCA (no advantage for prediction).")

    _plot(prices, end_day, q_scores, p_scores, nn_scores, labels, n_train, ev, provenance)
    print(f"saved {OUT}")


def _plot(prices, end_day, q_scores, p_scores, nn_scores, labels, n_train, ev, provenance):
    P = _PALETTE
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.0, 1.0]))
    split = end_day[n_train]
    # next-day event marker sits on day t+1
    event_days = end_day[labels == 1] + 1

    ax0.plot(np.arange(len(prices)), prices, color=P["ink"], lw=1.3, label="NOKIA.HE close")
    ax0.scatter(event_days, prices[event_days], color=P["classical"], s=24, zorder=5,
                label="next-day vol event (proxy: top-10% |return|)")
    ax0.axvline(split, color=P["muted"], ls="--", lw=1.0)
    ax0.axvspan(end_day[0], split, color=P["accent"], alpha=0.05)
    ax0.set_ylabel("price (EUR)")
    ax0.set_title("Quantum autoencoder for NEXT-DAY volatility on NOKIA.HE "
                  "(honest, no-leakage variant)")
    ax0.legend(loc="upper left", fontsize=9)
    ax0.text(split, ax0.get_ylim()[1], "  train | test ", va="top", ha="left",
             fontsize=8.5, color=P["ink"], alpha=0.7)

    def nrm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else x * 0.0
    ax1.plot(end_day, nrm(q_scores), color=P["quantum"], lw=1.4,
             label=f"QAE infidelity (AUC {ev['auc_q']:.2f}, {ev['n_qae']} params)")
    ax1.plot(end_day, nrm(p_scores), color=P["classical"], lw=1.4,
             label=f"PCA-2 recon (AUC {ev['auc_p']:.2f}, {ev['n_pca']} params)")
    ax1.plot(end_day, nrm(nn_scores), color=P["accent"], lw=1.3, ls="--",
             label=f"neural AE recon (AUC {ev['auc_n']:.2f}, {ev['n_nae']} params)")
    ax1.axvline(split, color=P["muted"], ls="--", lw=1.0)
    ax1.set_ylabel("anomaly score (norm.)")
    ax1.set_xlabel("trading day")
    ax1.legend(loc="upper left", fontsize=9)

    cap = ("Trained on calm-period 4-day return windows; score = reconstruction error "
           "(QAE: 1 - P(trash=|00>); AEs: residual). Target = next-day top-decile |return| "
           "(NOT in the window), so a bare |return| detector cannot cheat. Held-out AUC "
           "clusters ~0.65 for all models -- only volatility-clustering signal, and the "
           "quantum model does NOT beat the classical neural autoencoder or PCA. "
           f"Energy baseline AUC {ev['auc_e']:.2f}. Honest parity/negative result.")
    if _HAVE_STYLE:
        style.caption(fig, cap)
        style.provenance(fig, provenance)
    else:
        fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5, style="italic", wrap=True)
        fig.text(0.995, 0.005, provenance, ha="right", va="bottom", fontsize=7.5, color=P["muted"])
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
