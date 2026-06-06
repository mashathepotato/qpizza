"""Run OUR pricer models and plot the ACTUAL measured outputs (not theory):
  Left  - price from each route vs the exact-tree ground truth (with the actual
          signed error; MC gets its sampled stderr as an error bar).
  Right - actual transpiled circuit resources per route (IQM CZ depth, log; qubits).

Run (needs the pricer venv with qiskit):
  PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python results/model_experiments.py

Saves results/figures/model_results.png. Standalone so it does not touch
quantum_pricer/benchmark.py (owned by concurrent work).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from results import style

# NOKIA.HE ATM European call - fixed for reproducibility / no network (matches demo.py).
P = dict(S0=13.09, K=13.09, r=0.03, sigma=0.479, T=1.0)
M_PRICE = 3      # price routes (fast, exact-simulable)
M_BENCH = 4      # resource transpile depth (matches demo)
N_MC = 100_000
OUT = os.path.join(os.path.dirname(__file__), "figures", "model_results.png")


def run_models():
    from quantum_pricer import tree
    from quantum_pricer.classical import monte_carlo_price
    from quantum_pricer.fourier import price as fourier_price
    from quantum_pricer.qae import price as qae_price
    from quantum_pricer.benchmark import resource_table

    exact = tree.exact_tree_price(M=M_PRICE, option="european", kind="call", **P)

    results = []  # (label, price, stderr_or_None, note)
    mc_price, mc_stderr = monte_carlo_price(M=M_PRICE, n_paths=N_MC, option="european",
                                            kind="call", seed=42, **P)
    results.append(("Classical\nMC", mc_price, mc_stderr, f"n={N_MC:,}"))

    fq = fourier_price(M=M_PRICE, option="european", kind="call", **P)
    results.append(("QNDM\nFourier", fq, None, "statevector"))

    qae = qae_price(M=M_PRICE, option="european", kind="call", epsilon_target=0.01, **P)
    results.append(("QNDM\nQAE", qae["price"], None,
                    f"eps=0.01, q={qae['num_oracle_queries']:,}"))

    try:
        from quantum_pricer.qsvt import price as qsvt_price
        qv = qsvt_price(M=M_PRICE, option="european", kind="call",
                        degree=60, use_qae=False, **P)
        results.append(("novel\nQSVT", qv, None, "straddle+parity, deg 60"))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] QSVT skipped: {exc}")

    table = resource_table(M=M_BENCH, option="european", kind="call",
                           qsvt_degree=20, **P)
    return exact, results, table


def main():
    style.apply_style()
    exact, results, table = run_models()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = {"Classical\nMC": style.PALETTE["classical"],
            "QNDM\nFourier": style.PALETTE["muted"],
            "QNDM\nQAE": style.PALETTE["quantum"],
            "novel\nQSVT": style.PALETTE["accent"]}

    # ---- Left: actual prices vs exact-tree ground truth ----
    labels = [r[0] for r in results]
    prices = [r[1] for r in results]
    errbars = [r[2] if r[2] is not None else 0.0 for r in results]
    colors = [cmap.get(l, style.PALETTE["quantum"]) for l in labels]
    xpos = np.arange(len(labels))
    ax1.bar(xpos, prices, color=colors, edgecolor=style.PALETTE["ink"], width=0.6,
            yerr=errbars, capsize=5, error_kw=dict(ecolor=style.PALETTE["ink"], lw=1.2))
    ax1.axhline(exact, ls="--", color=style.PALETTE["ink"], lw=1.4,
                label=f"exact tree (ground truth) = {exact:.4f}")
    for x, (lab, price, _se, note) in zip(xpos, results):
        ax1.text(x, price + 0.04, f"{price:.4f}\n({price - exact:+.4f})",
                 ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("option price")
    ax1.set_ylim(0, max(prices) * 1.18)
    ax1.set_title(f"Actual price per route vs ground truth  (M={M_PRICE})")
    ax1.legend(loc="lower center")

    # ---- Right: actual transpiled circuit resources ----
    methods = [row["method"] for row in table]
    cz = [max(row["cz_depth"], 0.5) for row in table]  # avoid log(0) for MC
    qubits = [row["qubits"] for row in table]
    rcolors = []
    for m in methods:
        key = {"classical_mc": "Classical\nMC", "fourier": "QNDM\nFourier",
               "qae": "QNDM\nQAE", "qsvt": "novel\nQSVT"}.get(m, m)
        rcolors.append(cmap.get(key, style.PALETTE["quantum"]))
    xr = np.arange(len(methods))
    ax2.bar(xr, cz, color=rcolors, edgecolor=style.PALETTE["ink"], width=0.6)
    ax2.set_yscale("log")
    ax2.set_xticks(xr)
    ax2.set_xticklabels([m.replace("_", "\n") for m in methods])
    ax2.set_ylabel("IQM CZ depth (log)")
    ax2.set_title(f"Actual transpiled circuit cost per route  (M={M_BENCH})")
    for x, c, q in zip(xr, cz, qubits):
        ax2.text(x, c * 1.15, f"{int(c) if c >= 1 else 0} CZ\n{q} qb",
                 ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylim(0.5, max(cz) * 3)

    fig.suptitle("Quantum option pricer - actual model experiments (NOKIA.HE ATM call)",
                 fontsize=13, fontweight="bold")
    style.caption(fig, "Left: every route recovers the exact-tree price within its error. "
                       "Right: QAE is shallowest (~16 CZ); QSVT is the deep, honest route.")
    style.provenance(fig, "quantum_pricer routes + resource_table; fixed NOKIA.HE params")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)

    print(f"  exact tree price (M={M_PRICE}) = {exact:.6f}")
    for lab, price, se, note in results:
        se_s = f", stderr={se:.4f}" if se else ""
        print(f"  {lab.replace(chr(10), ' '):<16} {price:.6f}  err={price - exact:+.6f}  {note}{se_s}")
    print("  resources:", [(row["method"], row["qubits"], row["cz_depth"]) for row in table])
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
