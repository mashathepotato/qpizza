"""One clear error-vs-queries comparison: Monte Carlo vs the Greek-paper SOTA
oracle-QAE (Stamatopoulos et al.) vs OUR QNDM-QAE and OUR QSVT.

Estimation error (y, log) vs number of oracle queries / samples (x, log). Quantum
amplitude-estimation routes scale ~1/queries (slope -1); Monte Carlo ~1/sqrt(N)
(slope -0.5). Finite shots are used so the IAE routes actually iterate Grover rounds
and accumulate real query counts; results are averaged over seeds.

Each method's error is measured against the exact value IT estimates:
  * MC, our QAE, our QSVT  -> exact binomial tree price (M)
  * Greek SOTA oracle-QAE  -> its own discretised lognormal expectation (model_exact)
QSVT has a polynomial-degree bias floor; the Greek route has a rescaling-linearisation
bias floor -- both are shown honestly.

Run (pricer venv; slow ~ several min):
  PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python results/speedup_compare.py
Saves results/figures/speedup_compare.png + results/speedup_compare.json.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from results import style
from quantum_pricer import tree, classical, qae, qsvt, sota

P = dict(S0=13.09, K=13.09, r=0.03, sigma=0.479, T=1.0)
M = 3
EPS = [0.12, 0.06, 0.03, 0.015, 0.008]
MC_N = [250, 1000, 4000, 16000, 64000, 256000]
SEEDS = 5
SHOTS = 100
QSVT_DEGREE = 60
FIG = os.path.join(os.path.dirname(__file__), "figures", "speedup_compare.png")
JS = os.path.join(os.path.dirname(__file__), "speedup_compare.json")


def _avg(fn):
    """Run fn(seed)->(queries,err) over SEEDS; return (mean_queries, rms_err)."""
    qs, es = [], []
    for k in range(SEEDS):
        q, e = fn(k)
        qs.append(q); es.append(e)
    return float(np.mean(qs)), float(np.sqrt(np.mean(np.square(es))))


def main():
    style.apply_style()
    exact = tree.exact_tree_price(M=M, option="european", kind="call", **P)
    series = {}

    # Monte Carlo (vs tree)
    mc_x, mc_y = [], []
    for N in MC_N:
        _, rms = _avg(lambda k: (N, classical.monte_carlo_price(
            M=M, n_paths=N, option="european", kind="call", seed=k, **P)[0] - exact))
        mc_x.append(N); mc_y.append(rms)
    series["Monte Carlo"] = (mc_x, mc_y)
    print("MC done")

    # our QNDM-QAE (vs tree)
    x, y = [], []
    for eps in EPS:
        q, rms = _avg(lambda k: (
            (r := qae.price(M=M, option="european", kind="call",
                            epsilon_target=eps, shots=SHOTS, seed=k, **P))["num_oracle_queries"],
            r["price"] - exact))
        if q > 0:
            x.append(q); y.append(rms)
    series["our QNDM-QAE"] = (x, y)
    print("QAE done")

    # our QSVT + QAE (vs tree)
    x, y = [], []
    for eps in EPS:
        q, rms = _avg(lambda k: (
            (r := qsvt.price(M=M, option="european", kind="call", degree=QSVT_DEGREE,
                             use_qae=True, epsilon_target=eps, shots=SHOTS, seed=k,
                             return_meta=True, **P))["num_oracle_queries"],
            r["price"] - exact))
        if q > 0:
            x.append(q); y.append(rms)
    series["our QSVT"] = (x, y)
    print("QSVT done")

    # Greek SOTA oracle-QAE (vs its own model_exact)
    x, y = [], []
    for eps in EPS:
        def one(k):
            r = sota.price(S0=P["S0"], K=P["K"], r=P["r"], sigma=P["sigma"], T=P["T"],
                           num_uncertainty_qubits=3, epsilon_target=eps, shots=SHOTS, seed=k)
            return r["num_oracle_queries"], r["price"] - r["model_exact_price"]
        q, rms = _avg(one)
        if q > 0:
            x.append(q); y.append(rms)
    series["Greek SOTA oracle-QAE"] = (x, y)
    print("Greek done")

    json.dump({k: {"queries": xs, "rms": ys} for k, (xs, ys) in series.items()},
              open(JS, "w"), indent=1)

    # ---- plot ----
    colmap = {"Monte Carlo": style.PALETTE["classical"],
              "our QNDM-QAE": style.PALETTE["quantum"],
              "our QSVT": style.PALETTE["accent"],
              "Greek SOTA oracle-QAE": style.PALETTE["muted"]}
    mark = {"Monte Carlo": "s", "our QNDM-QAE": "o", "our QSVT": "^",
            "Greek SOTA oracle-QAE": "D"}
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for name, (xs, ys) in series.items():
        if not xs:
            continue
        xs, ys = np.array(xs, float), np.array(ys, float)
        s = np.polyfit(np.log(xs), np.log(ys), 1)[0] if len(xs) >= 2 else float("nan")
        ax.loglog(xs, ys, mark[name] + "-", color=colmap[name], markersize=8,
                  label=f"{name}  (slope {s:+.2f})")

    # reference slope guides -1 (quantum) and -1/2 (classical)
    allx = np.concatenate([np.array(v[0], float) for v in series.values() if v[0]])
    ally = np.concatenate([np.array(v[1], float) for v in series.values() if v[1]])
    x0, x1 = allx.min(), allx.max(); y0 = ally.max()
    for sl, lab in ((-1.0, "ideal quantum 1/N"), (-0.5, "ideal classical 1/sqrt(N)")):
        xg = np.array([x0, x1]); yg = y0 * (xg / x0) ** sl
        ax.loglog(xg, yg, ":", color=style.PALETTE["ink"], lw=0.8, alpha=0.5)
        ax.text(x1, yg[-1], " " + lab, fontsize=7.5, color=style.PALETTE["ink"],
                va="center", alpha=0.6)

    ax.set_xlabel("oracle queries (QAE/QSVT/Greek)  /  samples (MC)")
    ax.set_ylabel("estimation error  (vs each method's own exact target)")
    ax.set_title("Empirical speedup: error vs queries -- Monte Carlo vs Greek-SOTA "
                 "vs our QNDM-QAE / QSVT")
    ax.legend(loc="lower left", fontsize=9)
    style.caption(fig, "Quantum amplitude-estimation routes (QAE, QSVT, Greek-SOTA) descend ~1/N "
                       "(slope ~ -1); Monte Carlo ~1/sqrt(N) (slope ~ -0.5) -- the quadratic "
                       "speedup. QSVT flattens at its polynomial-degree bias floor; the Greek route "
                       "at its rescaling-linearisation floor. Finite shots (100), seed-averaged.")
    style.provenance(fig, "quantum_pricer qae/qsvt/sota + classical MC; NOKIA.HE; M=3")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)
    for name, (xs, ys) in series.items():
        print(f"  {name}: {len(xs)} pts")
    print("[saved]", FIG)


if __name__ == "__main__":
    main()
