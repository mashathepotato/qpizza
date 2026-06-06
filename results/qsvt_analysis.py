"""Focused QSVT analysis for the paper/demo. Produces four figures:

  1. error_scaling_qsvt.png   -- error vs queries: classical MC vs Greek-SOTA vs our
     QSVT (REUSES results/speedup_compare.json -- no re-run).
  2. qsvt_shots_convergence.png -- QSVT convergence vs #shots, BOTH senses:
       (a) direct Pr[s=0] sampling -> statevector value;
       (b) full IterativeAmplitudeEstimation pipeline.
  3. qsvt_value.png           -- discounted price e^{-rT} E[max(f-K,0)] for the three
     methods, EACH vs its own exact target (tree for MC/QSVT, lognormal for Greek).
  4. qsvt_qubit_scaling.png   -- measured qubit count of the QSVT route vs M (and vs
     degree), showing qubits = M+1, independent of accuracy (no price register).

Run (pricer venv):
  PYTHONPATH=<repo-root> quantum_pricer/.venv/bin/python results/qsvt_analysis.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from results import style
from quantum_pricer import tree, classical, qsvt, sota, backends

P = dict(S0=13.09, K=13.09, r=0.03, sigma=0.479, T=1.0)
M = 3
DEGREE = 60
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
DATA_OUT = os.path.join(os.path.dirname(__file__), "qsvt_analysis.json")
SPEEDUP_JSON = os.path.join(os.path.dirname(__file__), "speedup_compare.json")


def _fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0]) if len(xs) >= 2 else float("nan")


# ---- shared QSVT instrument (built once) --------------------------------------------

def build_qsvt_instrument():
    angles = tree.loading_angles(M=M, **P)
    values = tree.payoff_variable_values(M=M, option="european", **P)
    K = P["K"]
    Cmax = float(np.max(np.abs(values - K))) or 1.0
    c = 1.0 / Cmax
    coef = qsvt.straddle_poly(degree=DEGREE)
    phis, used_pyqsp = qsvt.qsp_phases(coef)
    w0, kappa, probe_resid = qsvt._probe_transfer_function(phis)
    qc, s = qsvt.build_qsvt_prep(angles, values, K, c, phis)
    F = qsvt.forward_price(S0=P["S0"], r=P["r"], T=P["T"], M=M, option="european")
    a_exact = float(Statevector(qc).probabilities([s])[0])
    return dict(qc=qc, s=s, Cmax=Cmax, w0=w0, kappa=kappa, F=F, K=K,
                a_exact=a_exact, probe_resid=probe_resid, used_pyqsp=used_pyqsp)


def price_from_a(a, inst):
    """The HONEST scaling chain: a -> E[|f-K|] -> call by put-call parity."""
    E_abs = inst["Cmax"] * (a - inst["w0"]) / inst["kappa"]
    return float(np.exp(-P["r"] * P["T"]) * (E_abs + (inst["F"] - inst["K"])) / 2.0)


# ====================================================================================
# Figure 1 -- error scaling (reuse speedup_compare.json)
# ====================================================================================

def greek_series(rescaling, eps_grid=(0.12, 0.06, 0.03, 0.015, 0.008), seeds=3, shots=100):
    """Greek SOTA error (vs its OWN discretised-lognormal target) vs oracle queries,
    at a chosen payoff-linearisation rescaling_factor. Seed-averaged RMS."""
    xs, ys = [], []
    for eps in eps_grid:
        qs, es = [], []
        for k in range(seeds):
            r = sota.price(S0=P["S0"], K=P["K"], r=P["r"], sigma=P["sigma"], T=P["T"],
                           num_uncertainty_qubits=3, rescaling_factor=rescaling,
                           epsilon_target=eps, shots=shots, seed=k)
            qs.append(r["num_oracle_queries"]); es.append(r["price"] - r["model_exact_price"])
        xs.append(float(np.mean(qs))); ys.append(float(np.sqrt(np.mean(np.square(es)))))
    return np.array(xs), np.array(ys)


def fig_error_scaling():
    j = json.load(open(SPEEDUP_JSON))
    # reused (no re-run): MC, our QSVT, and the default-rescaling (0.25) Greek run
    reuse = {"Monte Carlo": ("classical", "s", "-"),
             "our QSVT": ("quantum", "^", "-"),
             "Greek SOTA oracle-QAE": ("muted", "D", "-")}
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    out = {}
    for name, (colkey, mk, ls) in reuse.items():
        xs = np.array(j[name]["queries"], float); ys = np.array(j[name]["rms"], float)
        ok = (xs > 0) & (ys > 0); xs, ys = xs[ok], ys[ok]
        s = _fit(xs, ys)
        lab = (f"Greek SOTA  rescaling=0.25 (slope {s:+.2f}, bias-limited)"
               if name.startswith("Greek") else f"{name}  (slope {s:+.2f})")
        out[name] = dict(queries=xs.tolist(), rms=ys.tolist(), slope=s)
        ax.loglog(xs, ys, mk + ls, color=style.PALETTE[colkey], markersize=8,
                  lw=1.6, label=lab, alpha=0.95)

    # FRESH: Greek with a tuned rescaling so the bias floor drops below the eps range
    gx, gy = greek_series(rescaling=0.10)
    gs = _fit(gx, gy)
    out["Greek SOTA (rescaling 0.10, tuned)"] = dict(queries=gx.tolist(),
                                                     rms=gy.tolist(), slope=gs)
    ax.loglog(gx, gy, "D--", color=style.PALETTE["accent"], markersize=8, lw=1.8,
              label=f"Greek SOTA  rescaling=0.10 (slope {gs:+.2f}, ~ideal)")

    allx = np.concatenate([np.array(j[n]["queries"], float) for n in reuse] + [gx])
    ally = np.concatenate([np.array(j[n]["rms"], float) for n in reuse] + [gy])
    x0, x1, y0 = allx.min(), allx.max(), ally.max()
    for sl, lab in ((-1.0, "ideal quantum  1/N"), (-0.5, "classical  1/√N")):
        xg = np.array([x0, x1]); yg = y0 * (xg / x0) ** sl
        ax.loglog(xg, yg, ":", color=style.PALETTE["ink"], lw=0.8, alpha=0.5)
        ax.text(x1, yg[-1], " " + lab, fontsize=8, color=style.PALETTE["ink"],
                va="center", alpha=0.6)
    ax.set_xlabel("oracle queries (QSVT / Greek)  /  samples (MC)")
    ax.set_ylabel("estimation error  (vs each method's own exact target)")
    ax.set_title("Error scaling: classical MC vs Greek-SOTA vs our QSVT")
    ax.legend(loc="lower left", fontsize=9.0)
    style.caption(fig, "MC descends ~1/√N (slope -0.5); amplitude-estimation routes ~1/N "
                       "(slope -1). The Greek SOTA's shallow -0.4 is NOT a failed speed-up: "
                       "at the textbook rescaling 0.25 its FIXED O(c²) payoff-linearisation "
                       "bias (~0.18) dominates the whole eps range. Dropping rescaling to "
                       "0.10 pushes that floor below eps and the true -1 quadratic scaling "
                       "reappears. Our QSVT flattens at its degree-%d straddle floor." % DEGREE)
    style.provenance(fig, "MC/QSVT/Greek@0.25 reused from speedup_compare.json; "
                          "Greek@0.10 fresh; NOKIA.HE; M=3")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(FIGDIR, "error_scaling_qsvt.png")
    fig.savefig(out_path); plt.close(fig)
    print("[saved]", out_path)
    return out


# ====================================================================================
# Figure 2 -- QSVT convergence vs shots (both senses)
# ====================================================================================

def _sample_a(qc, s, n_shots, seed, sim):
    qcm = qc.copy(); qcm.measure_all()
    qcm = transpile(qcm, sim)
    counts = sim.run(qcm, shots=n_shots, seed_simulator=seed).result().get_counts()
    zero = sum(n for b, n in counts.items() if b[::-1][s] == "0")
    return zero / n_shots


def fig_shots_convergence(inst):
    exact = tree.exact_tree_price(M=M, option="european", kind="call", **P)
    sv_price = price_from_a(inst["a_exact"], inst)          # statevector (bias floor)
    floor = abs(sv_price - exact)
    sim = AerSimulator()

    # (a) direct Pr[s=0] sampling
    SHOTS_A = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    SEEDS_A = 16
    a_err = []
    for n in SHOTS_A:
        errs = [abs(price_from_a(_sample_a(inst["qc"], inst["s"], n, k, sim), inst) - exact)
                for k in range(SEEDS_A)]
        a_err.append(float(np.sqrt(np.mean(np.square(errs)))))
        print(f"  [shots-direct] N={n:>5}  rms_err={a_err[-1]:.4e}")

    # (b) full IterativeAmplitudeEstimation pipeline
    SHOTS_B = [64, 256, 1024, 4096]
    SEEDS_B = 4
    b_err, b_q = [], []
    for n in SHOTS_B:
        errs, qs = [], []
        for k in range(SEEDS_B):
            r = qsvt.price(M=M, option="european", kind="call", degree=DEGREE,
                           use_qae=True, epsilon_target=0.01, shots=n, seed=k,
                           return_meta=True, **P)
            errs.append(abs(r["price"] - exact)); qs.append(r["num_oracle_queries"])
        b_err.append(float(np.sqrt(np.mean(np.square(errs)))))
        b_q.append(float(np.mean(qs)))
        print(f"  [shots-IAE]    N={n:>5}  rms_err={b_err[-1]:.4e}  q~{b_q[-1]:.0f}")

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.loglog(SHOTS_A, np.maximum(a_err, 1e-6), "o-", color=style.PALETTE["quantum"],
              markersize=7, lw=1.6, label="(a) direct Pr[s=0] sampling")
    ax.loglog(SHOTS_B, np.maximum(b_err, 1e-6), "^-", color=style.PALETTE["accent"],
              markersize=8, lw=1.6, label="(b) full IAE pipeline (ε=0.01)")
    # 1/sqrt(N) guide anchored to the first direct point
    xg = np.array([SHOTS_A[0], SHOTS_A[-1]], float)
    yg = a_err[0] * (xg / SHOTS_A[0]) ** -0.5
    ax.loglog(xg, yg, ":", color=style.PALETTE["ink"], lw=0.9, alpha=0.55,
              label="1/√N guide")
    ax.axhline(floor, ls="--", color=style.PALETTE["classical"], lw=1.3,
               label=f"degree-{DEGREE} bias floor = {floor:.3f}")
    ax.set_xlabel("number of shots  N")
    ax.set_ylabel("|price − exact tree|  (RMS over seeds)")
    ax.set_title(f"QSVT convergence vs shots  (M={M}, degree={DEGREE})")
    ax.legend(loc="upper right", fontsize=9.5)
    amp = inst["Cmax"] / inst["kappa"]
    style.caption(fig, "Both senses of 'shots'. (a) Direct signal-qubit sampling falls as "
                       "1/√N but the straddle readout amplifies sampling error by "
                       f"C_max/κ ≈ {amp:.0f} (price_err ≈ {amp/2:.0f}·δa), so it is still "
                       f"~9× above the floor at 8192 shots (needs ~10^6 to reach it). "
                       "(b) Grover-based Iterative AE reaches the fixed degree-"
                       f"{DEGREE} bias floor (dashed) with ~10²–10³ shots/round — this gap "
                       "is exactly why amplitude estimation is used instead of sampling.")
    style.provenance(fig, f"quantum_pricer.qsvt on NOKIA.HE; M={M}; deg={DEGREE}; real runs")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(FIGDIR, "qsvt_shots_convergence.png")
    fig.savefig(out_path); plt.close(fig)
    print("[saved]", out_path)
    return dict(floor=floor, exact=exact, statevector_price=sv_price,
                direct=dict(shots=SHOTS_A, rms=a_err, seeds=SEEDS_A),
                iae=dict(shots=SHOTS_B, rms=b_err, queries=b_q, seeds=SEEDS_B))


# ====================================================================================
# Figure 3 -- discounted value e^{-rT} E[max(f-K,0)], each vs its own target
# ====================================================================================

def fig_value(inst):
    tree_exact = tree.exact_tree_price(M=M, option="european", kind="call", **P)
    mc, mc_se = classical.monte_carlo_price(M=M, n_paths=50_000, option="european",
                                            kind="call", seed=1, **P)
    qsvt_sv = price_from_a(inst["a_exact"], inst)
    g = sota.price(S0=P["S0"], K=P["K"], r=P["r"], sigma=P["sigma"], T=P["T"],
                   num_uncertainty_qubits=3, epsilon_target=0.01, shots=4096, seed=7)

    rows = [
        ("Classical MC",  mc,       tree_exact,            "exact tree",        "classical"),
        ("our QSVT",      qsvt_sv,  tree_exact,            "exact tree",        "quantum"),
        ("Greek SOTA",    g["price"], g["model_exact_price"], "own lognormal",  "muted"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    xs = np.arange(len(rows))
    for i, (name, val, tgt, tlab, ck) in enumerate(rows):
        ax.bar(i, val, width=0.55, color=style.PALETTE[ck], edgecolor=style.PALETTE["ink"],
               zorder=2)
        # each method's OWN target as a horizontal tick on its bar
        ax.plot([i - 0.32, i + 0.32], [tgt, tgt], color=style.PALETTE["ink"], lw=2.2,
                zorder=3)
        err = val - tgt
        ax.annotate(f"{val:.4f}\n({err:+.4f} vs {tlab})", (i, val),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=9.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel(r"discounted option price  $e^{-rT}\,\mathbb{E}[\max(f-K,0)]$  (EUR)")
    ax.set_title("Estimated value per method, each vs its OWN exact target  (M=%d)" % M)
    ax.set_ylim(0, max(r[1] for r in rows) * 1.25)
    # legend for the target tick
    ax.plot([], [], color=style.PALETTE["ink"], lw=2.2, label="method's own exact target")
    ax.legend(loc="upper right", fontsize=9)
    style.caption(fig, "Horizontal tick = each method's OWN ground truth: the binomial "
                       "tree for MC/QSVT (≈%.3f), the discretised lognormal for Greek-"
                       "SOTA (≈%.3f). The Greek target is a DIFFERENT instrument, so its "
                       "value is not directly comparable to ours — only each error-to-own-"
                       "target is meaningful." % (tree_exact, g["model_exact_price"]))
    style.provenance(fig, "quantum_pricer mc/qsvt/sota on NOKIA.HE; M=%d; QSVT deg=%d; "
                          "Greek n_p=3" % (M, DEGREE))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(FIGDIR, "qsvt_value.png")
    fig.savefig(out_path); plt.close(fig)
    print("[saved]", out_path)
    return dict(tree_exact=tree_exact, mc=mc, mc_stderr=mc_se, qsvt=qsvt_sv,
                greek_price=g["price"], greek_target=g["model_exact_price"])


# ====================================================================================
# Figure 4 -- QSVT qubit scaling (measured)
# ====================================================================================

def fig_qubit_scaling():
    Ms = list(range(2, 13))
    coef = qsvt.straddle_poly(degree=DEGREE)
    phis, _ = qsvt.qsp_phases(coef)
    qubits = []
    for m in Ms:
        angles = tree.loading_angles(M=m, **P)
        values = tree.payoff_variable_values(M=m, option="european", **P)
        Cmax = float(np.max(np.abs(values - P["K"]))) or 1.0
        qc, _ = qsvt.build_qsvt_prep(angles, values, P["K"], 1.0 / Cmax, phis)
        qubits.append(int(qc.num_qubits))
    # also: vary degree at fixed M to show qubit count is independent of accuracy
    m_fix = 6
    degs = [20, 40, 60, 100, 160]
    q_vs_deg = []
    angles = tree.loading_angles(M=m_fix, **P)
    values = tree.payoff_variable_values(M=m_fix, option="european", **P)
    Cmax = float(np.max(np.abs(values - P["K"]))) or 1.0
    for d in degs:
        c2 = qsvt.straddle_poly(degree=d)
        ph2, _ = qsvt.qsp_phases(c2)
        qc, _ = qsvt.build_qsvt_prep(angles, values, P["K"], 1.0 / Cmax, ph2)
        q_vs_deg.append(int(qc.num_qubits))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.plot(Ms, qubits, "o-", color=style.PALETTE["quantum"], markersize=8, lw=1.8,
             label="measured QSVT circuit width")
    ax1.plot(Ms, [m + 1 for m in Ms], ":", color=style.PALETTE["ink"], lw=1.2,
             label="$M+1$  (M path qubits + 1 signal ancilla)")
    ax1.set_xlabel("number of time steps  M")
    ax1.set_ylabel("qubits in QSVT state-preparation")
    ax1.set_title("QSVT qubit count vs M")
    ax1.legend(loc="upper left", fontsize=9.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(degs, q_vs_deg, "s-", color=style.PALETTE["accent"], markersize=8, lw=1.8)
    ax2.set_xlabel("polynomial degree  d  (↔ accuracy ε)")
    ax2.set_ylabel(f"qubits  (M={m_fix} fixed)")
    ax2.set_title("Qubit count is INDEPENDENT of degree / ε")
    ax2.set_ylim(0, m_fix + 3)
    ax2.grid(True, alpha=0.3)
    ax2.annotate(f"flat at {m_fix + 1} qubits\n(no price register)",
                 (degs[len(degs) // 2], q_vs_deg[len(degs) // 2]),
                 textcoords="offset points", xytext=(0, -38), ha="center", fontsize=10,
                 color=style.PALETTE["ink"])

    fig.suptitle("QSVT route: qubit scaling  (degree adds DEPTH, never WIDTH)",
                 fontsize=13, fontweight="bold")
    style.caption(fig, "The QSVT route uses exactly $M+1$ qubits: $M$ path qubits "
                       "(one R_Y per step) plus a single signal ancilla. Increasing the "
                       "polynomial degree d (hence accuracy) adds controlled-oracle CALLS "
                       "(depth), not qubits — there is no price-precision register, and "
                       "Iterative AE reuses the one ancilla (no QPE eval register).")
    style.provenance(fig, "measured quantum_pricer.qsvt.build_qsvt_prep widths; NOKIA.HE")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(FIGDIR, "qsvt_qubit_scaling.png")
    fig.savefig(out_path); plt.close(fig)
    print("[saved]", out_path)
    return dict(M=Ms, qubits=qubits, formula="M+1",
                degree_sweep=dict(M=m_fix, degrees=degs, qubits=q_vs_deg))


def fig_qsvt_resources():
    """Resources of the QSVT state-preparation as the qubit count (= M+1) grows,
    transpiled to the IQM {r, cz} native basis. Width is linear but the dense 2^M
    Diagonal phase oracle makes the gate cost exponential."""
    Ms = list(range(2, 10))
    coef = qsvt.straddle_poly(degree=DEGREE)
    phis, _ = qsvt.qsp_phases(coef)
    n_oracle_calls = len(phis) - 1
    qubits, cz, rgate, depth = [], [], [], []
    for m in Ms:
        ang = tree.loading_angles(M=m, **P)
        val = tree.payoff_variable_values(M=m, option="european", **P)
        Cmax = float(np.max(np.abs(val - P["K"]))) or 1.0
        qc, _ = qsvt.build_qsvt_prep(ang, val, P["K"], 1.0 / Cmax, phis)
        tq = transpile(qc, basis_gates=backends.IQM_BASIS, optimization_level=1)
        ops = tq.count_ops()
        qubits.append(int(qc.num_qubits)); cz.append(int(ops.get("cz", 0)))
        rgate.append(int(ops.get("r", 0))); depth.append(int(tq.depth()))
        print(f"  [qsvt-res] M={m} qubits={qubits[-1]} cz={cz[-1]} r={rgate[-1]} depth={depth[-1]}")

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.semilogy(qubits, cz, "o-", color=style.PALETTE["quantum"], markersize=8, lw=1.8,
                label="CZ count (IQM {r,cz})")
    ax.semilogy(qubits, rgate, "s-", color=style.PALETTE["accent"], markersize=7, lw=1.6,
                label="single-qubit r-gate count")
    ax.semilogy(qubits, depth, "^--", color=style.PALETTE["muted"], markersize=7, lw=1.4,
                label="transpiled depth")
    # exponential reference anchored to the first CZ point: ~2^M = 2^(qubits-1)
    qq = np.array(qubits, float)
    ref = cz[0] * 2.0 ** (qq - qubits[0])
    ax.semilogy(qubits, ref, ":", color=style.PALETTE["classical"], lw=1.2,
                label="$\\propto 2^{M}$ reference")
    ax.set_xlabel("number of qubits  (= M + 1)")
    ax.set_ylabel("native-gate resources  (log)")
    ax.set_title(f"QSVT route resources vs qubits  (degree {DEGREE}, "
                 f"{n_oracle_calls} controlled-oracle calls)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True, which="both", alpha=0.3)
    style.caption(fig, "Width is linear (M+1 qubits), but the gate cost grows ~2^M: the "
                       "current QSVT route builds the DENSE 2^M Diagonal phase oracle "
                       "(qsvt.py), so CZ ≈ (degree)·O(2^M). Swapping in the Hamming-weight "
                       "poly(M) phase oracle (see depth_crossover.png) would make this "
                       "polynomial in M — the qubit count would not change.")
    style.provenance(fig, f"measured transpile to IQM {{r,cz}}; quantum_pricer.qsvt; "
                          f"degree={DEGREE}; NOKIA.HE")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(FIGDIR, "qsvt_resources_vs_qubits.png")
    fig.savefig(out_path); plt.close(fig)
    print("[saved]", out_path)
    # log-log slope of CZ vs 2^M to confirm exponential
    cz_slope_per_qubit = float(np.polyfit(qq, np.log2(cz), 1)[0])
    return dict(M=Ms, qubits=qubits, cz=cz, r=rgate, depth=depth,
                n_oracle_calls=n_oracle_calls,
                log2_cz_slope_per_qubit=cz_slope_per_qubit)


def main():
    style.apply_style()
    os.makedirs(FIGDIR, exist_ok=True)
    print("building QSVT instrument (M=%d, degree=%d) ..." % (M, DEGREE))
    inst = build_qsvt_instrument()
    print(f"  a_exact={inst['a_exact']:.5f}  w0={inst['w0']:.4f}  kappa={inst['kappa']:.4f}"
          f"  probe_resid={inst['probe_resid']:.3e}  pyqsp={inst['used_pyqsp']}")
    data = {}
    print("\n[1/4] error scaling (reuse speedup_compare.json)")
    data["error_scaling"] = fig_error_scaling()
    print("\n[2/4] shots convergence (fresh runs)")
    data["shots_convergence"] = fig_shots_convergence(inst)
    print("\n[3/4] value per method (fresh)")
    data["value"] = fig_value(inst)
    print("\n[4/5] qubit scaling (measured)")
    data["qubit_scaling"] = fig_qubit_scaling()
    print("\n[5/5] QSVT resources vs qubits (measured transpile)")
    data["qsvt_resources"] = fig_qsvt_resources()
    json.dump(data, open(DATA_OUT, "w"), indent=1, default=float)
    print("\n[saved]", DATA_OUT)


if __name__ == "__main__":
    main()
