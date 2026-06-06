"""3-way benchmark: classical MC vs QNDM-QAE vs novel QSVT.

Outputs (1) error-vs-queries rows for the log-log speed-up plot and (2) an empirical
resource table (qubits, CZ depth after transpile to IQM {r,cz}).

Ground truth for every error is the EXACT TREE PRICE (`tree.exact_tree_price`), NOT
Black-Scholes -- that is the quantity the quantum routes actually estimate.

HONESTY (do not paper over these in the deck):
  * QAE / IterativeAmplitudeEstimation SATURATES its Grover-power schedule at small M:
    coarse epsilon_target values converge at Grover power 0 (0 queries) and finer ones
    jump to a single large fixed query count, so several distinct epsilon targets report
    the SAME number of queries. Rows where this happens carry note="qae_query_saturation".
    Use a slightly larger M (4-5) and a fine epsilon sweep to expose >1 distinct quantum
    point; we still report the saturation honestly rather than fabricating a smooth curve.
  * QSVT carries a polynomial-APPROXIMATION ERROR FLOOR independent of queries (the
    degree-`qsvt_degree` straddle-polynomial residual near the kink, ~1-2% at degree 60).
    Its error-vs-queries points plateau at that floor instead of dropping to zero; such
    rows carry note="qsvt_approx_floor". This is expected and honest, not a bug.
"""
import numpy as np
from qiskit import transpile
from quantum_pricer import classical, qae, oracles, tree, backends

try:
    from quantum_pricer import qsvt as _qsvt
    _HAS_QSVT = True
except Exception:  # pragma: no cover - qsvt import is expected to succeed
    _HAS_QSVT = False


def error_vs_queries(S0, K, r, sigma, T, M, option="european", kind="call",
                     mc_budgets=(10**3, 10**4, 10**5, 10**6),
                     qae_eps=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005),
                     qsvt_eps=(0.1, 0.05, 0.02, 0.01),
                     qsvt_degree=60, include_qsvt=True):
    """Return a list of row dicts {method, queries, abs_error, note}.

    queries: classical MC -> n_paths sampled; quantum -> oracle queries reported by IAE
    (Grover-power schedule, so it can stay 0 for coarse eps then jump -- see module docs).
    """
    exact = tree.exact_tree_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                  option=option, kind=kind)
    rows = []

    # classical Monte Carlo: honest O(1/sqrt(N)) -> O(1/eps^2) statistical error
    for n in mc_budgets:
        p, _ = classical.monte_carlo_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                           option=option, kind=kind, n_paths=n, seed=0)
        rows.append(dict(method="classical_mc", queries=int(n),
                         abs_error=abs(p - exact), note=""))

    # QNDM amplitude-QAE: O(1/eps) queries, but saturates at small M (annotated)
    seen_q = {}
    for eps in qae_eps:
        res = qae.price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, option=option,
                        kind=kind, epsilon_target=eps)
        q = int(res["num_oracle_queries"])
        note = "qae_query_saturation" if q in seen_q else ""
        seen_q[q] = seen_q.get(q, 0) + 1
        rows.append(dict(method="qae", queries=q,
                         abs_error=abs(res["price"] - exact), note=note))

    # novel QSVT: queries grow but error plateaus at the polynomial-approx floor (annotated)
    if include_qsvt and _HAS_QSVT:
        try:
            sv_only = _qsvt.price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, option=option,
                                  kind=kind, degree=qsvt_degree, use_qae=False)
            approx_floor = abs(sv_only - exact)
            for eps in qsvt_eps:
                res = _qsvt.price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, option=option,
                                  kind=kind, degree=qsvt_degree, use_qae=True,
                                  epsilon_target=eps, return_meta=True)
                err = abs(res["price"] - exact)
                # flag rows sitting at the (query-independent) approximation floor
                note = "qsvt_approx_floor" if err <= approx_floor * 1.2 + 1e-9 else ""
                rows.append(dict(method="qsvt",
                                 queries=int(res["num_oracle_queries"]),
                                 abs_error=err, note=note))
        except Exception as exc:  # degrade gracefully if QSVT/pyqsp misbehaves
            rows.append(dict(method="qsvt", queries=0, abs_error=float("nan"),
                             note=f"qsvt_unavailable: {exc}"))
    return rows


def _cz_depth(qc):
    t = transpile(qc, basis_gates=backends.IQM_BASIS, optimization_level=1)
    return t.depth(lambda instr: instr.operation.name == "cz")


def resource_table(S0, K, r, sigma, T, M, option="european", kind="call",
                   qsvt_degree=20):
    """Empirical resource table: qubits + CZ depth after transpile to IQM {r,cz}."""
    angles = tree.loading_angles(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    values = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                         option=option)
    payoff = np.maximum(values - K, 0.0)
    Cmax = float(payoff.max()) * 1.0001 if payoff.max() > 0 else 1.0
    fc = oracles.fourier_circuit(M, angles, values, K, lam=1.0, basis="X")
    ac, _ = oracles.payoff_amplitude_circuit(angles, payoff, Cmax)
    table = [
        dict(method="classical_mc", qubits=0, cz_depth=0),
        dict(method="fourier", qubits=fc.num_qubits, cz_depth=_cz_depth(fc)),
        dict(method="qae", qubits=ac.num_qubits, cz_depth=_cz_depth(ac)),
    ]
    if _HAS_QSVT:
        try:
            coef = _qsvt.straddle_poly(degree=qsvt_degree)
            phis, _ = _qsvt.qsp_phases(coef)
            c = 1.0 / max(np.max(np.abs(np.asarray(values, float) - K)), 1e-9)
            qc, _ = _qsvt.build_qsvt_prep(angles, values, K, c, phis)
            table.append(dict(method="qsvt", qubits=qc.num_qubits,
                              cz_depth=_cz_depth(qc)))
        except Exception:  # pragma: no cover - graceful QSVT degrade
            pass
    return table


def save_speedup_plot(rows, path="quantum_pricer/speedup.png"):
    """Write a log-log error-vs-queries PNG. queries==0 points are dropped (log axis);
    NaN errors (failed routes) are skipped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in sorted({r["method"] for r in rows}):
        pts = sorted((r["queries"], r["abs_error"]) for r in rows
                     if r["method"] == method
                     and r["queries"] > 0 and np.isfinite(r["abs_error"])
                     and r["abs_error"] > 0)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, "o-", label=method)
    ax.set_xlabel("queries / samples")
    ax.set_ylabel("absolute price error (vs exact tree)")
    ax.set_title("Error vs queries: classical 1/sqrt(N) vs quantum 1/N")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
