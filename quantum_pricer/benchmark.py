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
from quantum_pricer import classical, qae, oracles, tree, backends, hamming, fourier

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


def error_vs_queries_rms(S0, K, r, sigma, T, M, option="european", kind="call",
                         mc_budgets=(250, 1000, 4000, 16000, 64000, 256000),
                         qae_eps=(0.1, 0.05, 0.025, 0.012, 0.006, 0.003),
                         qae_shots=100, seeds=8):
    """Seed-averaged empirical RMS-error descent for the honest Figure 2.

    Unlike :func:`error_vs_queries` (single seed, statevector-exact QAE -> flat),
    this AVERAGES error over ``seeds`` random seeds at each budget so the descent is
    clean and the empirical slopes are visible:

      * classical MC: RMS error ~ 1/sqrt(N)  -> log-log slope ~ -1/2.
      * QAE (FINITE shots): a finite shot budget (``qae_shots``) makes IAE actually
        iterate Grover rounds, so estimation error is real (not machine-zero). RMS
        error ~ 1/queries -> log-log slope ~ -1.

    Ground truth is the EXACT TREE PRICE. Returns row dicts with keys
    {method, budget_x, rms_error, n_seeds, note}; budget_x is N for MC and the mean
    ``num_oracle_queries`` for QAE.
    """
    exact = tree.exact_tree_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                  option=option, kind=kind)
    rows = []

    # classical Monte Carlo: RMS error over `seeds` seeds at each sample count N
    for n in mc_budgets:
        errs = []
        for k in range(seeds):
            p, _ = classical.monte_carlo_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                               option=option, kind=kind,
                                               n_paths=int(n), seed=k)
            errs.append(p - exact)
        rms = float(np.sqrt(np.mean(np.square(errs))))
        rows.append(dict(method="classical_mc", budget_x=float(n), rms_error=rms,
                         n_seeds=seeds, note="rms_over_seeds"))

    # QAE: finite-shot Sampler so estimation error is genuine; RMS over `seeds` seeds
    qae_x, qae_y = [], []
    for eps in qae_eps:
        qs, errs = [], []
        for k in range(seeds):
            res = qae.price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, option=option,
                            kind=kind, epsilon_target=eps, shots=qae_shots, seed=k)
            qs.append(int(res["num_oracle_queries"]))
            errs.append(res["price"] - exact)
        mean_q = float(np.mean(qs))
        rms = float(np.sqrt(np.mean(np.square(errs))))
        rows.append(dict(method="qae", budget_x=mean_q, rms_error=rms,
                         n_seeds=seeds, note="finite_shots_rms"))
        if mean_q > 0:
            qae_x.append(mean_q)
            qae_y.append(rms)

    # Honesty fallback: if the IAE Grover-power schedule SATURATED (fewer than 3
    # distinct positive query levels even at finite shots) we cannot claim an
    # empirical descent -- overlay the theoretical pi/(2 eps) line (slope ~ -1) and
    # flag the QAE rows so the plot/legend report it instead of faking a curve.
    distinct = {round(x) for x in qae_x}
    if len(distinct) < 3:
        for row in rows:
            if row["method"] == "qae":
                row["note"] = "qae_saturated_theory"
        for eps in qae_eps:
            rows.append(dict(method="qae", budget_x=float(np.pi / (2.0 * eps)),
                             rms_error=float(eps), n_seeds=0,
                             note="qae_saturated_theory"))

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


def payoff_std(S0, K, r, sigma, T, M, option="european", kind="call"):
    """EXACT std of the discounted payoff under the risk-neutral path distribution.

    sigma_payoff^2 = sum_x p(x) (d*payoff(x))^2 - (sum_x p(x) d*payoff(x))^2,
    d = exp(-rT). Computed analytically from the full tree enumeration -- this is the
    population std that drives the classical CLT, NOT an estimate from running MC.
    """
    vals = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                       option=option)
    p = tree.path_probabilities(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    if kind == "call":
        payoff = np.maximum(vals - K, 0.0)
    elif kind == "put":
        payoff = np.maximum(K - vals, 0.0)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    d = np.exp(-r * T)
    disc = d * payoff
    mean = float(np.sum(p * disc))
    var = float(np.sum(p * disc ** 2) - mean ** 2)
    return float(np.sqrt(max(var, 0.0)))


def queries_to_accuracy(S0, K, r, sigma, T, M, option="european", kind="call",
                        epsilons=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005)):
    """Queries/samples each method needs to REACH each target accuracy eps.

    Shows SCALING (not noisy point-errors). Returns rows with keys
    {method, epsilon, queries, kind in {'empirical','theoretical'}, note}:

      * classical_mc (kind='theoretical'): analytic CLT sample complexity
        N = (sigma_payoff / eps)^2, sigma_payoff computed exactly from the tree.
        These are ANALYTIC -- we do not run MC to get them.
      * qae (kind='empirical'): num_oracle_queries actually reported by IAE at
        epsilon_target=eps. At small M IAE's Grover-power schedule SATURATES: once
        the query count stops growing as eps shrinks, those rows carry
        note='qae_saturated' (a real small-M simulator artifact, reported honestly).
      * qae (kind='theoretical'): the canonical amplitude-estimation query count
        N_QAE(eps) ~= pi / (2 eps), a 1/eps reference line so the quadratic-vs-linear
        scaling is visible even where the simulator saturates.
    """
    sigma_payoff = payoff_std(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                              option=option, kind=kind)
    rows = []

    # classical MC: analytic CLT sample complexity (1/eps^2 scaling)
    for eps in epsilons:
        n = (sigma_payoff / eps) ** 2
        rows.append(dict(method="classical_mc", epsilon=float(eps),
                         queries=float(n), kind="theoretical", note="analytic_clt"))

    # QAE empirical: oracle queries IAE actually spends (can saturate at small M)
    last_q = None
    saturated = False
    for eps in epsilons:
        res = qae.price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, option=option,
                        kind=kind, epsilon_target=eps)
        q = int(res["num_oracle_queries"])
        # once finer eps no longer buys more queries, the schedule has saturated
        if last_q is not None and q <= last_q:
            saturated = True
        note = "qae_saturated" if saturated else ""
        rows.append(dict(method="qae", epsilon=float(eps), queries=float(q),
                         kind="empirical", note=note))
        last_q = q

    # QAE theoretical: canonical N_QAE(eps) ~= pi/(2 eps) reference line (1/eps slope)
    for eps in epsilons:
        n = np.pi / (2.0 * eps)
        rows.append(dict(method="qae", epsilon=float(eps), queries=float(n),
                         kind="theoretical", note="theory_pi_over_2eps"))

    return rows


def _fig_style():
    """Apply the shared repo style (results/style.py); degrade gracefully when the
    module is vendored without it. Returns (PALETTE, caption_fn, provenance_fn)."""
    try:
        from results import style as _style
        _style.apply_style()
        return _style.PALETTE, _style.caption, _style.provenance
    except Exception:  # pragma: no cover - standalone fallback
        palette = dict(quantum="#2a6f97", classical="#c44536", accent="#2a9d8f",
                       muted="#8d99ae", ink="#22223b", grid="#d7d9e0")
        return palette, (lambda fig, text: None), (lambda fig, text: None)


_PROVENANCE = "qpizza quantum_pricer.benchmark — ground truth: exact CRR tree"


def save_complexity_plot(rows, path="quantum_pricer/complexity.png"):
    """Log-log query-complexity plot: x = 1/eps, y = queries.

    Plots analytic MC (slope ~2), empirical QAE points (where meaningful), and the
    theoretical QAE 1/eps reference line (slope ~1). Empirical log-log slopes are
    fitted via np.polyfit and ANNOTATED in the legend so ~2 vs ~1 is verifiable.
    Saturated empirical points are annotated on the figure itself.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fit_slope(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        m = (xs > 0) & (ys > 0)
        if m.sum() < 2:
            return None
        return float(np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)[0])

    def _series(method, kind):
        pts = sorted((1.0 / r["epsilon"], r["queries"]) for r in rows
                     if r["method"] == method and r["kind"] == kind
                     and r["epsilon"] > 0 and r["queries"] > 0)
        if not pts:
            return None, None
        xs, ys = zip(*pts)
        return list(xs), list(ys)

    P, caption, provenance = _fig_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    mc_x, mc_y = _series("classical_mc", "theoretical")
    if mc_x:
        s = _fit_slope(mc_x, mc_y)
        lab = "classical MC — analytic CLT (THEORETICAL)"
        if s is not None:
            lab += f", slope = {s:.2f}"
        ax.loglog(mc_x, mc_y, "s-", color=P["classical"], lw=1.8, ms=6, label=lab)

    qt_x, qt_y = _series("qae", "theoretical")
    if qt_x:
        s = _fit_slope(qt_x, qt_y)
        lab = r"QAE — theory $\pi/2\varepsilon$ (THEORETICAL)"
        if s is not None:
            lab += f", slope = {s:.2f}"
        ax.loglog(qt_x, qt_y, "--", color=P["quantum"], lw=1.8, label=lab)

    qe_x, qe_y = _series("qae", "empirical")
    if qe_x:
        ax.loglog(qe_x, qe_y, "o", color=P["quantum"], markersize=9,
                  markerfacecolor="white", markeredgewidth=1.8,
                  label="QAE — empirical IAE queries (SIMULATED)")
        # annotate saturation when the points have stopped growing with 1/eps
        if len(qe_y) >= 2 and qe_y[-1] <= qe_y[-2] * 1.05:
            ax.annotate("IAE schedule saturates at small $M$\n"
                        "(honest simulator artifact — reported, not hidden)",
                        xy=(qe_x[-1], qe_y[-1]), xytext=(0.97, 0.42),
                        textcoords="axes fraction", ha="right", fontsize=9,
                        color=P["ink"],
                        arrowprops=dict(arrowstyle="->", color=P["muted"], lw=1.2))

    ax.set_xlabel(r"$1/\varepsilon$  (target accuracy)")
    ax.set_ylabel(r"oracle queries / samples to reach $\varepsilon$")
    ax.set_title(r"Query complexity: classical $O(1/\varepsilon^2)$ vs QAE $O(1/\varepsilon)$")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    caption(fig, "Cost to reach target accuracy ε (log–log). Lines are THEORETICAL "
                 "(analytic CLT for MC; π/2ε for QAE); circles are SIMULATED IAE query counts.")
    provenance(fig, _PROVENANCE)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_speedup_plot(rows, path="quantum_pricer/speedup.png"):
    """Write a log-log error-vs-queries PNG. queries==0 points are dropped (log axis);
    NaN errors (failed routes) are skipped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P, caption, provenance = _fig_style()
    colors = {"classical_mc": P["classical"], "qae": P["quantum"], "qsvt": P["accent"]}
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for method in sorted({r["method"] for r in rows}):
        pts = sorted((r["queries"], r["abs_error"]) for r in rows
                     if r["method"] == method
                     and r["queries"] > 0 and np.isfinite(r["abs_error"])
                     and r["abs_error"] > 0)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, "o-", lw=1.6, ms=6, label=method,
                  color=colors.get(method, P["muted"]))
    ax.set_xlabel("queries / samples")
    ax.set_ylabel("absolute price error (vs exact tree)")
    ax.set_title(r"Error vs queries: classical $1/\sqrt{N}$ vs quantum $1/N$ (single seed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    caption(fig, "Single-seed diagnostic; prefer the seed-averaged RMS figure for slopes.")
    provenance(fig, _PROVENANCE)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_speedup_plot_rms(rows, path="quantum_pricer/speedup.png"):
    """Log-log empirical RMS-error-vs-queries PNG for Figure 2.

    x = queries (QAE) / samples (MC), y = seed-averaged RMS error vs the exact tree
    price. Each series is drawn with markers + line; log-log slopes are fitted via
    np.polyfit and put in the legend (expect MC ~ -0.5, QAE ~ -1.0). Saturated-QAE
    rows (note='qae_saturated_theory') are drawn as a dashed theoretical reference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fit_slope(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if len(xs) < 2:
            return None
        return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])

    def _pts(predicate):
        pts = sorted((r["budget_x"], r["rms_error"]) for r in rows
                     if predicate(r) and r["budget_x"] > 0
                     and np.isfinite(r["rms_error"]) and r["rms_error"] > 0)
        if not pts:
            return [], []
        xs, ys = zip(*pts)
        return list(xs), list(ys)

    P, caption, provenance = _fig_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    def _guide(x0, y0, slope, x1, label):
        """Dotted reference line of given log-log slope anchored at (x0, y0)."""
        xs = np.geomspace(x0, x1, 50)
        ys = y0 * (xs / x0) ** slope
        ax.loglog(xs, ys, ":", color=P["muted"], lw=1.2, zorder=1)
        ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8.5,
                    color=P["muted"], va="center")

    mc_x, mc_y = _pts(lambda r: r["method"] == "classical_mc")
    if mc_x:
        s = _fit_slope(mc_x, mc_y)
        lab = r"classical MC ($1/\sqrt{N}$)"
        if s is not None:
            lab += f", fitted slope = {s:.2f}"
        ax.loglog(mc_x, mc_y, "s-", color=P["classical"], lw=1.8, ms=6, label=lab)
        _guide(mc_x[0], mc_y[0], -0.5, mc_x[-1], r"$-\frac{1}{2}$")

    saturated = any(r["method"] == "qae" and r.get("note") == "qae_saturated_theory"
                    and r["n_seeds"] == 0 for r in rows)
    if saturated:
        # empirical (saturated) QAE points + dashed theoretical pi/(2 eps) line
        qe_x, qe_y = _pts(lambda r: r["method"] == "qae" and r["n_seeds"] > 0)
        if qe_x:
            ax.loglog(qe_x, qe_y, "o", color=P["quantum"], markersize=9,
                      markerfacecolor="white", markeredgewidth=1.8,
                      label="QAE empirical (saturated — see caption)")
        qt_x, qt_y = _pts(lambda r: r["method"] == "qae" and r["n_seeds"] == 0)
        if qt_x:
            s = _fit_slope(qt_x, qt_y)
            lab = r"QAE theory $\pi/2\varepsilon$"
            if s is not None:
                lab += f", slope = {s:.2f}"
            ax.loglog(qt_x, qt_y, "--", color=P["quantum"], lw=1.8, label=lab)
    else:
        qae_x, qae_y = _pts(lambda r: r["method"] == "qae")
        if qae_x:
            s = _fit_slope(qae_x, qae_y)
            lab = "QAE finite shots ($1/N$)"
            if s is not None:
                lab += f", fitted slope = {s:.2f}"
            ax.loglog(qae_x, qae_y, "o-", color=P["quantum"], lw=1.8, ms=7, label=lab)
            _guide(qae_x[0], qae_y[0], -1.0, qae_x[-1], r"$-1$")

    ax.set_xlabel("queries (QAE) / samples (MC)")
    ax.set_ylabel("RMS price error vs exact tree")
    ax.set_title(r"Empirical RMS error vs work: classical $1/\sqrt{N}$ vs QAE $1/N$")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    caption(fig, "SIMULATED, seed-averaged (RMS over 8 seeds per budget); ground truth = "
                 "exact tree. QAE runs with finite shots so its estimation error is genuine. "
                 "Dotted guides show the ideal slopes (MC −½, QAE −1).")
    provenance(fig, _PROVENANCE)
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Hamming-weight depth-vs-M crossover (headline result) ──────────────────────
#
# HONESTY: the naive phase oracle is a Diagonal (Fourier route) or UCRY (QAE route)
# over ALL 2**M paths -- its gate count grows exponentially, so beyond a modest M
# the circuit cannot even be SYNTHESIZED on a laptop (memory/time). We therefore cap
# naive synthesis at `naive_max_M` and record naive_cz=None / note="naive_infeasible"
# beyond it. That cap is not a benchmarking convenience -- it IS the result: the
# Hamming-weight circuit acts on a ceil(log2(M+1))-qubit weight register and stays
# poly(M), so it builds (and prices) at M where the naive route is impossible.


def _cz_count(qc):
    """Transpile to IQM {r,cz} (opt level 1) and return the CZ COUNT (more robust
    than depth for comparing exponential-vs-polynomial two-qubit cost)."""
    return transpile(qc, basis_gates=backends.IQM_BASIS,
                     optimization_level=1).count_ops().get("cz", 0)


def depth_vs_M(M_list, naive_max_M=12, route="fourier",
               S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0):
    """Measured CZ-count crossover: naive 2**M oracle vs Hamming-weight poly(M).

    For each M build the HAMMING circuit (always) and the NAIVE circuit (only when
    M <= naive_max_M -- beyond that the 2**M Diagonal/UCRY is infeasible to even
    synthesize, recorded as naive_cz=None, note="naive_infeasible"). Both are
    transpiled to IQM {r,cz} at optimization_level=1 and the CZ COUNT recorded.

    route="fourier" -> oracles.fourier_circuit at lam=1.0 (phase oracle);
    route="qae"     -> oracles.payoff_amplitude_circuit (amplitude oracle).

    Returns rows: dict(M, naive_cz, hamming_cz, note). naive_cz is None when
    M > naive_max_M.
    """
    rows = []
    for M in M_list:
        angles = tree.loading_angles(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
        ST_by_w = hamming.terminal_values_by_weight(S0=S0, K=K, r=r, sigma=sigma,
                                                     T=T, M=M)
        if route == "fourier":
            ham_qc = hamming.fourier_circuit_hamming(M, angles, ST_by_w, K,
                                                     lam=1.0, basis="X")
        elif route == "qae":
            payoff_w = np.maximum(ST_by_w - K, 0.0)
            Cmax = float(payoff_w.max()) * 1.0001 if payoff_w.max() > 0 else 1.0
            ham_qc, _ = hamming.payoff_amplitude_circuit_hamming(M, angles,
                                                                 payoff_w, Cmax)
        else:
            raise ValueError(f"unknown route {route!r}")
        hamming_cz = _cz_count(ham_qc)

        if M <= naive_max_M:
            values = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma,
                                                 T=T, M=M, option="european")
            if route == "fourier":
                naive_qc = oracles.fourier_circuit(M, angles, values, K,
                                                   lam=1.0, basis="X")
            else:
                payoff = np.maximum(values - K, 0.0)
                Cmax = float(payoff.max()) * 1.0001 if payoff.max() > 0 else 1.0
                naive_qc, _ = oracles.payoff_amplitude_circuit(angles, payoff, Cmax)
            naive_cz = _cz_count(naive_qc)
            note = ""
        else:
            naive_cz = None
            note = "naive_infeasible"  # 2**M oracle cannot be synthesized -- the point
        rows.append(dict(M=M, naive_cz=naive_cz, hamming_cz=hamming_cz, note=note))
    return rows


def save_depth_crossover_plot(rows, path="quantum_pricer/depth_crossover.png"):
    """Semilog-y plot of naive (exponential, truncated) vs Hamming (polynomial) CZ
    count vs M. Marks the crossover M where Hamming first beats naive. Returns path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["M"])
    Ms = [r["M"] for r in rows]
    ham = [r["hamming_cz"] for r in rows]
    naive_pts = [(r["M"], r["naive_cz"]) for r in rows if r["naive_cz"] is not None]
    infeasible_Ms = [r["M"] for r in rows if r["naive_cz"] is None]

    P, caption, provenance = _fig_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    if naive_pts:
        nx, ny = zip(*naive_pts)
        ax.semilogy(nx, ny, "s-", color=P["classical"], lw=1.8, ms=6,
                    label="naive $2^M$ oracle (MEASURED transpile)")
    ax.semilogy(Ms, [max(h, 1) for h in ham], "o-", color=P["quantum"], lw=1.8,
                ms=6, label="Hamming-weight poly($M$) (MEASURED transpile)")

    # shade the region where the naive oracle cannot even be synthesized
    if infeasible_Ms and naive_pts:
        lo = (naive_pts[-1][0] + infeasible_Ms[0]) / 2.0
        ax.axvspan(lo, max(Ms) + 0.5, color=P["classical"], alpha=0.08, zorder=0)
        ax.text((lo + max(Ms)) / 2.0, ax.get_ylim()[1] * 0.5,
                "naive $2^M$ oracle\nunsynthesizable\n(this IS the result)",
                ha="center", va="top", fontsize=9, color=P["classical"])

    # crossover: smallest M where both are present and Hamming < naive
    crossover = next((r["M"] for r in rows if r["naive_cz"] is not None
                      and r["hamming_cz"] < r["naive_cz"]), None)
    if crossover is not None:
        ax.axvline(crossover, color=P["muted"], ls=":", lw=1.2)
        ax.annotate(f"crossover $M={crossover}$", xy=(crossover, max(ham)),
                    xytext=(6, 6), textcoords="offset points", color=P["ink"],
                    fontsize=9)

    ax.set_xlabel("number of time steps $M$")
    ax.set_ylabel(r"CZ count after transpile to IQM $\{r, cz\}$ (log scale)")
    ax.set_title(r"Phase-oracle CZ count vs $M$: naive $2^M$ vs Hamming-weight poly($M$)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    caption(fig, "Two-qubit gate counts measured after transpiling to the IQM native basis "
                 "(optimization level 1). Naive synthesis is capped at M=12 — beyond that the "
                 "2^M-entry oracle cannot be built at all; the Hamming-weight route stays poly(M).")
    provenance(fig, _PROVENANCE)
    fig.savefig(path)
    plt.close(fig)
    return path


def large_m_price(M=18, S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
                  n_lambda=None):
    """Price a European call at large M via the Hamming-weight statevector route --
    a circuit the NAIVE route cannot even build (its oracle is a 2**M-entry diagonal;
    e.g. M=18 -> 262144 entries). Validates against the fast O(M) recombining price.

    Memory guard: keep M<=18 for the statevector simulation (M=18 -> 24 qubits ~268MB;
    M=20 -> 26 qubits ~1GB may OOM). For M>=20 use depth_vs_M (transpile only).

    Returns dict(M, n_qubits, hamming_price, exact_price, abs_error, naive_feasible).
    """
    if n_lambda is None:
        n_lambda = 2 * M + 4
    n_qubits = M + hamming.n_weight_qubits(M) + 1
    hamming_price = fourier.price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                  option="european", kind="call",
                                  n_lambda=n_lambda, use_hamming=True)
    exact_price = tree.exact_tree_price_recombining(S0=S0, K=K, r=r, sigma=sigma,
                                                    T=T, M=M, kind="call")
    return dict(M=M, n_qubits=n_qubits, hamming_price=hamming_price,
                exact_price=exact_price,
                abs_error=abs(hamming_price - exact_price),
                naive_feasible=False)
