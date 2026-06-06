"""Quantum-finance triage lab — end-to-end multi-tool demo.

Three tabs:
  1. Option pricing (QAE) — the triage winner
  2. Portfolio optimisation (QAOA) — honest tie with classical at small scale
  3. Fraud triage — quantum kernel, natural demo value, honest about parity

Run:  uv run streamlit run demo/app.py
"""
import numpy as np
import streamlit as st

from demo.inference import (
    price_option,
    optimize_portfolio,
    train_fraud_model,
    score_transaction,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Quantum-Finance Triage Lab", layout="wide")
st.title("Quantum-Finance Triage Lab — Live Demo")
st.caption(
    "Triage verdict: **QAE shows a measured quantum advantage**; "
    "QAOA & fraud-kernel tie classical at simulable scale (honest)."
)

# ---------------------------------------------------------------------------
# Shared sidebar
# ---------------------------------------------------------------------------
backend = st.sidebar.selectbox(
    "Quantum backend",
    ["local_aer", "lumi_aer", "q50_fake", "q50_hw"],
    help=(
        "local_aer = Qiskit Aer statevector/shot simulator (default). "
        "lumi_aer = LUMI supercomputer Aer instance. "
        "q50_fake = VTT Q50 native gates + noise model (offline). "
        "q50_hw = real VTT Q50 (on-site only)."
    ),
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_qae, tab_qaoa, tab_fraud = st.tabs([
    "Option Pricing (QAE)",
    "Portfolio (QAOA)",
    "Fraud Triage",
])

# ===========================================================================
# Tab 1 — Option pricing via QAE (the triage winner)
# ===========================================================================
with tab_qae:
    st.header("European Call Option Pricing via Quantum Amplitude Estimation")
    st.markdown(
        "QAE encodes the payoff distribution into quantum amplitudes and "
        "uses iterative amplitude estimation to price the option. Its oracle "
        "complexity scales as **O(1/eps)** — quadratically fewer evaluations "
        "than classical Monte Carlo's **O(1/eps²)**. This is the lab's "
        "**measured quantum advantage**."
    )

    col1, col2 = st.columns(2)
    with col1:
        strike = st.number_input(
            "Strike price K", min_value=0.1, max_value=10.0, value=1.9, step=0.1,
            help="Option strike price (underlying S0 is fixed at 2.0).",
        )
        vol = st.number_input(
            "Volatility (sigma)", min_value=0.01, max_value=2.0, value=0.4, step=0.05,
            help="Annualised log-normal volatility.",
        )
        maturity = st.number_input(
            "Maturity T (years)", min_value=0.01, max_value=5.0, value=0.1, step=0.05,
            help="Time to expiry in years.",
        )
    with col2:
        eps = st.number_input(
            "Target accuracy eps", min_value=0.001, max_value=0.2, value=0.01,
            step=0.005, format="%.3f",
            help="Desired +/- pricing accuracy (drives both QAE depth and MC baseline).",
        )
        n_qubits_qae = st.selectbox(
            "Uncertainty qubits", options=[2, 3], index=1,
            help=(
                "Qubits used to discretise the log-normal distribution. "
                "3 qubits = 8 grid points; circuit transpiles to VTT Q50."
            ),
        )

    if st.button("Price it", key="btn_qae"):
        with st.spinner("Running IQAE on the option pricing circuit…"):
            try:
                r = price_option(
                    strike=strike, vol=vol, maturity=maturity,
                    eps=eps, n_qubits=n_qubits_qae,
                    backend=backend, seed=7,
                )
            except Exception as exc:
                st.error(f"Pricing failed: {exc}")
                st.stop()

        # --- Results ---
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("QAE option price", f"{r['price']:.4f}")
        mcol2.metric("Oracle queries (measured)", f"{r['oracle_queries']:,}")
        mcol3.metric("MC samples (same eps)", f"{r['mc_samples']:,}")

        q = r["oracle_queries"]
        mc = r["mc_samples"]
        ratio = mc / max(q, 1)
        st.success(
            f"QAE: ~{q:,} oracle queries vs ~{mc:,} MC samples for eps={eps:.3f} "
            f"— **{ratio:.1f}x fewer** at tight accuracy."
        )

        with st.expander("Circuit and pricing details"):
            st.write(f"**Total circuit qubits:** {r['n_qubits']} "
                     f"(uncertainty + ancilla; transpiles to VTT Q50)")
            st.write(f"**Exact discretised payoff** (ground truth from grid): "
                     f"{r['exact_payoff']:.4f}")
            st.write(f"**Speedup ratio** mc_samples / oracle_queries: {r['speedup']:.2f}")
            st.write(
                "The circuit loads a log-normal S_T on "
                f"{n_qubits_qae} uncertainty qubits via qiskit-finance "
                "LogNormalDistribution and EuropeanCallPricing. "
                "IQAE oracle queries are MEASURED from runtime Grover powers "
                "(sum (2k+1) × shots per round), not the analytic 1/eps formula."
            )

# ===========================================================================
# Tab 2 — Portfolio optimisation via QAOA
# ===========================================================================
with tab_qaoa:
    st.header("Cardinality-Constrained Portfolio Optimisation via QAOA")
    st.markdown(
        "QAOA encodes the portfolio selection problem as a QUBO, maps it to "
        "an Ising Hamiltonian, and optimises a variational ansatz to find the "
        "best k-asset portfolio. At small scale classical enumeration also "
        "finds the optimum — **QAOA's edge is asymptotic** when brute force "
        "becomes infeasible."
    )

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        n_assets = st.slider(
            "Number of assets (n)", min_value=4, max_value=10, value=4, step=1,
            help="Total assets to choose from. Random mu/cov generated from seed.",
        )
        k_assets = st.slider(
            "Assets to pick (k)", min_value=1, max_value=5, value=2, step=1,
            help="Cardinality constraint: exactly k assets must be selected.",
        )
    with pcol2:
        risk_lambda = st.number_input(
            "Risk aversion (lambda)", min_value=0.0, max_value=10.0,
            value=1.0, step=0.25,
            help="Weight on the variance penalty in mu.x - lambda * x^T cov x.",
        )
        qaoa_reps = st.slider(
            "QAOA repetitions (p)", min_value=1, max_value=4, value=2, step=1,
            help="Number of QAOA layers (depth). More = better approximation ratio, slower.",
        )
        portfolio_seed = st.number_input(
            "Random seed", min_value=0, max_value=999, value=42, step=1,
            help="Seed for the random mu/cov instance. Change to explore different problems.",
        )

    # Clamp k to be <= n_assets
    k_assets = min(k_assets, n_assets)

    if st.button("Optimize", key="btn_qaoa"):
        with st.spinner("Running QAOA + classical comparison…"):
            try:
                pr = optimize_portfolio(
                    n_assets=n_assets, k=k_assets,
                    risk=risk_lambda, reps=qaoa_reps,
                    seed=int(portfolio_seed), backend=backend,
                )
            except Exception as exc:
                st.error(f"Portfolio optimisation failed: {exc}")
                st.stop()

        chosen_str = ", ".join(str(i) for i in pr["chosen"])
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("QAOA selected assets", f"[{chosen_str}]")
        rcol2.metric("QAOA objective", f"{pr['objective']:.4f}")
        rcol3.metric("Approx. ratio vs exact", f"{pr['approx_ratio']:.1%}")

        st.info(
            f"QAOA selected assets [{chosen_str}]; objective {pr['objective']:.4f} "
            f"= **{pr['approx_ratio']:.0%}** of the exact optimum "
            f"({pr['optimum']:.4f}). "
            "(At this size classical also solves it; QAOA's edge is asymptotic.)"
        )

        with st.expander("Problem and result details"):
            st.write(
                f"Objective: maximise  mu·x − {risk_lambda}·x^T cov x  "
                f"subject to sum(x) = {k_assets}, x ∈ {{0,1}}^{n_assets}"
            )
            st.write(f"Exact optimum: {pr['optimum']:.6f}")
            st.write(f"QAOA objective: {pr['objective']:.6f}")
            st.write(f"Approximation ratio: {pr['approx_ratio']:.4f}")
            st.write(
                "mu and cov are randomly generated from the seed above "
                "(uniform mu ∈ [0, 0.2]; cov = A·Aᵀ, A ~ Normal(0, 0.02)). "
                "QAOA uses COBYLA optimiser with 250 iterations."
            )

# ===========================================================================
# Tab 3 — Fraud triage (existing, re-housed)
# ===========================================================================
with tab_fraud:
    st.header("Quantum-Kernel Fraud Triage")
    st.markdown(
        "A quantum-kernel SVM scores incoming transactions using a "
        "parameterised feature map whose inner products are evaluated on the "
        "quantum device. At tabular scale (4–8 features, 200 training points) "
        "this matches a classical RBF-SVM."
    )
    st.info(
        "Honest note: the quantum kernel **ties classical RBF** on tabular "
        "fraud data at this scale — included for its natural demo value, "
        "not as a measured quantum win."
    )

    n_features = st.sidebar.slider("Feature qubits", 2, 8, 4)
    threshold = st.sidebar.slider("Flag threshold", 0.0, 1.0, 0.5)

    @st.cache_resource
    def _model(backend, n_features):
        return train_fraud_model(backend=backend, n=200, n_features=n_features, seed=0)

    with st.spinner("Training quantum-kernel model…"):
        model = _model(backend, n_features)

    st.subheader("Incoming transactions")
    if st.button("Generate a batch", key="btn_fraud"):
        rng = np.random.default_rng()
        rows = []
        for i in range(8):
            x = rng.normal(0, 1, n_features) + (1.8 if rng.random() < 0.3 else 0.0)
            p = score_transaction(model, x)
            rows.append({
                "txn": f"T{i:03d}",
                "fraud_prob": round(p, 3),
                "flag": "FRAUD" if p >= threshold else "ok",
            })
        st.dataframe(rows, use_container_width=True)
        st.caption(
            f"Scored on backend: {backend} — quantum-kernel SVM "
            f"({n_features}-qubit feature map)"
        )
