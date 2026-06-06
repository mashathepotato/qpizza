"""End-to-end demo: 3-way quantum option-pricing benchmark on Nokia (NOKIA.HE).

Run as:  python -m quantum_pricer.demo

Ground truth:  exact CRR binomial-tree price  (what the quantum routes estimate).
Continuum check: Black-Scholes (agrees with tree as M -> infinity).
Three quantum/classical routes priced:
  1. Classical MC     O(1/eps^2) samples
  2. QNDM Fourier     shallow-circuit characteristic-function route (Q50 anchor)
  3. QNDM QAE         amplitude-estimation O(1/eps) oracle queries
  4. Novel QSVT       straddle + put-call parity (honest ~1.4% approx residual)
"""
import os
import sys

# ── params ────────────────────────────────────────────────────────────────────
# Use M=3 for pricing runs (fast, < 10 s on a laptop).
# Use M=4 for the benchmark (more distinct QAE points, still finishes quickly).
M_PRICE = 3   # for individual route prices
M_BENCH = 4   # for error_vs_queries / resource table / complexity plot
T = 1.0
N_MC_PATHS = 100_000

# ── paths for saved figures ───────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEEDUP_PATH = os.path.join(_HERE, "speedup.png")
_COMPLEXITY_PATH = os.path.join(_HERE, "complexity.png")


def _banner(text):
    width = 72
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    # ── 1. Load Nokia params ──────────────────────────────────────────────────
    _banner("Step 1 — Market data (NOKIA.HE)")
    from quantum_pricer.data import nokia_params
    params, meta = nokia_params(allow_network=True)
    S0    = params["S0"]
    sigma = params["sigma"]
    r     = params["r"]
    K     = round(S0, 2)   # ATM strike

    print(f"  Source   : {meta['source']}")
    if "ticker" in meta:
        print(f"  Ticker   : {meta['ticker']}  ({meta.get('start','')} – {meta.get('end','')},"
              f"  n={meta.get('n_obs','')} obs)")
    if meta.get("source") == "synthetic":
        print(f"  Reason   : {meta.get('reason', 'network disabled')}  [SYNTHETIC FALLBACK]")
    print(f"  S0 = {S0:.4f}   sigma = {sigma:.4f}   r = {r:.4f}")
    print(f"  K (ATM)  : {K:.2f}   T = {T}   M_price = {M_PRICE}   M_bench = {M_BENCH}")

    # ── 2. Ground truth + continuum check ────────────────────────────────────
    _banner("Step 2 — Ground truth & continuum check")
    from quantum_pricer.tree import exact_tree_price
    from quantum_pricer.classical import black_scholes_call

    exact_price = exact_tree_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_PRICE)
    bs_price    = black_scholes_call(S0=S0, K=K, r=r, sigma=sigma, T=T)

    print(f"  Exact tree price  (M={M_PRICE}) [GROUND TRUTH]  : {exact_price:.6f}")
    print(f"  Black-Scholes                  [CONTINUUM CHECK]: {bs_price:.6f}")
    print(f"  BS – tree gap : {bs_price - exact_price:+.6f}  "
          f"(small at M={M_PRICE}; shrinks as M -> inf)")

    # ── 3. Four routes ───────────────────────────────────────────────────────
    _banner("Step 3 — Prices from all four routes  (M={})".format(M_PRICE))
    print(f"  {'Method':<22}  {'Price':>10}  {'Error vs tree':>14}  Note")
    print(f"  {'-'*22}  {'-'*10}  {'-'*14}  {'-'*30}")

    def _row(name, price_val, note=""):
        err = price_val - exact_price
        print(f"  {name:<22}  {price_val:>10.6f}  {err:>+14.6f}  {note}")

    # 3a. Classical MC
    from quantum_pricer.classical import monte_carlo_price
    mc_price, mc_stderr = monte_carlo_price(
        S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_PRICE,
        n_paths=N_MC_PATHS, option="european", kind="call", seed=42)
    _row("classical MC",   mc_price,
         f"stderr={mc_stderr:.4f}, n={N_MC_PATHS:,}")

    # 3b. QNDM Fourier
    from quantum_pricer.fourier import price as fourier_price
    fq_price = fourier_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_PRICE,
                             option="european", kind="call")
    _row("QNDM Fourier",   fq_price,
         "shots=None (statevector, O(1/eps^2) in shots)")

    # 3c. QNDM QAE
    from quantum_pricer.qae import price as qae_price
    qae_result = qae_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_PRICE,
                           option="european", kind="call", epsilon_target=0.01)
    _row("QNDM QAE",       qae_result["price"],
         f"eps=0.01, queries={qae_result['num_oracle_queries']}")

    # 3d. Novel QSVT (wrap in try/except so a failure prints rather than crashes)
    try:
        from quantum_pricer.qsvt import price as qsvt_price
        qsvt_val = qsvt_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_PRICE,
                              option="european", kind="call",
                              degree=60, use_qae=False)
        _row("novel QSVT",     qsvt_val,
             "straddle+parity, ~1.4% approx floor (degree 60)")
    except Exception as exc:
        print(f"  {'novel QSVT':<22}  {'(skipped)'!r:>10}  "
              f"{'':>14}  reason: {exc}")

    # ── 4. Resource table ─────────────────────────────────────────────────────
    _banner("Step 4 — Resource table (qubits + IQM CZ depth, M={})".format(M_BENCH))
    from quantum_pricer.benchmark import resource_table
    table = resource_table(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_BENCH,
                           option="european", kind="call", qsvt_degree=20)
    print(f"  {'Method':<18}  {'Qubits':>6}  {'CZ depth':>8}")
    print(f"  {'-'*18}  {'-'*6}  {'-'*8}")
    for row in table:
        print(f"  {row['method']:<18}  {row['qubits']:>6}  {row['cz_depth']:>8}")

    # ── 5. Complexity plot (money slide) + speedup plot ───────────────────────
    _banner("Step 5 — Query-complexity plot (money slide)")
    from quantum_pricer.benchmark import (
        queries_to_accuracy, save_complexity_plot,
        error_vs_queries, save_speedup_plot,
    )
    import numpy as np

    print(f"  Running queries_to_accuracy at M={M_BENCH} (analytic MC + empirical QAE)…")
    crows = queries_to_accuracy(
        S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_BENCH,
        option="european", kind="call",
        epsilons=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005),
    )

    # Fit slopes for MC and QAE-theory to print the numbers
    def _slope(method, kind_key):
        pts = [(1.0 / row["epsilon"], row["queries"]) for row in crows
               if row["method"] == method and row["kind"] == kind_key
               and row["epsilon"] > 0 and row["queries"] > 0]
        if len(pts) < 2:
            return None
        xs, ys = zip(*pts)
        return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])

    mc_slope  = _slope("classical_mc", "theoretical")
    qae_slope = _slope("qae",          "theoretical")

    save_complexity_plot(crows, path=_COMPLEXITY_PATH)
    print(f"  Saved complexity plot : {_COMPLEXITY_PATH}")
    if mc_slope is not None:
        print(f"  Fitted slope  classical MC  : {mc_slope:.3f}  (theory = 2.0)")
    if qae_slope is not None:
        print(f"  Fitted slope  QAE theory    : {qae_slope:.3f}  (theory = 1.0)")

    print(f"\n  Running error_vs_queries at M={M_BENCH} (MC + QAE; QSVT skipped for speed)…")
    srows = error_vs_queries(
        S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_BENCH,
        option="european", kind="call",
        mc_budgets=(1_000, 10_000, 100_000),
        qae_eps=(0.2, 0.1, 0.05, 0.02, 0.01),
        include_qsvt=False,          # QSVT is slow at M=4; excluded for wall-clock
    )
    save_speedup_plot(srows, path=_SPEEDUP_PATH)
    print(f"  Saved speedup plot    : {_SPEEDUP_PATH}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    _banner("Summary")
    src_label = "LIVE yfinance" if meta["source"] == "yfinance" else "SYNTHETIC (offline fallback)"
    print(f"  Data source     : {src_label}")
    print(f"  S0={S0:.4f}  K={K:.2f}  sigma={sigma:.4f}  r={r:.4f}  T={T}  M={M_PRICE}")
    print(f"  Ground truth (tree, M={M_PRICE}) : {exact_price:.6f}")
    print(f"  Black-Scholes (continuum)       : {bs_price:.6f}")
    print()
    print("  Quadratic advantage (query-complexity plot):")
    if mc_slope is not None:
        print(f"    MC analytic slope  ~{mc_slope:.2f}  (expect 2)")
    if qae_slope is not None:
        print(f"    QAE theory slope   ~{qae_slope:.2f}  (expect 1)")
    print()
    print("  Figures saved:")
    print(f"    {_COMPLEXITY_PATH}")
    print(f"    {_SPEEDUP_PATH}")
    print()
    print("  HONESTY NOTES:")
    print("    * QSVT measures the straddle E[|f-K|] and recovers the call via")
    print("      put-call parity; it carries a ~1.4% polynomial-approx residual.")
    print("    * QNDM Fourier (shots=None) is O(1/eps^2) in shots — its win is")
    print("      shallow depth / exact loading / Q50 feasibility, not eps-scaling.")
    print("    * QAE's query schedule saturates at small M (annotated in complexity rows).")
    print()


if __name__ == "__main__":
    main()
