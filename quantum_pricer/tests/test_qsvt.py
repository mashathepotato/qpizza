import numpy as np
from quantum_pricer import qsvt, tree


def test_relu_poly_approximates_target():
    # The even target polynomial P(x) ~ sqrt(A + B*|theta(x)|), |theta|=2*arccos(|x|),
    # should track its target on the ACTIVE QSP band x in [cos(0.5), 1] (where the
    # eigenphase theta lives), away from the cusp at x=1.  (pyqsp-free.)
    A, B, deg = 0.10, 0.35, 60
    coef = qsvt.straddle_poly(degree=deg, A=A, B=B)
    # the construction must produce an exactly even polynomial
    assert np.max(np.abs(coef[1::2])) < 1e-9
    xs = np.linspace(np.cos(0.5), 1.0, 400)
    abs_theta = np.clip(2.0 * np.arccos(np.clip(xs, -1.0, 1.0)), 0.0, 1.0)
    target = np.sqrt(A + B * abs_theta)
    approx = qsvt.eval_cheb(coef, xs)
    # allow error near the cusp (small |theta| <-> x near 1); check away from it
    mask = abs_theta > 0.15
    assert np.max(np.abs(approx[mask] - target[mask])) < 0.05


def test_transfer_function_is_linear_in_abs_theta():
    # Probe W(theta)=Pr[s=0] of the QET-U sequence on KNOWN swept phases and assert
    # W(theta) ~ w0 + kappa*|theta|.  This proves the circuit measures |f-K| (the
    # straddle), and it is a calibration with known inputs -- not the option answer.
    coef = qsvt.straddle_poly(degree=60, A=0.10, B=0.35)
    phis, used_pyqsp = qsvt.qsp_phases(coef)
    assert used_pyqsp  # pyqsp must converge for the documented chain
    w0, kappa, resid = qsvt._probe_transfer_function(phis, n=41)
    # documented residual at deg=60: ~0.006 absolute (see report).
    assert resid < 0.02
    assert kappa > 0.1            # genuine, non-degenerate |theta| slope
    # slope tracks the design constant B (~0.35) up to the cusp distortion
    assert abs(kappa - 0.35) < 0.1


def test_qsvt_call_price_matches_exact_tree_statevector(base_params):
    # Statevector path, M=2 ATM call, honest straddle + put-call parity.
    # Limiting error is the finite-degree sqrt-cusp approximation of |theta|.
    # Documented residual at degree=60: call ~= 10.04 vs exact ~= 10.19,
    # absolute residual ~= -0.15 (~ -1.5% relative).  Higher degree tightens it.
    M = 2
    exact = tree.exact_tree_price(M=M, option="european", kind="call", **base_params)
    res = qsvt.price(M=M, option="european", kind="call",
                     degree=60, use_qae=False, return_meta=True, **base_params)
    # sanity: forward equals the distribution expected underlying (NOT the option price)
    values = tree.payoff_variable_values(M=M, option="european", **base_params)
    p = tree.path_probabilities(M=M, **base_params)
    assert abs(res["forward"] - float(np.sum(p * values))) < 1e-6
    assert abs(res["price"] - exact) < 0.3     # honest polynomial-approximation tol


def test_qsvt_single_run_qae(base_params):
    # use_qae=True, finite-shot Sampler -> IAE actually iterates (queries > 0).
    M = 2
    exact = tree.exact_tree_price(M=M, option="european", kind="call", **base_params)
    res = qsvt.price(M=M, option="european", kind="call", degree=60, use_qae=True,
                     epsilon_target=0.02, shots=4096, seed=7,
                     return_meta=True, **base_params)
    assert res["num_oracle_queries"] > 0
    assert abs(res["price"] - exact) < 0.6     # looser: finite-shot QAE noise + poly
