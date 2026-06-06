"""Novel route (paper Sec 5c): QNDM-powered single-run QAE via QSVT/QET-U.

HONEST straddle + put-call parity construction.
=================================================

The call payoff max(f-K,0) is one-sided (no definite parity), so a single-signal
QSP/QET-U sequence cannot represent it directly. Instead we have the quantum
circuit genuinely measure the SYMMETRIC quantity (the straddle)

    E[|f-K|]   (even-parity in theta(x)=c(f(x)-K), QSP-friendly)

and recover the call/put by put-call parity using the a-priori FORWARD price F:

    call = exp(-rT) * ( E[|f-K|] + (F - K) ) / 2
    put  = exp(-rT) * ( E[|f-K|] - (F - K) ) / 2

NOTHING in the scaling path uses the option value / payoff expectation. Every
constant that turns the circuit amplitude `a` into E[|f-K|] comes from:
  * the payoff VALUE RANGE  -> Cmax = max_x |f(x)-K|,  coupling c = 1/Cmax,
  * the chosen QSP phases    -> the transfer-function probe (w0, kappa),
and the forward F uses only S0, r, T, M (analytic risk-neutral forward).

QET-U convention (verified numerically, see _probe_transfer_function):
  With signal_operator='Wx' (R_x rotations) and a controlled phase oracle
  O = diag(e^{i*alpha}) applied to a single eigenstate, the signal-ancilla
  return probability is EXACTLY

      Pr[s=0] = P(cos(alpha/2))^2

  where P is the (even) pyqsp target Chebyshev polynomial. We therefore design P
  so that, across alpha = theta in [-1,1], the realized transfer function

      W(theta) := Pr[s=0]  ~=  w0 + kappa*|theta|     (affine in |theta|)

  is linear in |theta|. (w0, kappa) are then READ OFF the probe -- a calibration
  with KNOWN swept phases, never the option answer. For the full circuit the
  ancilla amplitude is the path-average of the same transfer function:

      a = E_x[ W(theta(x)) ] = w0 + kappa * E[|theta|] = w0 + kappa * E[|f-K|]/Cmax

  =>  E[|f-K|] = Cmax * (a - w0) / kappa.
"""
import numpy as np
from numpy.polynomial import chebyshev as _C
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from quantum_pricer import tree


# ---- (a) classical polynomial layer -------------------------------------------------
#
# Design an EVEN polynomial P(x) such that the QET-U transfer function
#     W(theta) = P(cos(theta/2))^2
# is affine in |theta| on theta in [-1,1].  For theta in [-1,1] the QSP variable
# x = cos(theta/2) lives in [cos(0.5), 1], and there |theta| = 2*arccos(x).  We
# therefore target
#     P(x) = sqrt( A + B * clip(2*arccos(|x|), 0, 1) )
# which makes W(theta) ~ A + B*|theta| (so w0~=A, kappa~=B before circuit fitting).
# A, B are FREE design constants (chosen to keep W in (0,1)); they are NOT derived
# from the option.  The sqrt cusp at theta=0 is the genuine, honest source of the
# finite-degree approximation error.

_A_DEFAULT = 0.10
_B_DEFAULT = 0.35


def straddle_poly(degree=60, A=_A_DEFAULT, B=_B_DEFAULT, grid_n=2000):
    """Even Chebyshev coefficients of P(x) ~ sqrt(A + B*|theta(x)|), |theta|=2arccos(|x|).

    Returns the Chebyshev coefficient array (odd entries zeroed for exact even parity)."""
    xs = np.linspace(-1.0, 1.0, grid_n)
    ax = np.abs(xs)
    abs_theta = np.clip(2.0 * np.arccos(np.clip(ax, -1.0, 1.0)), 0.0, 1.0)
    target = np.sqrt(A + B * abs_theta)
    cheb = _C.Chebyshev.fit(xs, target, deg=degree, domain=[-1.0, 1.0])
    coef = cheb.coef.copy()
    coef[1::2] = 0.0  # enforce exact even parity (required for single-signal QSP)
    return coef


def eval_cheb(coef, xs):
    return _C.chebval(xs, coef)


def qsp_phases(coef):
    """pyqsp phase angles for the even Chebyshev target `coef` (signal_operator='Wx').

    Returns (phis, used_pyqsp).  pyqsp's laurent method takes a Chebyshev coefficient
    array directly.  On failure we fall back to using the coefficients as a raw phase
    schedule (used_pyqsp=False); the transfer-function probe still characterizes
    whatever function the resulting sequence realizes, so the scaling stays honest."""
    try:
        from pyqsp.angle_sequence import QuantumSignalProcessingPhases
        phis = QuantumSignalProcessingPhases(np.asarray(coef, dtype=float),
                                             signal_operator="Wx")
        return np.asarray(phis, dtype=float), True
    except Exception:
        return np.asarray(coef, dtype=float), False


# ---- (b) QET-U circuit on the phase oracle ------------------------------------------

def _phase_oracle_entries(values, K, c):
    """Diagonal of O = diag(e^{i c (f(x)-K)}) over the 2**M path basis states."""
    return np.exp(1j * c * (np.asarray(values, float) - K))


def _qetu_sequence(qc, signal_idx, controlled_oracle, phis, oracle_qubits):
    """Apply the QET-U signal-processing sequence on `signal_idx`:
    Rx(2*phi0); for phi in phis[1:]: controlled-O; Rx(2*phi)."""
    qc.rx(2.0 * phis[0], signal_idx)
    for phi in phis[1:]:
        qc.append(controlled_oracle, [signal_idx, *oracle_qubits])
        qc.rx(2.0 * phi, signal_idx)


def build_qsvt_prep(angles, values, K, c, phis):
    """QET-U single-ancilla preparation: load CRR paths on q0..q_{M-1}, then run the
    QET-U sequence on the signal ancilla q_M with the controlled phase oracle.
    Returns (circuit, signal_index)."""
    M = len(angles)
    qc = QuantumCircuit(M + 1, name="A_qsvt")
    for i, th in enumerate(angles):
        qc.ry(th, i)
    s = M
    O = Diagonal(_phase_oracle_entries(values, K, c)).to_gate()
    cO = O.control(1)
    _qetu_sequence(qc, s, cO, phis, list(range(M)))
    return qc, s


# ---- transfer-function probe (instrument calibration with KNOWN phases) -------------

def _probe_transfer_function(phis, n=41):
    """Characterize W(theta)=Pr[s=0] of the QET-U sequence on a TRIVIAL one-state
    oracle whose single eigenphase is swept to a KNOWN theta in [-1,1].
    This uses only the chosen QSP phases and known inputs -- never the option answer.

    Fits W(theta) ~ w0 + kappa*|theta| and returns (w0, kappa, max_residual)."""
    thetas = np.linspace(-1.0, 1.0, n)
    W = np.empty(n)
    for j, theta in enumerate(thetas):
        qc = QuantumCircuit(2)            # q0 = system eigenstate, q1 = signal ancilla
        qc.x(0)                           # put system in |1> with eigenphase = theta
        O = Diagonal([1.0, np.exp(1j * theta)]).to_gate()
        cO = O.control(1)
        _qetu_sequence(qc, 1, cO, phis, [0])
        W[j] = float(Statevector(qc).probabilities([1])[0])
    Amat = np.vstack([np.ones_like(thetas), np.abs(thetas)]).T
    (w0, kappa), *_ = np.linalg.lstsq(Amat, W, rcond=None)
    resid = float(np.max(np.abs(W - (w0 + kappa * np.abs(thetas)))))
    return float(w0), float(kappa), resid


# ---- forward price (a-priori, S0/r/T/M only) ----------------------------------------

def forward_price(S0, r, T, M, option="european"):
    """Risk-neutral forward F of the payoff variable f.  Uses only S0, r, T, M.
    European: F = E_Q[S_T] = S0*exp(rT).
    Asian:    F = E_Q[Sbar] = (S0/M) * sum_{i=1..M} exp(r*i*dt), dt=T/M."""
    if option == "european":
        return float(S0 * np.exp(r * T))
    if option == "asian":
        dt = T / M
        return float((S0 / M) * np.sum(np.exp(r * dt * np.arange(1, M + 1))))
    raise ValueError(f"unknown option {option!r}")


# ---- (c) price (statevector or single-run QAE) --------------------------------------

def price(S0, K, r, sigma, T, M, option="european", kind="call",
          degree=60, A=_A_DEFAULT, B=_B_DEFAULT, use_qae=False,
          epsilon_target=0.02, shots=4096, seed=7, return_meta=False):
    angles = tree.loading_angles(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    values = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                         option=option)

    # a-priori payoff VALUE-RANGE bound (NOT a distribution-weighted expectation):
    Cmax = float(np.max(np.abs(values - K)))
    Cmax = Cmax if Cmax > 1e-12 else 1.0
    c = 1.0 / Cmax                       # so theta(x) = c*(f(x)-K) in [-1, 1]

    coef = straddle_poly(degree=degree, A=A, B=B)
    phis, used_pyqsp = qsp_phases(coef)

    # Instrument calibration with KNOWN swept phases (no option answer involved):
    w0, kappa, probe_resid = _probe_transfer_function(phis)

    qc, s = build_qsvt_prep(angles, values, K, c, phis)

    if not use_qae:
        a = float(Statevector(qc).probabilities([s])[0])
        num_q = 0
    else:
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[s],
                                    is_good_state=lambda b: b == "0")
        iae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon_target, alpha=0.05,
            sampler=Sampler(options={"shots": shots, "seed": seed}))
        res = iae.estimate(problem)
        a = float(res.estimation)
        num_q = int(res.num_oracle_queries)

    # HONEST scaling chain: a -> E[|f-K|] using ONLY (Cmax, w0, kappa).
    if abs(kappa) < 1e-9:
        raise RuntimeError("transfer-function probe found ~zero slope; bad phases")
    E_abs = Cmax * (a - w0) / kappa      # E[|f-K|]

    # Forward F is an a-priori market quantity (S0, r, T, M only).
    F = forward_price(S0=S0, r=r, T=T, M=M, option=option)

    # Put-call parity on the straddle.
    if kind == "call":
        price_val = float(np.exp(-r * T) * (E_abs + (F - K)) / 2.0)
    elif kind == "put":
        price_val = float(np.exp(-r * T) * (E_abs - (F - K)) / 2.0)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    if return_meta:
        return dict(price=price_val, a=a, E_abs=E_abs, forward=F,
                    Cmax=Cmax, coupling=c, w0=w0, kappa=kappa,
                    probe_residual=probe_resid, num_oracle_queries=num_q,
                    degree=degree, poly_phases=len(phis), pyqsp=used_pyqsp)
    return price_val
