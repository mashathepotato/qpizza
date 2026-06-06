"""QNDM Fourier route (paper Sec 5a). Estimate the characteristic function
G(lam)=E[e^{i lam (f-K)}] on a lambda-grid via X/Y detector measurements, then recover
the discrete payoff-variable distribution (the tree has few distinct f-values) by
least-squares inversion of the characteristic-function system, and price.

NOTE (honesty): with plain sampling, each G(lam) is a Bernoulli estimate -> O(1/eps^2)
shots, parallel to classical MC. The Fourier route's advantage is qubit count / shallow
depth / exact loading and Q50 feasibility, NOT the eps-scaling. The 1/eps scaling lives
in the QAE and QSVT routes."""
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from quantum_pricer import hamming, oracles, tree


def _g_exact(qc_x, qc_y):
    re = 2.0 * Statevector(qc_x).probabilities([qc_x.num_qubits - 1])[0] - 1.0
    im = 2.0 * Statevector(qc_y).probabilities([qc_y.num_qubits - 1])[0] - 1.0
    return re + 1j * im


def _g_shots(qc_x, qc_y, shots, sim, seed):
    def pr0(qc):
        c = qc.copy(); c.measure_all()
        # AerSimulator does not natively understand the controlled-Diagonal gate;
        # transpile to its basis before running the shot-based simulation.
        c = transpile(c, sim)
        res = sim.run(c, shots=shots, seed_simulator=seed).result().get_counts()
        det = qc.num_qubits - 1
        zero = sum(n for b, n in res.items() if b[::-1][det] == "0")
        return zero / shots
    return (2 * pr0(qc_x) - 1) + 1j * (2 * pr0(qc_y) - 1)


def estimate_G(M, angles, values, K, lambdas, shots=None, seed=None):
    """Return G(lam) for each lam in lambdas."""
    sim = AerSimulator() if shots else None
    out = np.empty(len(lambdas), dtype=complex)
    for j, lam in enumerate(lambdas):
        qc_x = oracles.fourier_circuit(M, angles, values, K, lam, basis="X")
        qc_y = oracles.fourier_circuit(M, angles, values, K, lam, basis="Y")
        out[j] = _g_exact(qc_x, qc_y) if shots is None \
            else _g_shots(qc_x, qc_y, shots, sim, seed)
    return out


def estimate_G_hamming(M, angles, ST_by_weight, K, lambdas, shots=None, seed=None):
    """Hamming-weight European analogue of estimate_G: poly(M)-depth Fourier
    circuit acting on a log-sized weight register. Detector is the last qubit, so
    the same _g_exact / _g_shots readout applies."""
    sim = AerSimulator() if shots else None
    out = np.empty(len(lambdas), dtype=complex)
    for j, lam in enumerate(lambdas):
        qc_x = hamming.fourier_circuit_hamming(M, angles, ST_by_weight, K, lam, basis="X")
        qc_y = hamming.fourier_circuit_hamming(M, angles, ST_by_weight, K, lam, basis="Y")
        out[j] = _g_exact(qc_x, qc_y) if shots is None \
            else _g_shots(qc_x, qc_y, shots, sim, seed)
    return out


def price(S0, K, r, sigma, T, M, option="european", kind="call",
          n_lambda=24, shots=None, seed=None, use_hamming=False):
    angles = tree.loading_angles(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    if use_hamming:
        assert option == "european", "Hamming-weight route is European-only"
        # On a recombining tree S_T depends only on the up-count w; the M+1 distinct
        # ST_by_weight values are exactly the recovery support.
        ST_by_weight = hamming.terminal_values_by_weight(S0=S0, K=K, r=r, sigma=sigma,
                                                         T=T, M=M)
        values = ST_by_weight
    else:
        values = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                             option=option)
    distinct = np.unique(np.round(values, 9))        # tree support of f
    # lambda grid scaled to the spread of (f-K) so phases stay informative
    spread = max(distinct.max() - distinct.min(), 1e-6)
    lambdas = np.linspace(-np.pi, np.pi, n_lambda) / spread
    if use_hamming:
        G = estimate_G_hamming(M, angles, ST_by_weight, K, lambdas, shots=shots, seed=seed)
    else:
        G = estimate_G(M, angles, values, K, lambdas, shots=shots, seed=seed)
    # Recover p_v on the known support: G(lam) = sum_v p_v e^{i lam (v-K)}
    A = np.exp(1j * np.outer(lambdas, distinct - K))   # (n_lambda, n_support)
    p_v, *_ = np.linalg.lstsq(A, G, rcond=None)
    p_v = np.real(p_v)
    payoff = np.maximum(distinct - K, 0.0) if kind == "call" \
        else np.maximum(K - distinct, 0.0)
    return float(np.exp(-r * T) * np.sum(p_v * payoff))
