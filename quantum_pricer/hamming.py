"""Hamming-weight optimization (paper Sec 3 remark). On a recombining tree the
European terminal value S_T depends only on the up-move count w, so the phase
oracle / payoff rotation can act on a ceil(log2(M+1))-qubit weight register
instead of being built from all 2^M paths -> poly(M) depth instead of 2^M.

Weight register convention: reg[0] = LSB. n_w = ceil(log2(M+1)).
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal, UCRYGate
from quantum_pricer.tree import crr_params, loading_angles


def n_weight_qubits(M):
    return int(np.ceil(np.log2(M + 1)))


def controlled_increment(qc, ctrl, reg):
    """Add 1 to reg (qubit-index list, reg[0]=LSB) if ctrl=1. Ripple incrementer.
    Processes high->low so carries read pre-increment low bits."""
    n = len(reg)
    for k in range(n - 1, 0, -1):
        qc.mcx([ctrl] + reg[:k], reg[k])
    qc.cx(ctrl, reg[0])


def _count_weight(qc, path_qubits, weight_reg):
    for i in path_qubits:
        controlled_increment(qc, i, list(weight_reg))


def _uncount_weight(qc, path_qubits, weight_reg):
    for i in reversed(list(path_qubits)):
        controlled_increment(qc, i, list(weight_reg))


def terminal_values_by_weight(S0, K, r, sigma, T, M):
    """European S_T(w) = S0 u^w d^(M-w) for w=0..M (length M+1)."""
    u, d, _ = crr_params(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    return np.array([S0 * u**w * d**(M - w) for w in range(M + 1)])


def fourier_circuit_hamming(M, angles, ST_by_weight, K, lam, basis="X"):
    """Hamming-weight European Fourier circuit. Same semantics as
    oracles.fourier_circuit but poly(M) depth. Layout: q0..q_{M-1}=paths,
    next n_w = weight register, last qubit = detector."""
    n_w = n_weight_qubits(M)
    det = M + n_w
    qc = QuantumCircuit(M + n_w + 1, name=f"fourier_ham_{basis}")
    for i, a in enumerate(angles):
        qc.ry(a, i)
    qc.h(det)
    wreg = [M + j for j in range(n_w)]
    _count_weight(qc, range(M), wreg)
    entries = np.ones(2 ** n_w, dtype=complex)
    for w in range(M + 1):
        entries[w] = np.exp(1j * lam * (ST_by_weight[w] - K))
    qc.append(Diagonal(entries).to_gate().control(1), [det, *wreg])
    _uncount_weight(qc, range(M), wreg)
    if basis == "X":
        qc.h(det)
    elif basis == "Y":
        qc.sdg(det); qc.h(det)
    elif basis == "Z":
        pass
    else:
        raise ValueError(f"unknown basis {basis!r}")
    return qc


def payoff_amplitude_circuit_hamming(M, angles, payoff_by_weight, Cmax):
    """Hamming-weight European QAE preparation operator A. Same semantics as
    oracles.payoff_amplitude_circuit but poly(M) depth. Returns (qc, target_idx).
    Pr[target=1] = E[max(S_T-K,0)]/Cmax."""
    n_w = n_weight_qubits(M)
    target = M + n_w
    qc = QuantumCircuit(M + n_w + 1, name="A_ham")
    for i, a in enumerate(angles):
        qc.ry(a, i)
    wreg = [M + j for j in range(n_w)]
    _count_weight(qc, range(M), wreg)
    ang = np.zeros(2 ** n_w)
    for w in range(M + 1):
        ang[w] = 2.0 * np.arcsin(np.sqrt(min(payoff_by_weight[w] / Cmax, 1.0)))
    qc.append(UCRYGate(list(ang)), [target, *wreg])
    _uncount_weight(qc, range(M), wreg)
    return qc, target
