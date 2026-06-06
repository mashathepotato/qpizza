"""Shared circuit primitives used by all quantum routes.

Conventions: path qubits q0..q_{M-1} (q0 = LSB = step 1). Detector / signal / target
ancillas are appended ABOVE the path register (higher qubit index).
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal, UCRYGate


def path_loader(angles):
    """M single-qubit R_Y(theta_i) -> sum_x sqrt(p(x)) |x>. Cost: M gates, exact."""
    M = len(angles)
    qc = QuantumCircuit(M, name="load")
    for i, theta in enumerate(angles):
        qc.ry(theta, i)
    return qc


def phase_oracle_gate(values, K, lam):
    """Controlled diagonal phase: |x>|1>_d -> e^{i lam (f(x)-K)} |x>|1>_d (and |0>_d
    untouched). Returns a gate acting on [detector] + path qubits.
    Built from the classically-known values (exact, no arithmetic register)."""
    entries = np.exp(1j * lam * (np.asarray(values, dtype=float) - K))
    diag = Diagonal(entries).to_gate()          # acts on M path qubits
    return diag.control(1)                       # control = detector (added first)


def fourier_circuit(M, angles, values, K, lam, basis="X"):
    """Full QNDM Fourier circuit: load paths, detector |+>, phase oracle, then rotate
    the detector into the requested measurement basis.
      basis 'X' -> Pr[0] = (1+Re G)/2 ; 'Y' -> Pr[0] = (1+Im G)/2 ; 'Z' -> no rotation.
    Qubit layout: q0..q_{M-1} = paths, q_M = detector."""
    qc = QuantumCircuit(M + 1, name=f"fourier_{basis}")
    qc.compose(path_loader(angles), qubits=range(M), inplace=True)
    det = M
    qc.h(det)
    qc.append(phase_oracle_gate(values, K, lam), [det, *range(M)])
    if basis == "X":
        qc.h(det)
    elif basis == "Y":
        qc.sdg(det)
        qc.h(det)
    elif basis == "Z":
        pass
    else:
        raise ValueError(f"unknown basis {basis!r}")
    return qc


def payoff_amplitude_circuit(angles, payoff, Cmax):
    """The QAE preparation operator A: load paths, then a uniformly-controlled R_Y on a
    target qubit with angle 2 arcsin sqrt(payoff(x)/Cmax). Exact, multiplexed over paths.
    Returns (circuit, target_qubit_index). Pr[target=1] = E[payoff]/Cmax."""
    M = len(angles)
    ratios = np.clip(np.asarray(payoff, dtype=float) / Cmax, 0.0, 1.0)
    ry_angles = 2.0 * np.arcsin(np.sqrt(ratios))   # indexed by path integer x
    qc = QuantumCircuit(M + 1, name="A")
    qc.compose(path_loader(angles), qubits=range(M), inplace=True)
    target = M
    # UCRYGate angles ordered by control state integer (control0 = LSB) -> matches our x.
    # Append order is [target, *controls] for Qiskit's UCRYGate.
    qc.append(UCRYGate(list(ry_angles)), [target, *range(M)])
    return qc, target
