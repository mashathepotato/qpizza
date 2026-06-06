import numpy as np
from qiskit.quantum_info import Statevector
from quantum_pricer import oracles, tree


def test_loader_reproduces_path_probabilities(base_params):
    M = 3
    angles = tree.loading_angles(M=M, **base_params)
    qc = oracles.path_loader(angles)
    probs = Statevector(qc).probabilities()           # over M path qubits
    expected = tree.path_probabilities(M=M, **base_params)
    assert np.allclose(probs, expected, atol=1e-9)


def test_phase_oracle_writes_relative_phase(base_params):
    # detector |+> branch should pick up e^{i lam (f(x)-K)} on |1>_d, |0>_d untouched
    M = 2
    lam = 0.01
    vals = tree.payoff_variable_values(M=M, **base_params)
    qc = oracles.fourier_circuit(M=M, angles=[0.0, 0.0], values=vals,
                                 K=base_params["K"], lam=lam, basis="Z")
    # angles=0 => path register is |00> (x=0). Detector |+> -> phase only on |1>_d.
    sv = Statevector(qc)
    # qubit layout: path q0,q1 then detector q2 (most significant). amplitude of |0>_d vs |1>_d:
    amp0 = sv.data[0b000]     # x=0, detector 0
    amp1 = sv.data[0b100]     # x=0, detector 1
    phase = np.angle(amp1 / amp0)
    assert np.isclose(phase % (2 * np.pi),
                      (lam * (vals[0] - base_params["K"])) % (2 * np.pi), atol=1e-6)


def test_payoff_amplitude_encodes_expected_payoff(base_params):
    # A operator: Pr[target=1] == E[max(f-K,0)]/Cmax
    M = 3
    angles = tree.loading_angles(M=M, **base_params)
    vals = tree.payoff_variable_values(M=M, **base_params)
    payoff = np.maximum(vals - base_params["K"], 0.0)
    Cmax = payoff.max() * 1.0001
    qc, target_idx = oracles.payoff_amplitude_circuit(angles, payoff, Cmax)
    sv = Statevector(qc)
    probs = sv.probabilities([target_idx])
    p = tree.path_probabilities(M=M, **base_params)
    expected_a = float(np.sum(p * payoff) / Cmax)
    assert np.isclose(probs[1], expected_a, atol=1e-9)
