import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from quantum_pricer import hamming, oracles, tree


def test_counter_computes_popcount(base_params):
    """For M=4, n_w=3: loading every classical path bitstring x and applying the
    Hamming counter must leave the weight register holding popcount(x)."""
    M = 4
    n_w = hamming.n_weight_qubits(M)
    assert n_w == 3
    for x in range(2 ** M):
        qc = QuantumCircuit(M + n_w)
        for i in range(M):
            if (x >> i) & 1:
                qc.x(i)
        wreg = [M + j for j in range(n_w)]
        hamming._count_weight(qc, range(M), wreg)
        sv = Statevector(qc)
        probs = sv.probabilities()
        idx = int(np.argmax(probs))
        # extract weight-register bits from the basis-state integer
        weight = (idx >> M) & ((1 << n_w) - 1)
        assert weight == bin(x).count("1")


@pytest.mark.parametrize("M", [3, 4, 5])
def test_hamming_qae_amplitude_matches_naive(M, base_params):
    """Pr[target=1] from the Hamming QAE operator must equal the naive one to 1e-9."""
    angles = tree.loading_angles(M=M, **base_params)
    K = base_params["K"]
    ST_by_weight = hamming.terminal_values_by_weight(M=M, **base_params)
    payoff_by_weight = np.maximum(ST_by_weight - K, 0.0)
    Cmax = float(payoff_by_weight.max()) * 1.0001

    qc_h, target_h = hamming.payoff_amplitude_circuit_hamming(
        M, angles, payoff_by_weight, Cmax)
    a_h = Statevector(qc_h).probabilities([target_h])[1]

    values = tree.payoff_variable_values(M=M, **base_params)
    payoff = np.maximum(values - K, 0.0)
    qc_n, target_n = oracles.payoff_amplitude_circuit(angles, payoff, Cmax)
    a_n = Statevector(qc_n).probabilities([target_n])[1]

    assert np.isclose(a_h, a_n, atol=1e-9)


def test_hamming_fourier_G_matches_naive(base_params):
    """Re G from the Hamming Fourier circuit (basis X) must equal the naive one to 1e-9."""
    M = 4
    angles = tree.loading_angles(M=M, **base_params)
    K = base_params["K"]
    ST_by_weight = hamming.terminal_values_by_weight(M=M, **base_params)
    values = tree.payoff_variable_values(M=M, **base_params)

    for lam in [0.001, 0.005, 0.01, 0.05, -0.02]:
        qc_h = hamming.fourier_circuit_hamming(M, angles, ST_by_weight, K, lam, basis="X")
        re_h = 2.0 * Statevector(qc_h).probabilities([qc_h.num_qubits - 1])[0] - 1.0

        qc_n = oracles.fourier_circuit(M, angles, values, K, lam, basis="X")
        re_n = 2.0 * Statevector(qc_n).probabilities([qc_n.num_qubits - 1])[0] - 1.0

        assert np.isclose(re_h, re_n, atol=1e-9)


def test_hamming_price_matches_exact_tree(base_params):
    """End-to-end: Hamming QAE amplitude -> price within 1e-6 of the exact tree."""
    M = 5
    angles = tree.loading_angles(M=M, **base_params)
    K = base_params["K"]
    r = base_params["r"]
    T = base_params["T"]
    ST_by_weight = hamming.terminal_values_by_weight(M=M, **base_params)
    payoff_by_weight = np.maximum(ST_by_weight - K, 0.0)
    Cmax = float(payoff_by_weight.max()) * 1.0001

    qc, target = hamming.payoff_amplitude_circuit_hamming(M, angles, payoff_by_weight, Cmax)
    a = Statevector(qc).probabilities([target])[1]
    price = float(np.exp(-r * T) * Cmax * a)

    expected = tree.exact_tree_price(M=M, option="european", kind="call", **base_params)
    assert np.isclose(price, expected, atol=1e-6)
