from qiskit import transpile
from quantum_pricer import oracles, tree, backends


def test_fourier_transpiles_to_iqm_basis(base_params):
    M = 2
    angles = tree.loading_angles(M=M, **base_params)
    values = tree.payoff_variable_values(M=M, **base_params)
    qc = oracles.fourier_circuit(M, angles, values, base_params["K"], lam=1.0, basis="X")
    t = transpile(qc, basis_gates=backends.IQM_BASIS, optimization_level=1)
    used = set(t.count_ops())
    assert used <= {"r", "cz", "measure", "barrier"}
