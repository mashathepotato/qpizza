import pytest
from backends import get_backend, BackendUnavailable


def test_local_aer_runs_a_bell_circuit():
    from qiskit import QuantumCircuit, transpile
    backend = get_backend("local_aer")
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
    result = backend.run(transpile(qc, backend), shots=256).result()
    counts = result.get_counts()
    assert sum(counts.values()) == 256
    assert set(counts) <= {"00", "11", "01", "10"}


def test_q50_fake_is_iqm_fake_backend():
    backend = get_backend("q50_fake")
    names = set(backend.target.operation_names)
    assert "cz" in names


def test_q50_hw_guarded_without_credentials():
    with pytest.raises(BackendUnavailable):
        get_backend("q50_hw")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("nonsense")
