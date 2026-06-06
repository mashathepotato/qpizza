"""TDD tests for triage/q50_noise.py — NISQ reality check (ideal vs IQMFakeAphrodite)."""
import pytest
from qiskit import QuantumCircuit


def test_representative_circuits_present():
    from triage.q50_noise import representative_circuits
    circuits = representative_circuits()
    assert set(circuits.keys()) == {"qae", "qaoa", "fraud_qml"}
    for name, qc in circuits.items():
        assert isinstance(qc, QuantumCircuit), f"{name} must be a QuantumCircuit"
        # Each circuit must have at least one measurement
        assert qc.num_clbits > 0, f"{name} circuit must have measurements"


def test_noise_degradation_returns_valid_tvd():
    from triage.q50_noise import representative_circuits, noise_degradation
    circuits = representative_circuits()
    result = noise_degradation(circuits["qae"], shots=1024)
    assert "tvd" in result
    assert "n_qubits" in result
    assert "ideal_top" in result
    assert "noisy_top" in result
    assert 0.0 <= result["tvd"] <= 1.0, f"TVD must be in [0,1], got {result['tvd']}"
    assert result["n_qubits"] == 1


def test_run_all_covers_all_methods():
    from triage.q50_noise import run_all
    results = run_all(shots=512)
    assert set(results.keys()) == {"qae", "qaoa", "fraud_qml"}
    for name, res in results.items():
        assert "tvd" in res, f"Missing tvd for {name}"
        assert 0.0 <= res["tvd"] <= 1.0, f"TVD out of range for {name}: {res['tvd']}"
        assert isinstance(res["tvd"], float)
