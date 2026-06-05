def test_core_imports():
    import qiskit
    import qiskit_aer
    import qiskit_algorithms
    import qiskit_finance
    import qiskit_optimization
    import iqm.qiskit_iqm as qiskit_iqm  # qiskit-iqm uses iqm.* namespace packages
    import pennylane
    import sklearn
    assert qiskit_aer.AerSimulator is not None
