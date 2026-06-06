"""Shared variational-quantum gate cell for the quantum recurrent models.

A single reusable VQC (PennyLane `default.qubit`, torch interface, backprop) that
maps a (state, scalar-input) vector to an `n_qubits`-dim output via data
re-uploading: the carried state is angle-encoded (RY), the scalar input is
re-uploaded (RX on every wire), then a StronglyEntanglingLayers block, then
<Z> on each wire. Every quantum recurrent gate (QRNN recurrence, QGRU/QLSTM
gates, Chen et al. 2020) is one of these layers.
"""
import pennylane as qml


def make_gate_layer(n_qubits=4, n_layers=2):
    """Return a `qml.qnn.TorchLayer` taking inputs of shape (..., n_qubits+1)
    (first n_qubits = carried state, last = scalar timestep input) and returning
    (..., n_qubits) expectation values in [-1, 1]."""
    dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)          # encode carried state
        for i in range(n_qubits):
            qml.RX(inputs[..., n_qubits], wires=i)    # re-upload scalar input
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return qml.qnn.TorchLayer(qnode, {"weights": shape})
