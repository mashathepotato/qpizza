"""Quantum recurrent classifier (QRNN) for the next-day-volatility task.

A variational quantum circuit reused across timesteps with a carried hidden
state (data re-uploading). At each step the cell maps [hidden, x_t] -> new hidden
(n_qubits <Z> values); the final hidden state feeds a small linear head ->
sigmoid. Fully simulable on PennyLane default.qubit; trains with the shared
classical_rnn.train loop (torch backprop through the circuit). The quantum analogue
of classical_rnn.RNNClassifier.
"""
import torch
import torch.nn as nn

from qcell import make_gate_layer


class QRNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.cell = make_gate_layer(n_qubits, n_layers)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x):                          # x: (B, L)
        B, L = x.shape
        h = torch.zeros(B, self.n_qubits)
        for t in range(L):
            v = torch.cat([h, x[:, t:t + 1]], dim=1)   # (B, n_qubits+1)
            h = self.cell(v)                            # (B, n_qubits)
        return torch.sigmoid(self.head(h)).squeeze(-1)  # (B,)
