"""Quantum GRU classifier (QGRU) for the next-day-volatility task.

Chen et al. 2020 style quantum GRU: the classical GRU gates (reset, update,
candidate) are each replaced by a variational quantum circuit reused across
timesteps via data re-uploading. Each gate maps [hidden, x_t] -> n_qubits <Z>
values; gating combines them into the next hidden state, and the final hidden
state feeds a small linear head -> sigmoid. Fully simulable on PennyLane
default.qubit; trains with the shared classical_rnn.train loop (torch backprop
through the circuits). The quantum analogue of classical_rnn.GRUClassifier.
"""
import torch
import torch.nn as nn

from qcell import make_gate_layer


class QGRU(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.vqc_r = make_gate_layer(n_qubits, n_layers)   # reset gate
        self.vqc_z = make_gate_layer(n_qubits, n_layers)   # update gate
        self.vqc_n = make_gate_layer(n_qubits, n_layers)   # candidate
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x):                          # x: (B, L)
        B, L = x.shape
        h = torch.zeros(B, self.n_qubits)
        for t in range(L):
            x_t = x[:, t:t + 1]                            # (B, 1)
            v = torch.cat([h, x_t], dim=1)                # (B, n_qubits+1)
            r = torch.sigmoid(self.vqc_r(v))              # reset gate, (B, H)
            z = torch.sigmoid(self.vqc_z(v))              # update gate, (B, H)
            v2 = torch.cat([r * h, x_t], dim=1)           # (B, n_qubits+1)
            n = torch.tanh(self.vqc_n(v2))                # candidate, (B, H)
            h = (1 - z) * n + z * h
        return torch.sigmoid(self.head(h)).squeeze(-1)    # (B,)
