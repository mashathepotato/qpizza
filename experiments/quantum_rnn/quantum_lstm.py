"""Quantum LSTM classifier (QLSTM) for the next-day-volatility task.

The Chen et al. (2020) quantum LSTM: the four classical LSTM gates (forget,
input, candidate, output) are each replaced by a variational quantum circuit,
reused across timesteps with carried hidden and cell states (data re-uploading).
At each step every gate maps [hidden, x_t] -> n_qubits <Z> values; the gated
cell/hidden updates follow the standard LSTM recurrence and the final hidden
state feeds a small linear head -> sigmoid. Fully simulable on PennyLane
default.qubit; trains with the shared classical_rnn.train loop (torch backprop
through the circuits). The quantum analogue of classical_rnn.LSTMClassifier.
"""
import torch
import torch.nn as nn

from qcell import make_gate_layer


class QLSTM(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.vqc_f = make_gate_layer(n_qubits, n_layers)   # forget gate
        self.vqc_i = make_gate_layer(n_qubits, n_layers)   # input gate
        self.vqc_g = make_gate_layer(n_qubits, n_layers)   # candidate cell
        self.vqc_o = make_gate_layer(n_qubits, n_layers)   # output gate
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x):                          # x: (B, L)
        B, L = x.shape
        h = torch.zeros(B, self.n_qubits)
        c = torch.zeros(B, self.n_qubits)
        for t in range(L):
            v = torch.cat([h, x[:, t:t + 1]], dim=1)   # (B, n_qubits+1)
            f = torch.sigmoid(self.vqc_f(v))            # forget gate
            i = torch.sigmoid(self.vqc_i(v))            # input gate
            g = torch.tanh(self.vqc_g(v))               # candidate cell
            o = torch.sigmoid(self.vqc_o(v))            # output gate
            c = f * c + i * g
            h = o * torch.tanh(c)
        return torch.sigmoid(self.head(h)).squeeze(-1)  # (B,)
