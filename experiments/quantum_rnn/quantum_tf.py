"""Quantum Transformer (quantum self-attention) classifier — QSANN-style
(Li et al. 2022). The query/key/value projections of a self-attention block are
data-reuploading variational quantum circuits instead of linear layers; the
scaled-dot-product attention and softmax stay classical. Embed each return to an
n_qubit token vector + sinusoidal positional encoding, compute Q/K/V via three
VQCs, attend, mean-pool, linear head -> sigmoid. Fully simulable on
default.qubit; trains with the shared classical_rnn.train loop (torch backprop).
The quantum analogue of classical_tf.TransformerClassifier.
"""
import math

import torch
import torch.nn as nn

from qcell import make_token_layer
from classical_tf import sinusoidal_pe


class QuantumTransformer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, L=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.d = n_qubits
        self.embed = nn.Linear(1, n_qubits)
        self.register_buffer("pe", sinusoidal_pe(L, n_qubits))
        self.vqc_q = make_token_layer(n_qubits, n_layers)   # quantum Q projection
        self.vqc_k = make_token_layer(n_qubits, n_layers)   # quantum K projection
        self.vqc_v = make_token_layer(n_qubits, n_layers)   # quantum V projection
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x):                          # x: (B, L)
        B, L = x.shape
        h = self.embed(x.unsqueeze(-1)) + self.pe[:L]   # (B, L, d)
        flat = h.reshape(B * L, self.d)
        Q = self.vqc_q(flat).reshape(B, L, self.d)
        K = self.vqc_k(flat).reshape(B, L, self.d)
        V = self.vqc_v(flat).reshape(B, L, self.d)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d)  # (B, L, L)
        ctx = torch.matmul(torch.softmax(scores, dim=-1), V)             # (B, L, d)
        return torch.sigmoid(self.head(ctx.mean(dim=1))).squeeze(-1)     # (B,)
