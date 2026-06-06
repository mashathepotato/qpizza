"""Classical Transformer-encoder classifier for the next-day-volatility task.

A small self-attention encoder: embed each scalar return to d_model, add a
sinusoidal positional encoding, run TransformerEncoder layers (multi-head
attention + FFN), mean-pool over the sequence, linear head -> sigmoid. Same
training loop and seeding as every other model in the benchmark.
"""
import math

import torch
import torch.nn as nn


def sinusoidal_pe(L, d):
    """Standard fixed sinusoidal positional encoding, shape (L, d). Parameter-free
    so it is identical (and fair) across the classical and quantum transformers."""
    pe = torch.zeros(L, d)
    pos = torch.arange(L).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerClassifier(nn.Module):
    def __init__(self, d_model=4, nhead=2, dim_ff=8, n_layers=1, L=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Linear(1, d_model)
        self.register_buffer("pe", sinusoidal_pe(L, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_ff, dropout=0.0,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                          # x: (B, L)
        L = x.shape[1]
        h = self.embed(x.unsqueeze(-1)) + self.pe[:L]   # (B, L, d_model)
        h = self.encoder(h)
        return torch.sigmoid(self.head(h.mean(dim=1))).squeeze(-1)  # (B,)
