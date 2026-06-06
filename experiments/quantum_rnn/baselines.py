"""Floor baselines: the bar every fancier model must clear.

  persistence_scores -- score = magnitude of the most recent return. The trivial
                        "volatility persists" heuristic; no training.
  LogisticClassifier  -- logistic regression on the flattened window (a linear,
                        non-recurrent learned baseline). Trains via the shared
                        classical_rnn.train loop.
"""
import numpy as np
import torch
import torch.nn as nn


def persistence_scores(X_raw):
    """Anomaly score = |most recent return| (last column of the raw sequence)."""
    return np.abs(np.asarray(X_raw, dtype=float)[:, -1])


class LogisticClassifier(nn.Module):
    def __init__(self, n_features, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):                       # x: (B, n_features)
        return torch.sigmoid(self.linear(x)).squeeze(-1)
