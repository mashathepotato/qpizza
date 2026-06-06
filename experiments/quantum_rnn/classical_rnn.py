"""Classical PyTorch recurrent classifiers for the next-day-volatility task.

RNN / GRU / LSTM, each: input_size=1 -> hidden -> last hidden state ->
Linear(hidden, 1) -> sigmoid. Same training loop and seeding as every other
model in the benchmark so the comparison is fair.
"""
import numpy as np
import torch
import torch.nn as nn

_CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class _RecurrentClassifier(nn.Module):
    kind = "rnn"

    def __init__(self, hidden=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.rnn = _CELLS[self.kind](input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (B, L)
        out, _ = self.rnn(x.unsqueeze(-1))      # (B, L, hidden)
        return torch.sigmoid(self.head(out[:, -1, :])).squeeze(-1)  # (B,)


class RNNClassifier(_RecurrentClassifier):
    kind = "rnn"


class GRUClassifier(_RecurrentClassifier):
    kind = "gru"


class LSTMClassifier(_RecurrentClassifier):
    kind = "lstm"


def train(model, X, y, epochs=120, lr=0.05, seed=0, weight_decay=0.0):
    """Full-batch Adam on BCE loss. `weight_decay` is L2 regularization (important
    on this tiny dataset, where the larger classical models overfit badly without
    it). Returns the loss history."""
    torch.manual_seed(seed)
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossfn = nn.BCELoss()
    history = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossfn(model(Xt), yt)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


def scores(model, X):
    """Predicted event probabilities (anomaly scores) for sequences X."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(np.asarray(X), dtype=torch.float32)).numpy()


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))
