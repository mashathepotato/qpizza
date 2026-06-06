"""Shared sequence-task harness for the NOKIA.HE next-day-volatility benchmark.

One honest supervised task used by every model (classical RNN/GRU/LSTM, quantum
QRNN/QGRU/QLSTM, GARCH/AR baselines): given a length-L sequence of log-returns
ending day t, classify whether day t+1 is a top-decile |return| day (volatility
event). The target return (day t+1) is NOT in the sequence -> no leakage. All
models are scored by held-out ROC-AUC on identical splits.
"""
import numpy as np


def make_sequences(returns, L):
    """Sliding length-L windows, stride 1 -> shape (len-L+1, L)."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns) - L + 1
    if n <= 0:
        raise ValueError(f"need at least L={L} returns, got {len(returns)}")
    return np.stack([returns[i:i + L] for i in range(n)])


def zscore_fit(train_sequences):
    """Scalar mean/std over all training-sequence elements (returns are
    position-exchangeable, so a single scale is appropriate)."""
    a = np.asarray(train_sequences, dtype=float)
    mean = float(a.mean())
    std = float(a.std())
    return mean, (std if std > 1e-12 else 1.0)


def zscore_apply(sequences, mean, std):
    return (np.asarray(sequences, dtype=float) - mean) / std


def build_task_from_returns(returns, L=10, train_frac=0.5, proxy_q=0.9):
    """Assemble the supervised next-day-volatility task from a return series.

    Returns a dict: X (N,L) z-scored sequences, y (N,) proxy labels, next_abs
    (N,) next-day |return|, n_train split index, thresh, mean/std."""
    returns = np.asarray(returns, dtype=float)
    seqs = make_sequences(returns, L)[:-1]          # drop last: no next-day target
    next_abs = np.abs(returns[L:])                   # day t+1 |return| per sequence
    n = len(seqs)
    n_train = int(n * train_frac)

    mean, std = zscore_fit(seqs[:n_train])
    X = zscore_apply(seqs, mean, std)

    thresh = float(np.quantile(next_abs[:n_train], proxy_q))
    y = (next_abs >= thresh).astype(int)
    return dict(X=X, X_raw=seqs, y=y, next_abs=next_abs, n_train=n_train,
                thresh=thresh, mean=mean, std=std)


def roc_auc(scores, labels):
    """Tie-aware ROC-AUC via the Mann-Whitney U rank statistic."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    avg = np.empty(len(scores))
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s[j + 1] == s[i]:
            j += 1
        avg[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(scores))
    ranks[order] = avg
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fetch_prices(period="1y"):
    """Real NOKIA.HE daily closes, or a labelled synthetic fallback offline."""
    try:
        import yfinance as yf
        v = yf.Ticker("NOKIA.HE").history(period=period)["Close"].dropna().values
        if len(v) > 60:
            return np.asarray(v, float), "yfinance NOKIA.HE 1y daily"
    except Exception as exc:
        print(f"[synthetic fallback] {exc}")
    rng = np.random.default_rng(0)
    lr = rng.normal(0, 0.012, 250)
    lr[120:160] = rng.normal(0, 0.05, 40)            # injected volatile burst
    return 4.20 * np.exp(np.cumsum(np.concatenate([[0.0], lr]))), \
        "[SYNTHETIC FALLBACK] not real market data"
