"""Quantum autoencoder anomaly detector for NOKIA.HE return windows.

A Romero-Olson-Aspuru-Guzik (2017) style quantum autoencoder: angle-encode a
length-k window of (z-scored) daily log-returns, run a parameterized encoder,
and train so the `trash` qubits disentangle to |0>. The reconstruction
infidelity ``1 - P(trash = |0...0>)`` is the anomaly score -- high on windows
unlike the calm training regime. A numpy-SVD PCA reconstructor is the honest
classical baseline. Everything runs on PennyLane ``default.qubit`` (statevector,
fully simulable).
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# ---------------------------------------------------------------------------
# Windowing + scaling (training-stats only, no leakage)
# ---------------------------------------------------------------------------
def make_windows(returns, k=4):
    """Sliding windows of `k` consecutive returns, stride 1 -> shape (N-k+1, k)."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns) - k + 1
    if n <= 0:
        raise ValueError(f"need at least k={k} returns, got {len(returns)}")
    return np.stack([returns[i:i + k] for i in range(n)])


def zscore_fit(train_windows):
    """Per-column mean/std from the training windows (std floored to avoid /0)."""
    w = np.asarray(train_windows, dtype=float)
    mean = w.mean(axis=0)
    std = w.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def zscore_apply(windows, mean, std):
    return (np.asarray(windows, dtype=float) - mean) / std


# ---------------------------------------------------------------------------
# Quantum autoencoder
# ---------------------------------------------------------------------------
class QuantumAutoencoder:
    """Trash-qubit quantum autoencoder. Latent wires are the first `n_latent`;
    the remaining wires are trash and are pushed toward |0...0>."""

    def __init__(self, n_qubits=4, n_latent=2, n_layers=2, seed=0, angle_scale=1.0):
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_trash = n_qubits - n_latent
        self.n_layers = n_layers
        self.angle_scale = angle_scale
        self.trash_wires = list(range(n_latent, n_qubits))
        self.n_params = 2 * n_qubits * n_layers  # RY + RZ per wire per layer
        self.dev = qml.device("default.qubit", wires=n_qubits)
        rng = np.random.default_rng(seed)
        self.params = pnp.array(
            rng.normal(0.0, 0.1, size=(n_layers, n_qubits, 2)), requires_grad=True
        )
        self._qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, params, window):
        for i in range(self.n_qubits):
            qml.RY(self.angle_scale * window[i], wires=i)
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(params[l, i, 0], wires=i)
                qml.RZ(params[l, i, 1], wires=i)
            for i in range(self.n_qubits):
                qml.CZ(wires=[i, (i + 1) % self.n_qubits])
        return qml.probs(wires=self.trash_wires)

    def _p_trash_zero(self, params, window):
        return self._qnode(params, window)[0]  # P(|0...0>)

    def loss(self, params, windows):
        windows = np.asarray(windows, dtype=float)
        total = 0.0
        for w in windows:
            total = total + (1.0 - self._p_trash_zero(params, w))
        return total / len(windows)

    def train(self, windows, steps=60, lr=0.1):
        """Adam on the mean trash infidelity. Returns the loss history."""
        windows = np.asarray(windows, dtype=float)
        opt = qml.AdamOptimizer(stepsize=lr)
        params = self.params
        history = []
        cost = lambda p: self.loss(p, windows)
        for _ in range(steps):
            params, c = opt.step_and_cost(cost, params)
            history.append(float(c))
        self.params = params
        return history

    def score(self, window):
        """Anomaly score = reconstruction infidelity in [0, 1]."""
        window = np.asarray(window, dtype=float)
        return float(1.0 - self._p_trash_zero(self.params, window))


# ---------------------------------------------------------------------------
# Classical PCA baseline (numpy SVD; no sklearn)
# ---------------------------------------------------------------------------
class PCAReconstructor:
    """Linear autoencoder baseline: keep `n_components` principal directions,
    score = squared reconstruction residual norm."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, windows):
        X = np.asarray(windows, dtype=float)
        self.mean_ = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[: self.n_components]  # (n_components, k)
        return self

    def score(self, window):
        x = np.asarray(window, dtype=float) - self.mean_
        recon = self.components_.T @ (self.components_ @ x)
        resid = x - recon
        return float(resid @ resid)

    @property
    def n_params(self):
        return int(self.components_.size)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def roc_auc(scores, labels):
    """Tie-aware ROC-AUC via the Mann-Whitney U statistic (rank formulation)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    avg_ranks = np.empty(len(scores))
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_ranks[i:j + 1] = (i + j) / 2.0 + 1.0  # 1-based average rank
        i = j + 1
    ranks = np.empty(len(scores))
    ranks[order] = avg_ranks
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
