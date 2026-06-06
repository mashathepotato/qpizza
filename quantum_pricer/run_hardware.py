"""Run the shallow Fourier route on a Q50 backend (q50_fake by default; q50_hw on-site).
Reports the price recovered from noisy hardware against the exact tree price."""
import argparse
import numpy as np
from qiskit import transpile
from quantum_pricer import oracles, tree, backends


def run(backend_name="q50_fake", M=1, shots=20000, S0=4.2, K=4.2, r=0.03,
        sigma=0.30, T=1.0, n_lambda=12):
    backend = backends.get_backend(backend_name)
    angles = tree.loading_angles(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    values = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    exact = tree.exact_tree_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    spread = max(values.max() - values.min(), 1e-6)
    lambdas = np.linspace(-np.pi, np.pi, n_lambda) / spread
    G = np.empty(len(lambdas), dtype=complex)
    det = M
    for j, lam in enumerate(lambdas):
        gx = _pr0(backend, oracles.fourier_circuit(M, angles, values, K, lam, "X"), det, shots)
        gy = _pr0(backend, oracles.fourier_circuit(M, angles, values, K, lam, "Y"), det, shots)
        G[j] = (2 * gx - 1) + 1j * (2 * gy - 1)
    distinct = np.unique(np.round(values, 9))
    A = np.exp(1j * np.outer(lambdas, distinct - K))
    p_v = np.real(np.linalg.lstsq(A, G, rcond=None)[0])
    price = float(np.exp(-r * T) * np.sum(p_v * np.maximum(distinct - K, 0.0)))
    print(f"[{backend_name}] M={M} shots={shots}  price={price:.4f}  exact={exact:.4f}  "
          f"abs_err={abs(price - exact):.4f}")
    return price, exact


def _pr0(backend, qc, det, shots):
    qc = qc.copy()
    qc.measure_all()
    tqc = transpile(qc, backend, optimization_level=2)
    counts = backend.run(tqc, shots=shots).result().get_counts()
    zero = sum(n for b, n in counts.items() if b[::-1][det] == "0")
    return zero / shots


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="q50_fake")
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--shots", type=int, default=20000)
    args = ap.parse_args()
    run(backend_name=args.backend, M=args.M, shots=args.shots)
