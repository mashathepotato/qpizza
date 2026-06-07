"""
Inspect-only module: PRINT the arithmetic-Asian characteristic-function circuit.
================================================================================
See the circuit (no simulation, no Q50, no matplotlib -- only qiskit):

    python show_circuit.py            # X-basis circuit (paths + running weight + detector)
    python show_circuit.py --basis Y  # Y-basis
    python show_circuit.py --mpl      # also save asian_char_circuit.png

Fully self-contained: it does NOT run the simulation/Q50 code in asian_pricing.py.
"""

import argparse
import math
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate

M = 3
LAM0 = 0.5
S0, K, R, SIGMA, T = 4.20, 4.20, 0.03, 0.30, 1.0


def crr_params():
    dt = T / M
    drift = (R - 0.5 * SIGMA ** 2) * dt
    vol = SIGMA * math.sqrt(dt)
    u = math.exp(drift + vol)
    d = math.exp(drift - vol)
    q = (math.exp(R * dt) - d) / (u - d)
    return u, d, q


def S_at(w, t):
    u, d, _ = crr_params()
    return S0 * (u ** w) * (d ** (t - w))


def n_weight_qubits():
    return int(math.ceil(math.log2(M + 1)))


def controlled_increment(qc, ctrl, reg):
    n = len(reg)
    for k in range(n - 1, 0, -1):
        qc.mcx([ctrl] + reg[:k], reg[k])
    qc.cx(ctrl, reg[0])


def controlled_decrement(qc, ctrl, reg):
    qc.cx(ctrl, reg[0])
    for k in range(1, len(reg)):
        qc.mcx([ctrl] + reg[:k], reg[k])


def build_asian_char(lam, basis):
    n_w = n_weight_qubits()
    det = M + n_w
    wreg = [M + j for j in range(n_w)]
    _, _, q = crr_params()
    qc = QuantumCircuit(M + n_w + 1, name=f"asian_char_{basis}")
    for i in range(M):
        qc.ry(2.0 * math.asin(math.sqrt(q)), i)
    qc.h(det)
    for i in range(M):
        t = i + 1
        controlled_increment(qc, i, wreg)
        entries = np.ones(2 ** n_w, dtype=complex)
        for w in range(t + 1):
            entries[w] = np.exp(1j * lam * S_at(w, t) / M)
        qc.append(DiagonalGate(list(entries)).control(1), [det, *wreg])
    qc.p(-lam * K, det)
    for i in range(M):
        controlled_decrement(qc, i, wreg)
    if basis == "Y":
        qc.sdg(det)
    qc.h(det)
    qc.measure_all()
    return qc, det


def main():
    ap = argparse.ArgumentParser(description="Print the Asian char-function circuit.")
    ap.add_argument("--basis", default="X", choices=["X", "Y"])
    ap.add_argument("--mpl", action="store_true", help="also save a PNG (needs matplotlib)")
    args = ap.parse_args()

    qc, det = build_asian_char(LAM0, args.basis)
    print("=" * 66)
    print(f"Arithmetic Asian char-function circuit  "
          f"({qc.num_qubits} qubits, detector q{det}, basis={args.basis}, lambda={LAM0})")
    print("=" * 66)
    print(qc.draw(output="text", fold=-1))

    if args.mpl:
        try:
            from qiskit.visualization import circuit_drawer
            circuit_drawer(qc, output="mpl", fold=-1).savefig("asian_char_circuit.png", dpi=150)
            print("\nSaved asian_char_circuit.png")
        except Exception as e:
            print(f"[WARN] mpl drawing failed (non-fatal): {repr(e)}")


if __name__ == "__main__":
    main()
