"""
TRIVIAL, hardware-runnable arithmetic-Asian pricing  (Route I, QNDM phase encoding)
================================================================================
This is the SMALL sibling of asian_pricing.py, designed to actually run on the
real IQM Q50 NISQ device.  The full M=3 circuit (6 qubits) is correct in
simulation but FAILS on hardware: per step it uses a 3-qubit controlled
DiagonalGate plus MCX increment/decrement, which transpile to a deep CZ+SWAP
chain that decoheres on Q50 (observed G = -0.11 +0.34j vs exact +0.90 +0.03j).

How we make it trivial WITHOUT leaving the paper:
  * Use the smallest genuinely PATH-DEPENDENT Asian case, M = 2.  With two steps
    the paths 01 and 10 have different averages, so it is a real Asian option
    (not a European in disguise).
  * At M = 2 no Hamming-weight register is needed.  The QNDM phase
        phi(x) = lambda (Sbar(x) - K),     Sbar(x) = (S_1(x) + S_2(x)) / 2
    is a function on only 4 points, so it decomposes EXACTLY into a handful of
    Pauli-Z phases on {path0, path1, detector} -- exactly the e^{i lambda Z...Z}
    detector<->path couplings of the paper's read-out circuit (Appendix
    Fig. "QNDM read-out").  Detector-controlled phase => an 8-entry diagonal on
    3 qubits, which transpiles to a shallow Rz + Rzz network: NO MCX, NO QFT,
    NO weight register, constant depth -> inside Q50 coherence.

Same observable as the big circuit:
    detector |+> ; apply detector-controlled e^{i phi(x)} ; H on detector
    Pr[d=0] = (1 + Re G)/2  (X-basis),  Pr[d=0] = (1 + Im G)/2  (Y-basis)
    G(lambda) = E[e^{i lambda (Sbar - K)}].

Run:
  python asian_trivial.py            # show circuit + validate + price + Q50
"""

import os
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

SHOTS = 4096
M = 2                                    # 2 path qubits -> genuinely Asian, 3 qubits total
OUTDIR = "."

S0, K, R, SIGMA, T = 4.20, 4.20, 0.03, 0.30, 1.0
LAM0 = 0.5


# ─────────────────────────── classical tree helpers ───────────────────────────
def crr_params():
    dt = T / M
    drift = (R - 0.5 * SIGMA ** 2) * dt
    vol = SIGMA * math.sqrt(dt)
    u = math.exp(drift + vol)
    d = math.exp(drift - vol)
    q = (math.exp(R * dt) - d) / (u - d)
    return u, d, q


def S_at(w, t):
    """Price at step t with w up-moves: S_t = S0 u^w d^(t-w)."""
    u, d, _ = crr_params()
    return S0 * (u ** w) * (d ** (t - w))


def sbar_of_path(bits):
    """Arithmetic average Sbar over the M steps for a path (list of bits)."""
    w, acc = 0, 0.0
    for t in range(1, M + 1):
        w += bits[t - 1]
        acc += S_at(w, t)
    return acc / M


def exact_asian_distribution():
    """Enumerate all 2^M paths -> (values of Sbar, their probabilities)."""
    _, _, q = crr_params()
    vals, probs = [], []
    for x in range(2 ** M):
        bits = [(x >> i) & 1 for i in range(M)]
        p = 1.0
        for b in bits:
            p *= q if b else (1 - q)
        vals.append(sbar_of_path(bits))
        probs.append(p)
    return np.array(vals), np.array(probs)


def classical_mc_price(n_samples, seed=0):
    u, d, q = crr_params()
    rng = np.random.default_rng(seed)
    up = rng.random((n_samples, M)) < q
    w = np.cumsum(up, axis=1)
    S = S0 * (u ** w) * (d ** (np.arange(1, M + 1) - w))
    payoff = np.maximum(S.mean(axis=1) - K, 0.0)
    disc = math.exp(-R * T)
    return disc * payoff.mean(), disc * payoff.std(ddof=1) / math.sqrt(n_samples)


# ─────────────────────────── trivial QNDM circuit (3 qubits) ───────────────────
def phase_diagonal_entries(lam):
    """8 complex entries of the 3-qubit diagonal on [path0, path1, detector].

    Detector-controlled QNDM phase: e^{i lambda (Sbar(x) - K)} on |1>_det,
    identity on |0>_det.  Entry index k (LSB = path0): bits q0,q1,det.
    """
    entries = np.ones(8, dtype=complex)
    for k in range(8):
        q0 = k & 1
        q1 = (k >> 1) & 1
        det = (k >> 2) & 1
        if det == 1:
            phi = lam * (sbar_of_path([q0, q1]) - K)
            entries[k] = np.exp(1j * phi)
    return list(entries)


def build_asian_trivial(lam, basis, measure=False):
    """3-qubit Asian QNDM characteristic-function circuit.

    Layout: q0,q1 = path qubits ; q2 = detector.
    Reads Pr[d=0] = (1 + Re G)/2 (basis 'X') or (1 + Im G)/2 (basis 'Y').
    """
    det = 2
    _, _, q = crr_params()
    qc = QuantumCircuit(3, name=f"asian_trivial_{basis}")

    theta = 2.0 * math.asin(math.sqrt(q))
    qc.ry(theta, 0)                      # exact, oracle-free loading (per step)
    qc.ry(theta, 1)
    qc.h(det)                            # detector |+>

    # detector-controlled e^{i lambda (Sbar - K)} : shallow 8-entry diagonal
    qc.append(DiagonalGate(phase_diagonal_entries(lam)), [0, 1, det])

    if basis == "Y":
        qc.sdg(det)
    qc.h(det)                            # X-basis read-out (Sdg+H = Y-basis)
    if measure:
        qc.measure_all()
    return qc, det


def char_G_from_statevector(lam):
    qcx, det = build_asian_trivial(lam, "X")
    qcy, _ = build_asian_trivial(lam, "Y")
    p0x = float(Statevector(qcx).probabilities([det])[0])
    p0y = float(Statevector(qcy).probabilities([det])[0])
    return complex(2 * p0x - 1, 2 * p0y - 1)


# ─────────────────────────── price from G via COS inversion ───────────────────
def asian_price_from_char(char_fn, n_cos=200):
    u_, d_, _ = crr_params()
    s_lo = sum(S0 * d_ ** t for t in range(1, M + 1)) / M
    s_hi = sum(S0 * u_ ** t for t in range(1, M + 1)) / M
    a, b = s_lo - 0.5, s_hi + 0.5

    ks = np.arange(n_cos)
    om = ks * math.pi / (b - a)
    phi = np.array([math.e ** (1j * o * K) * char_fn(o) for o in om])
    Fk = (2.0 / (b - a)) * np.real(phi * np.exp(-1j * om * a))
    Fk[0] *= 0.5

    xs = np.linspace(a, b, 4000)
    fX = (Fk[:, None] * np.cos(om[:, None] * (xs[None, :] - a))).sum(axis=0)
    price = math.exp(-R * T) * trapz(np.maximum(xs - K, 0.0) * fX, xs)
    return float(price), float(trapz(fX, xs))


# ─────────────────────────── reporting ────────────────────────────────────────
def p1_from_counts(counts, n_shots, det_index, n_qubits):
    """Pr[detector == 1] from measure_all counts (bitstring is qN-1 ... q0)."""
    p1 = 0
    for bs, c in counts.items():
        b = bs.replace(" ", "")
        bit = b[n_qubits - 1 - det_index]
        if bit == "1":
            p1 += c
    return p1 / n_shots


vals, probs = exact_asian_distribution()
price_exact = math.exp(-R * T) * float(np.sum(probs * np.maximum(vals - K, 0.0)))

print("=" * 66)
print("TRIVIAL arithmetic-Asian pricing  -  3-qubit QNDM (hardware-ready)")
print(f"{M} path qubits + 1 detector = 3 qubits   (no weight register)")
print("=" * 66)
print(f"S0={S0}  K={K}  r={R}  sigma={SIGMA}  T={T}   payoff f = mean_t S_t")
print(f"exact Asian price (enumeration)   = {price_exact:.6f}")
print(f"path averages Sbar = {np.round(vals, 5).tolist()}  "
      f"(01 vs 10 differ -> truly Asian)")


# ─────────────────────────── 0. SHOW THE CIRCUIT ──────────────────────────────
_qc, _det = build_asian_trivial(LAM0, "X", measure=True)
print(f"\ndetector index: {_det}")
print(f"\nAsian trivial circuit (X-basis, lambda={LAM0}):")
print(_qc.draw(output="text", fold=-1))

# transpiled depth (decomposed to basic gates) -- shows it is shallow
_dec = transpile(build_asian_trivial(LAM0, "X")[0],
                 basis_gates=["rz", "ry", "rx", "cz", "cx", "h", "sdg"],
                 optimization_level=3)
print(f"\nDecomposed to [rz,ry,rx,cz,cx,h]: depth={_dec.depth()}  "
      f"2q-gates={sum(_dec.count_ops().get(g, 0) for g in ('cz', 'cx'))}")

try:
    from qiskit.visualization import circuit_drawer
    circuit_drawer(_qc, output="mpl", fold=-1).savefig(
        f"{OUTDIR}/asian_trivial_circuit.png", dpi=150)
    print("Saved asian_trivial_circuit.png")
except Exception as e:
    print(f"[WARN] circuit PNG failed (non-fatal): {repr(e)}")


# ─────────────────────────── 1. SIMULATION ────────────────────────────────────
print("\n" + "=" * 66)
print("1a  Validate the quantum characteristic function G(lambda)")
print("=" * 66)
G_circ = char_G_from_statevector(LAM0)
G_ref = complex(np.sum(probs * np.exp(1j * LAM0 * (vals - K))))
print(f"G_circuit(lambda={LAM0}) = {G_circ.real:+.6f} {G_circ.imag:+.6f}j")
print(f"G_exact  (lambda={LAM0}) = {G_ref.real:+.6f} {G_ref.imag:+.6f}j   "
      f"|diff|={abs(G_circ - G_ref):.2e}")

print("\n" + "=" * 66)
print("1b  Price from the quantum G via Fourier (COS) inversion")
print("=" * 66)
price_cos, mass = asian_price_from_char(char_G_from_statevector, n_cos=200)
mc_price, mc_se = classical_mc_price(200_000, seed=0)
print(f"price from quantum G (COS)        = {price_cos:.6f}   "
      f"|err vs exact|={abs(price_cos - price_exact):.2e}  (density mass {mass:.4f})")
print(f"classical Monte Carlo (N=200000)  = {mc_price:.6f} +/- {mc_se:.6f}")
print(f"exact (enumeration)               = {price_exact:.6f}")


# ─────────────────────────── 2. REAL Q50 ──────────────────────────────────────
print("\n" + "=" * 66)
print("2  Real hardware on Q50  (shallow 3-qubit circuit)")
print("=" * 66)
hw = {}
try:
    url = os.getenv("Q50_CORTEX_URL")
    if not url:
        raise RuntimeError("Q50_CORTEX_URL not set "
                           "(run inside the event module on the reserved partition)")
    from iqm.qiskit_iqm import IQMProvider
    provider = IQMProvider(url, quantum_computer="q50")
    backend = provider.get_backend()
    for basis in ("X", "Y"):
        qc, det = build_asian_trivial(LAM0, basis, measure=True)
        tc = transpile(qc, backend, optimization_level=3)
        counts = backend.run(tc, shots=SHOTS).result().get_counts()
        p0 = 1.0 - p1_from_counts(counts, SHOTS, det, qc.num_qubits)
        hw[basis] = 2 * p0 - 1
        print(f"[Q50] {basis}-basis -> {'Re' if basis == 'X' else 'Im'} G "
              f"= {hw[basis]:+.4f}")
    print(f"[Q50] G(lambda0) = {hw['X']:+.4f} {hw['Y']:+.4f}j   (exact "
          f"{G_ref.real:+.4f} {G_ref.imag:+.4f}j)")
except Exception as e:
    print(f"[WARN] Q50 run skipped: {repr(e)}")
    print("       Continuing with simulation only.")


# ─────────────────────────── plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5))
names = ["exact\n(enum)", "quantum G\n(COS)", "classical\nMC"]
pv = [price_exact, price_cos, mc_price]
ax.bar(names, pv, color=["gray", "mediumpurple", "darkorange"])
ax.errorbar([2], [mc_price], yerr=[mc_se], fmt="none", ecolor="black", capsize=4)
ax.set_ylabel("Asian call price")
ax.set_ylim(0, max(pv) * 1.25)
for i, v in enumerate(pv):
    ax.text(i, v + max(pv) * 0.02, f"{v:.4f}", ha="center", fontsize=9)
ax.set_title("Trivial M=2 Asian price: exact vs quantum-via-G vs classical MC")
fig.tight_layout()
plt.savefig(f"{OUTDIR}/asian_trivial_comparison.png", dpi=150)

print("\n" + "=" * 66)
print(f"price: exact={price_exact:.5f}  quantumG-COS={price_cos:.5f}  MC={mc_price:.5f}")
print("Saved asian_trivial_circuit.png, asian_trivial_comparison.png")
print("Done.")
