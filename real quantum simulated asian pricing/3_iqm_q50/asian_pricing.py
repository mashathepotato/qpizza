"""
Arithmetic Asian option pricing on a CRR tree  -  quantum characteristic-function
route, with an honest query-complexity comparison against classical Monte Carlo
================================================================================
This is the PATH-DEPENDENT (arithmetic Asian) sibling of the European circuits.
Payoff functional:  f = Sbar = (1/M) sum_{t=1}^M S_t .

Why the European circuit does NOT work here.  On a recombining tree
S_t = S0 u^{w_t} d^{t-w_t} depends only on the *running* Hamming weight w_t.
The European payoff depends only on the *final* weight w_M, so one weight register
and ONE diagonal suffice.  The Asian average depends on the WHOLE trajectory
(w_1,...,w_M): two paths with the same w_M (e.g. up-down vs down-up) have different
averages.  So we need the running-weight construction of the paper (Eq. U_t with
phi_t = S_t / M):

    detector |+> ; weight |0>
    for t = 1..M:
        increment weight by x_t            # register now holds w_t
        detector-controlled diagonal phase  e^{i lam S_t(w_t)/M}   (only t+1 entries)
    apply e^{-i lam K} on |1>_detector      # absorb the strike
    uncompute the weight register -> |0>    # poly(M): M decrements
    rotate+measure detector

The detector then carries  e^{i lam (Sbar - K)}  coherently over all paths, so
    Pr[d=0] = (1 + Re G(lam))/2,   G(lam) = E[e^{i lam (Sbar - K)}]   (Y-basis -> Im G),
the quantum characteristic function of the Asian average.  Loading is the exact,
oracle-free  M x R_Y  of Sec. "Exact loading"; the oracle is O(log M) qubits and
poly(M) depth -- no 2^M diagonal, no register storing Sbar.

From G(lam) the price follows by Fourier inversion (COS method), validated here
against exact enumeration and against classical Monte Carlo.

THE COMPETITIVE POINT (honest).  We do NOT beat classical wall-clock at this size,
and the deep amplification is fault-tolerant.  What we show is the *query-complexity*
advantage that underlies the method: estimating the payoff-relevant amplitude to
error eps costs O(1/eps) quantum oracle calls (amplitude estimation) versus the
O(1/eps^2) samples of classical MC.  We demonstrate this slope difference in
simulation (Sec. 1c) -- exactly the 1/eps convergence study the paper flags.

Self-contained (no qpizza imports), toy CRR params, run like the others:
  0. SHOW     - print + save the Asian characteristic-function circuit.
  1. SIMULATE - (a) validate G(lam); (b) price via COS vs exact & MC;
                (c) MLAE vs classical sampling: error-vs-queries.
  2. REAL Q50 - the shallow pieces on the IQM Q50, then compare.
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

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 2.x / 1.x

SHOTS  = 1024
M      = 3                       # path qubits (kept small & legible)
OUTDIR = "."

S0, K, R, SIGMA, T = 4.20, 4.20, 0.03, 0.30, 1.0

LAM0 = 0.5                       # representative frequency for the scaling study
MLAE_MAXPOWERS = [1, 2, 4, 8, 16]
SHOTS_PER = 100                  # shots per Grover power in the MLAE schedule


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


def n_weight_qubits():
    return int(math.ceil(math.log2(M + 1)))


def exact_asian_distribution():
    """Enumerate all 2^M paths -> (values of Sbar, their probabilities)."""
    _, _, q = crr_params()
    vals, probs = [], []
    for x in range(2 ** M):
        bits = [(x >> i) & 1 for i in range(M)]
        p = 1.0
        for b in bits:
            p *= q if b else (1 - q)
        sbar, w = 0.0, 0
        for t in range(1, M + 1):
            w += bits[t - 1]
            sbar += S_at(w, t)
        vals.append(sbar / M)
        probs.append(p)
    return np.array(vals), np.array(probs)


def classical_mc_price(n_samples, seed=0):
    """Classical Monte Carlo Asian price (the baseline). Error ~ 1/sqrt(N)."""
    u, d, q = crr_params()
    rng = np.random.default_rng(seed)
    up = rng.random((n_samples, M)) < q
    w = np.cumsum(up, axis=1)
    S = S0 * (u ** w) * (d ** (np.arange(1, M + 1) - w))
    payoff = np.maximum(S.mean(axis=1) - K, 0.0)
    disc = math.exp(-R * T)
    return disc * payoff.mean(), disc * payoff.std(ddof=1) / math.sqrt(n_samples)


# ─────────────────────────── quantum building blocks ──────────────────────────
def controlled_increment(qc, ctrl, reg):
    n = len(reg)
    for k in range(n - 1, 0, -1):
        qc.mcx([ctrl] + reg[:k], reg[k])
    qc.cx(ctrl, reg[0])


def controlled_decrement(qc, ctrl, reg):
    """Exact inverse of controlled_increment (gates in reverse order)."""
    qc.cx(ctrl, reg[0])
    for k in range(1, len(reg)):
        qc.mcx([ctrl] + reg[:k], reg[k])


def build_asian_char(lam, basis, measure=False):
    """Quantum characteristic function of the arithmetic Asian average.

    Layout: q0..q_{M-1}=paths, next n_w = running-weight register, last = detector.
    Reads Pr[d=0]=(1+Re G)/2 (basis 'X') or (1+Im G)/2 (basis 'Y'),
    G(lam)=E[e^{i lam (Sbar-K)}].
    """
    n_w = n_weight_qubits()
    det = M + n_w
    wreg = [M + j for j in range(n_w)]
    _, _, q = crr_params()
    qc = QuantumCircuit(M + n_w + 1, name=f"asian_char_{basis}")

    for i in range(M):                                   # exact, oracle-free loading
        qc.ry(2.0 * math.asin(math.sqrt(q)), i)
    qc.h(det)                                            # detector |+>

    for i in range(M):                                   # running-weight oracle
        t = i + 1                                        # path qubit i -> step t
        controlled_increment(qc, i, wreg)                # register now holds w_t
        entries = np.ones(2 ** n_w, dtype=complex)
        for w in range(t + 1):                           # only t+1 reachable weights
            entries[w] = np.exp(1j * lam * S_at(w, t) / M)   # add S_t/M to the phase
        qc.append(DiagonalGate(list(entries)).control(1), [det, *wreg])
    qc.p(-lam * K, det)                                  # absorb strike: e^{-i lam K} on |1>_d
    for i in range(M):                                   # uncompute running weight -> |0>
        controlled_decrement(qc, i, wreg)

    if basis == "Y":
        qc.sdg(det)
    qc.h(det)                                            # X-basis read-out (with Sdg = Y)
    if measure:
        qc.measure_all()
    return qc, det


def char_G_from_statevector(lam):
    """Exact (simulated) G(lam) from the detector marginal of the two circuits."""
    qcx, det = build_asian_char(lam, "X")
    qcy, _ = build_asian_char(lam, "Y")
    p0x = float(Statevector(qcx).probabilities([det])[0])
    p0y = float(Statevector(qcy).probabilities([det])[0])
    return complex(2 * p0x - 1, 2 * p0y - 1)


# ─────────────────────────── price from G via COS inversion ───────────────────
def asian_price_from_char(char_fn, n_cos=200):
    """COS method (Fang-Oosterlee). char_fn(lam)->G(lam)=E[e^{i lam (Sbar-K)}].

    phi_X(u)=E[e^{i u Sbar}] = e^{i u K} G(u); reconstruct the density of Sbar on
    a truncation [a,b] from the tree bounds, then integrate the call payoff."""
    u_, d_, _ = crr_params()
    s_lo = sum(S0 * d_ ** t for t in range(1, M + 1)) / M    # all-down average
    s_hi = sum(S0 * u_ ** t for t in range(1, M + 1)) / M    # all-up average
    a, b = s_lo - 0.5, s_hi + 0.5

    ks = np.arange(n_cos)
    om = ks * math.pi / (b - a)
    phi = np.array([math.e ** (1j * o * K) * char_fn(o) for o in om])    # phi_X(u)
    Fk = (2.0 / (b - a)) * np.real(phi * np.exp(-1j * om * a))
    Fk[0] *= 0.5

    xs = np.linspace(a, b, 4000)
    fX = (Fk[:, None] * np.cos(om[:, None] * (xs[None, :] - a))).sum(axis=0)
    price = math.exp(-R * T) * trapz(np.maximum(xs - K, 0.0) * fX, xs)
    return float(price), float(trapz(fX, xs))


# ─────────────────────────── Grover + MLAE (query scaling) ─────────────────────
def reflect_about_zero(qc, qubits):
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)


def grover_iterate(lam):
    """Q = A S0 A^dag S_chi for the Asian char circuit, good = (detector == 1)."""
    A, det = build_asian_char(lam, "X")
    n = A.num_qubits
    q = QuantumCircuit(n, name="Q")
    q.z(det)                                  # S_chi: flip sign of good (d=1)
    q.compose(A.inverse(), inplace=True)
    reflect_about_zero(q, list(range(n)))
    q.compose(A, inplace=True)
    return q.to_gate(), det, A, n


def amplified_p1_exact(lam, m):
    Q, det, A, n = grover_iterate(lam)
    qc = QuantumCircuit(n)
    qc.compose(A, inplace=True)
    for _ in range(m):
        qc.append(Q, range(n))
    return float(Statevector(qc).probabilities([det])[1])


def mlae(powers, hits, shots):
    """Maximum-likelihood amplitude estimation (Suzuki 2020), grid MLE."""
    thetas = np.linspace(1e-6, math.pi / 2 - 1e-6, 100001)
    ll = np.zeros_like(thetas)
    for m, h, N in zip(powers, hits, shots):
        p = np.clip(np.sin((2 * m + 1) * thetas) ** 2, 1e-12, 1 - 1e-12)
        ll += h * np.log(p) + (N - h) * np.log(1 - p)
    return float(np.sin(thetas[int(np.argmax(ll))]) ** 2)


# ─────────────────────────── reporting ────────────────────────────────────────
def p1_from_counts(counts, n_shots):
    p1 = 0
    for bs, c in counts.items():
        if bs.replace(" ", "")[0] == "1":
            p1 += c
    return p1 / n_shots


vals, probs = exact_asian_distribution()
price_exact = math.exp(-R * T) * float(np.sum(probs * np.maximum(vals - K, 0.0)))
n_w = n_weight_qubits()

print("=" * 66)
print(f"Arithmetic Asian pricing  -  quantum characteristic-function route")
print(f"{M} path qubits + {n_w} running-weight qubits + 1 detector "
      f"= {M + n_w + 1} qubits")
print("=" * 66)
print(f"S0={S0}  K={K}  r={R}  sigma={SIGMA}  T={T}   payoff f = mean_t S_t")
print(f"exact Asian price (enumeration)   = {price_exact:.6f}")


# ─────────────────────────── 0. SHOW THE CIRCUIT ──────────────────────────────
_qc, _det = build_asian_char(LAM0, "X", measure=True)
print(f"\ndetector index: {_det}")
print(f"\nAsian characteristic-function circuit (X-basis, lambda={LAM0}):")
print(_qc.draw(output="text", fold=-1))
try:
    from qiskit.visualization import circuit_drawer
    circuit_drawer(_qc, output="mpl", fold=-1).savefig(f"{OUTDIR}/asian_char_circuit.png", dpi=150)
    print("Saved asian_char_circuit.png")
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

print("\n" + "=" * 66)
print("1c  Competitive claim: amplitude estimation vs classical sampling")
print("=" * 66)
print("Estimate a = Pr[d=1] = (1-Re G)/2 at lambda0; same quantity, two ways.")
a_true = (1.0 - G_circ.real) / 2.0
theta_a = math.asin(math.sqrt(a_true))
print(f"target a = {a_true:.6f}\n")
print(f"{'queries':>9} {'classical err':>14} {'quantum (MLAE) err':>20}")
rng = np.random.default_rng(0)
scaling = []
for maxm in MLAE_MAXPOWERS:
    powers = sorted({0, *[2 ** k for k in range(int(math.log2(maxm)) + 1)]})
    powers = [m for m in powers if m <= maxm]
    p_ex = {m: amplified_p1_exact(LAM0, m) for m in powers}
    queries = sum(SHOTS_PER * (2 * m + 1) for m in powers)
    eq, ec = [], []
    for _ in range(60):
        hits = [int(rng.binomial(SHOTS_PER, p_ex[m])) for m in powers]
        eq.append(abs(mlae(powers, hits, [SHOTS_PER] * len(powers)) - a_true))
        hc = int(rng.binomial(queries, a_true))           # same circuit-use budget
        ec.append(abs(hc / queries - a_true))
    scaling.append((queries, float(np.mean(ec)), float(np.mean(eq))))
    print(f"{queries:>9} {np.mean(ec):>14.5f} {np.mean(eq):>20.5f}")


# ─────────────────────────── 2. REAL Q50 ──────────────────────────────────────
print("\n" + "=" * 66)
print("2  Real hardware on Q50  (shallow pieces only)")
print("=" * 66)
print("NISQ note: the bare characteristic-function circuit and m=1 are runnable;")
print("the deep Grover amplification for the full 1/eps speed-up is FT-era.")
hw = {}
hw_ok = False
try:
    url = os.getenv("Q50_CORTEX_URL")
    if not url:
        raise RuntimeError("Q50_CORTEX_URL not set "
                           "(run inside the event module on the reserved partition)")
    from iqm.qiskit_iqm import IQMProvider
    provider = IQMProvider(url, quantum_computer="q50")
    backend = provider.get_backend()
    for basis in ("X", "Y"):
        qc, det = build_asian_char(LAM0, basis, measure=True)
        tc = transpile(qc, backend)
        counts = backend.run(tc, shots=SHOTS).result().get_counts()
        p0 = 1.0 - p1_from_counts(counts, SHOTS)
        hw[basis] = 2 * p0 - 1
        print(f"[Q50] {basis}-basis -> {'Re' if basis == 'X' else 'Im'} G = {hw[basis]:+.4f}")
    print(f"[Q50] G(lambda0) = {hw['X']:+.4f} {hw['Y']:+.4f}j   (exact "
          f"{G_ref.real:+.4f} {G_ref.imag:+.4f}j)")
    hw_ok = True
except Exception as e:
    print(f"[WARN] Q50 run skipped: {repr(e)}")
    print("       Continuing with simulation only.")


# ─────────────────────────── plots ────────────────────────────────────────────
qs = np.array([s[0] for s in scaling], float)
ec = np.array([s[1] for s in scaling], float)
eq = np.array([s[2] for s in scaling], float)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.loglog(qs, ec, "o-", color="darkorange", label="classical sampling  ~ 1/sqrt(N)")
ax.loglog(qs, eq, "s-", color="mediumpurple", label="quantum MLAE  ~ 1/N")
ax.loglog(qs, ec[0] * (qs / qs[0]) ** -0.5, "--", color="darkorange", lw=0.8, alpha=0.7)
ax.loglog(qs, eq[0] * (qs / qs[0]) ** -1.0, "--", color="mediumpurple", lw=0.8, alpha=0.7)
ax.set_xlabel("circuit uses / oracle queries")
ax.set_ylabel("estimation error of a = Pr[d=1]")
ax.set_title("Asian option: amplitude estimation vs classical sampling\n"
             "exact oracle-free loading + running-weight HW oracle (M=%d)" % M)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
plt.savefig(f"{OUTDIR}/asian_query_scaling.png", dpi=150)

# price comparison bar
fig2, ax2 = plt.subplots(figsize=(6.5, 5))
names = ["exact\n(enum)", "quantum G\n(COS)", "classical\nMC"]
pv = [price_exact, price_cos, mc_price]
ax2.bar(names, pv, color=["gray", "mediumpurple", "darkorange"])
ax2.errorbar([2], [mc_price], yerr=[mc_se], fmt="none", ecolor="black", capsize=4)
ax2.set_ylabel("Asian call price")
ax2.set_ylim(0, max(pv) * 1.25)
for i, v in enumerate(pv):
    ax2.text(i, v + max(pv) * 0.02, f"{v:.4f}", ha="center", fontsize=9)
ax2.set_title("Arithmetic Asian price: exact vs quantum-via-G vs classical MC")
fig2.tight_layout()
plt.savefig(f"{OUTDIR}/asian_price_comparison.png", dpi=150)

print("\n" + "=" * 66)
print(f"price: exact={price_exact:.5f}  quantumG-COS={price_cos:.5f}  MC={mc_price:.5f}")
print("Saved asian_query_scaling.png, asian_price_comparison.png")
print("Done.")
