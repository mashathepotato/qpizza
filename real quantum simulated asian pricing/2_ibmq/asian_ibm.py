"""
Arithmetic-Asian QNDM characteristic function on IBM hardware (ibm_boston, Heron r3)
====================================================================================
Goal: show that the REAL quantum run of the Route-I (QNDM) characteristic-function
circuit reaches G(lambda0) ~ equivalent to the NOISELESS simulation.

Priority: the TRIVIAL M=2 circuit (3 qubits, no weight register) -- the same
shallow Rz/Rzz diagonal that already validates exactly in simulation.
Optionally also the full M=3 circuit (6 qubits) to show the depth degradation.

We compute, per circuit and per measurement basis (X -> Re G, Y -> Im G):
    * NOISELESS   : exact Statevector marginal of the detector
    * HARDWARE    : ibm_boston via SamplerV2 (SLA: DD + measurement/gate twirling)
and compare G_hw vs G_noiseless, plus the COS-reconstructed Asian price.

Credentials (read from environment variables):
    export IBM_TOKEN="<your IBM Quantum Platform API token>"
    export IBM_CRN="<your instance CRN>"

Run:
    . .venv/bin/activate
    python asian_ibm.py                 # trivial only (default)
    python asian_ibm.py --full          # also the M=3 circuit
    python asian_ibm.py --no-hw         # noiseless only (skip the IBM run)
"""

import os
import math
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ─────────────────────────── IBM credentials (from env) ───────────────────────
IBM_TOKEN = os.getenv("IBM_TOKEN")
IBM_CRN = os.getenv("IBM_CRN")
BACKEND_NAME = os.getenv("IBM_BACKEND", "ibm_boston")

# ─────────────────────────── model parameters ─────────────────────────────────
S0, K, R, SIGMA, T = 4.20, 4.20, 0.03, 0.30, 1.0
LAM0 = 0.5
SHOTS = 8192


# ─────────────────────────── CRR tree helpers ─────────────────────────────────
def crr_params(M):
    dt = T / M
    drift = (R - 0.5 * SIGMA ** 2) * dt
    vol = SIGMA * math.sqrt(dt)
    u = math.exp(drift + vol)
    d = math.exp(drift - vol)
    q = (math.exp(R * dt) - d) / (u - d)
    return u, d, q


def S_at(M, w, t):
    u, d, _ = crr_params(M)
    return S0 * (u ** w) * (d ** (t - w))


def sbar_of_path(M, bits):
    w, acc = 0, 0.0
    for t in range(1, M + 1):
        w += bits[t - 1]
        acc += S_at(M, w, t)
    return acc / M


def exact_asian_distribution(M):
    _, _, q = crr_params(M)
    vals, probs = [], []
    for x in range(2 ** M):
        bits = [(x >> i) & 1 for i in range(M)]
        p = 1.0
        for b in bits:
            p *= q if b else (1 - q)
        vals.append(sbar_of_path(M, bits))
        probs.append(p)
    return np.array(vals), np.array(probs)


def classical_mc_price(M, n_samples, seed=0):
    u, d, q = crr_params(M)
    rng = np.random.default_rng(seed)
    up = rng.random((n_samples, M)) < q
    w = np.cumsum(up, axis=1)
    S = S0 * (u ** w) * (d ** (np.arange(1, M + 1) - w))
    payoff = np.maximum(S.mean(axis=1) - K, 0.0)
    disc = math.exp(-R * T)
    return disc * payoff.mean(), disc * payoff.std(ddof=1) / math.sqrt(n_samples)


# ─────────────────────────── circuits ─────────────────────────────────────────
def n_weight_qubits(M):
    return int(math.ceil(math.log2(M + 1)))


def controlled_increment(qc, ctrl, reg):
    for k in range(len(reg) - 1, 0, -1):
        qc.mcx([ctrl] + reg[:k], reg[k])
    qc.cx(ctrl, reg[0])


def controlled_decrement(qc, ctrl, reg):
    qc.cx(ctrl, reg[0])
    for k in range(1, len(reg)):
        qc.mcx([ctrl] + reg[:k], reg[k])


def build_trivial(lam, basis, measure=False):
    """M=2 Asian QNDM: 3 qubits (q0,q1 paths; q2 detector), shallow 8-entry diagonal."""
    M = 2
    det = 2
    _, _, q = crr_params(M)
    qc = QuantumCircuit(3, name=f"asian_trivial_{basis}")
    theta = 2.0 * math.asin(math.sqrt(q))
    qc.ry(theta, 0)
    qc.ry(theta, 1)
    qc.h(det)
    entries = np.ones(8, dtype=complex)
    for k in range(8):
        q0, q1, d = k & 1, (k >> 1) & 1, (k >> 2) & 1
        if d == 1:
            entries[k] = np.exp(1j * lam * (sbar_of_path(M, [q0, q1]) - K))
    qc.append(DiagonalGate(list(entries)), [0, 1, det])
    if basis == "Y":
        qc.sdg(det)
    qc.h(det)
    if measure:
        qc.measure_all()
    return qc, det, M


def build_full(lam, basis, measure=False):
    """M=3 Asian QNDM with running-weight register (6 qubits)."""
    M = 3
    n_w = n_weight_qubits(M)
    det = M + n_w
    wreg = [M + j for j in range(n_w)]
    _, _, q = crr_params(M)
    qc = QuantumCircuit(M + n_w + 1, name=f"asian_char_{basis}")
    for i in range(M):
        qc.ry(2.0 * math.asin(math.sqrt(q)), i)
    qc.h(det)
    for i in range(M):
        t = i + 1
        controlled_increment(qc, i, wreg)
        entries = np.ones(2 ** n_w, dtype=complex)
        for w in range(t + 1):
            entries[w] = np.exp(1j * lam * S_at(M, w, t) / M)
        qc.append(DiagonalGate(list(entries)).control(1), [det, *wreg])
    qc.p(-lam * K, det)
    for i in range(M):
        controlled_decrement(qc, i, wreg)
    if basis == "Y":
        qc.sdg(det)
    qc.h(det)
    if measure:
        qc.measure_all()
    return qc, det, M


# ─────────────────────────── noiseless G ──────────────────────────────────────
def G_noiseless(builder, lam):
    qcx, det, _ = builder(lam, "X")
    qcy, _, _ = builder(lam, "Y")
    p0x = float(Statevector(qcx).probabilities([det])[0])
    p0y = float(Statevector(qcy).probabilities([det])[0])
    return complex(2 * p0x - 1, 2 * p0y - 1)


# ─────────────────────────── COS price from G ─────────────────────────────────
def asian_price_from_char(M, char_fn, n_cos=200):
    u_, d_, _ = crr_params(M)
    s_lo = sum(S0 * d_ ** t for t in range(1, M + 1)) / M
    s_hi = sum(S0 * u_ ** t for t in range(1, M + 1)) / M
    a, b = s_lo - 0.5, s_hi + 0.5
    ks = np.arange(n_cos)
    om = ks * math.pi / (b - a)
    phi = np.array([np.exp(1j * o * K) * char_fn(o) for o in om])
    Fk = (2.0 / (b - a)) * np.real(phi * np.exp(-1j * om * a))
    Fk[0] *= 0.5
    xs = np.linspace(a, b, 4000)
    fX = (Fk[:, None] * np.cos(om[:, None] * (xs[None, :] - a))).sum(axis=0)
    price = math.exp(-R * T) * trapz(np.maximum(xs - K, 0.0) * fX, xs)
    return float(price), float(trapz(fX, xs))


# ─────────────────────────── hardware readout helper ──────────────────────────
def p1_from_counts(counts, n_shots, det_index, n_qubits=None):
    """Pr[detector == 1] from a counts dict; Qiskit bitstring is q_{n-1}...q_0.

    BUGFIX: the bitstring length equals the number of MEASURED bits (measure_all
    on the original logical qubits), NOT the number of physical qubits of the
    backend, so we derive it from the bitstring itself (len(b)). This avoids the
    IndexError seen when using isa.num_qubits (= 156 physical qubits).
    """
    p1 = 0
    for bs, c in counts.items():
        b = bs.replace(" ", "")
        nbits = len(b)
        if b[nbits - 1 - det_index] == "1":
            p1 += c
    return p1 / n_shots


# ─────────────────────────── main ─────────────────────────────────────────────
def run_circuit_family(name, builder, do_hw, service, backend, pm, sampler):
    M = builder(LAM0, "X")[2]
    vals, probs = exact_asian_distribution(M)
    price_exact = math.exp(-R * T) * float(np.sum(probs * np.maximum(vals - K, 0.0)))

    print("\n" + "=" * 70)
    print(f"  {name}  (M={M})")
    print("=" * 70)
    print(f"exact Asian price (enumeration) = {price_exact:.6f}")

    # noiseless reference
    G_nl = G_noiseless(builder, LAM0)
    G_ref = complex(np.sum(probs * np.exp(1j * LAM0 * (vals - K))))
    print(f"G_noiseless(l0={LAM0}) = {G_nl.real:+.6f} {G_nl.imag:+.6f}j  "
          f"(analytic {G_ref.real:+.6f} {G_ref.imag:+.6f}j)")

    price_nl, _ = asian_price_from_char(M, lambda l: G_noiseless(builder, l))
    print(f"noiseless price (COS)  = {price_nl:.6f}")

    G_hw = None
    if do_hw:
        from qiskit_ibm_runtime import SamplerV2
        # build, transpile to ISA, submit both bases in ONE job
        qcx, detx, _ = builder(LAM0, "X", measure=True)
        qcy, dety, _ = builder(LAM0, "Y", measure=True)
        isa = pm.run([qcx, qcy])
        print(f"[{BACKEND_NAME}] transpiled depths: "
              f"X={isa[0].depth()} Y={isa[1].depth()}  -> submitting (shots={SHOTS})")
        job = sampler.run(isa, shots=SHOTS)
        print(f"[{BACKEND_NAME}] job id = {job.job_id()}  (waiting for result...)")
        res = job.result()
        cx = res[0].data.meas.get_counts()
        cy = res[1].data.meas.get_counts()
        nq = isa[0].num_qubits
        p0x = 1.0 - p1_from_counts(cx, SHOTS, detx, nq)
        p0y = 1.0 - p1_from_counts(cy, SHOTS, dety, nq)
        G_hw = complex(2 * p0x - 1, 2 * p0y - 1)
        print(f"[{BACKEND_NAME}] G_hw = {G_hw.real:+.6f} {G_hw.imag:+.6f}j")
        print(f"[{BACKEND_NAME}] |G_hw - G_noiseless| = {abs(G_hw - G_nl):.4f}")

    return dict(name=name, M=M, price_exact=price_exact, G_nl=G_nl,
                price_nl=price_nl, G_hw=G_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run the M=3 (6-qubit) circuit")
    ap.add_argument("--no-hw", action="store_true",
                    help="noiseless only, skip the IBM hardware run")
    args = ap.parse_args()
    do_hw = not args.no_hw

    service = backend = pm = sampler = None
    if do_hw:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager)
            try:
                service = QiskitRuntimeService()
            except Exception:
                if not IBM_TOKEN or not IBM_CRN:
                    raise RuntimeError(
                        "No saved IBM account and IBM_TOKEN / IBM_CRN env vars "
                        "are not set. Export them first:\n"
                        "  export IBM_TOKEN=\"<your token>\"\n"
                        "  export IBM_CRN=\"<your instance CRN>\"")
                service = QiskitRuntimeService(
                    channel="ibm_quantum_platform",
                    token=IBM_TOKEN, instance=IBM_CRN)
            backend = service.backend(BACKEND_NAME)
            print(f"Connected to {backend.name}  "
                  f"({backend.num_qubits} qubits)")
            pm = generate_preset_pass_manager(
                target=backend.target, optimization_level=3)
            sampler = SamplerV2(mode=backend)
            # error mitigation: dynamical decoupling + twirling
            opt = sampler.options
            opt.dynamical_decoupling.enable = True
            opt.dynamical_decoupling.sequence_type = "XY4"
            opt.twirling.enable_gates = True
            opt.twirling.enable_measure = True
        except Exception as e:
            print(f"[WARN] IBM hardware unavailable ({repr(e)}). "
                  f"Continuing noiseless only.")
            do_hw = False

    results = [run_circuit_family("TRIVIAL Asian QNDM (3 qubits)",
                                  build_trivial, do_hw, service, backend, pm, sampler)]
    if args.full:
        results.append(run_circuit_family("FULL Asian QNDM (6 qubits)",
                                          build_full, do_hw, service, backend, pm, sampler))

    # ── plot G comparison ──
    for r in results:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        labels = ["Re G", "Im G"]
        x = np.arange(2)
        w = 0.35
        nl = [r["G_nl"].real, r["G_nl"].imag]
        ax.bar(x - w / 2, nl, w, label="noiseless", color="mediumpurple")
        if r["G_hw"] is not None:
            hw = [r["G_hw"].real, r["G_hw"].imag]
            ax.bar(x + w / 2, hw, w, label=f"{BACKEND_NAME}", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(f"G(lambda0={LAM0})")
        ax.set_title(f"{r['name']}: noiseless vs hardware")
        ax.legend()
        fig.tight_layout()
        fn = f"asian_ibm_G_M{r['M']}.png"
        plt.savefig(fn, dpi=150)
        print(f"Saved {fn}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        line = (f"{r['name']:32s}  exact={r['price_exact']:.5f}  "
                f"G_nl={r['G_nl'].real:+.4f}{r['G_nl'].imag:+.4f}j")
        if r["G_hw"] is not None:
            line += (f"  G_hw={r['G_hw'].real:+.4f}{r['G_hw'].imag:+.4f}j"
                     f"  |dG|={abs(r['G_hw'] - r['G_nl']):.4f}")
        print(line)
    print("Done.")


if __name__ == "__main__":
    main()
