"""
Retrieve the results of an already-executed IBM Runtime job (from its job id)
and compute G_hw correctly, avoiding the p1_from_counts bug that used
isa.num_qubits (= physical backend qubits, 156) instead of the number of bits
that were actually measured.

Credentials (read from environment variables):
    export IBM_TOKEN="<your IBM Quantum Platform API token>"
    export IBM_CRN="<your instance CRN>"

Usage:
    python fetch_ibm.py d8i9mflv8cos73f5ati0
    python fetch_ibm.py d8i9mflv8cos73f5ati0 --full   # if the job was M=3 (6 qubits)
"""
import os
import argparse

from qiskit_ibm_runtime import QiskitRuntimeService

# Same credentials as asian_ibm.py (fallback if there is no saved account).
IBM_TOKEN = os.getenv("IBM_TOKEN")
IBM_CRN = os.getenv("IBM_CRN")

LAM0 = 0.5
SHOTS = 8192


def p1_from_counts(counts, n_shots, det_index):
    """Pr[detector == 1]. The bitstring length = number of measured bits."""
    p1 = 0
    for bs, c in counts.items():
        b = bs.replace(" ", "")
        nbits = len(b)            # <-- FIX: use the measured bits, not physical qubits
        if b[nbits - 1 - det_index] == "1":
            p1 += c
    return p1 / n_shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--full", action="store_true",
                    help="the job was the M=3 circuit (det=5) instead of the trivial (det=2)")
    args = ap.parse_args()

    det = 5 if args.full else 2

    try:
        service = QiskitRuntimeService()
    except Exception:
        if not IBM_TOKEN or not IBM_CRN:
            raise RuntimeError(
                "No saved IBM account and IBM_TOKEN / IBM_CRN env vars are not "
                "set. Export them first:\n"
                "  export IBM_TOKEN=\"<your token>\"\n"
                "  export IBM_CRN=\"<your instance CRN>\"")
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=IBM_TOKEN, instance=IBM_CRN)

    job = service.job(args.job_id)
    print(f"job {args.job_id}: status = {job.status()}")
    res = job.result()

    cx = res[0].data.meas.get_counts()
    cy = res[1].data.meas.get_counts()

    p0x = 1.0 - p1_from_counts(cx, SHOTS, det)
    p0y = 1.0 - p1_from_counts(cy, SHOTS, det)
    G_hw = complex(2 * p0x - 1, 2 * p0y - 1)

    print(f"G_hw(l0={LAM0}) = {G_hw.real:+.6f} {G_hw.imag:+.6f}j")
    print(f"(noiseless expected = +0.876253 +0.037384j)")
    print(f"|G_hw - G_noiseless| = {abs(G_hw - complex(0.876253, 0.037384)):.4f}")

    # save the raw counts for safety
    out = f"counts_{args.job_id}.txt"
    with open(out, "w") as f:
        f.write(f"# job {args.job_id}  det={det}\n")
        f.write(f"# basis X counts:\n{cx}\n")
        f.write(f"# basis Y counts:\n{cy}\n")
        f.write(f"G_hw = {G_hw}\n")
    print(f"Raw counts saved to {out}")


if __name__ == "__main__":
    main()
