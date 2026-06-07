# 2 — IBM Quantum (`ibm_boston`, Heron r3)

Real-hardware runs of the Route-I (QNDM) Asian characteristic-function circuit on
**IBM Quantum**, using `SamplerV2` with **dynamical decoupling (XY4)** and
**gate/measurement twirling**. We compare the hardware `G(λ0)` against the
noiseless statevector reference and report `|dG| = |G_hw − G_noiseless|`.

## Credentials (environment variables — never hardcoded)

```bash
export IBM_TOKEN="<your IBM Quantum Platform API token>"
export IBM_CRN="<your instance CRN>"
# optional: export IBM_BACKEND="ibm_boston"
```

If the variables are missing (and no account is saved), the scripts stop with a
clear error message.

## Files

| File | What it does |
|---|---|
| `asian_ibm.py` | Builds + transpiles the circuits, submits both measurement bases (X, Y) in one job, computes `G_hw`, plots noiseless vs hardware, and prints the summary. Runs the trivial **M=2** by default; `--full` also runs **M=3**; `--no-hw` does noiseless only. |
| `fetch_ibm.py` | Retrieves an already-executed job by id and recomputes `G_hw` (with the fixed counts readout), saving the raw counts. |

## Run

```bash
python asian_ibm.py            # trivial M=2 (default)
python asian_ibm.py --full     # also the M=3 circuit
python asian_ibm.py --no-hw    # noiseless only
python fetch_ibm.py <job_id>          # M=2 job
python fetch_ibm.py <job_id> --full   # M=3 job
```

## Real results (`ibm_boston`)

| Circuit | `|dG| = |G_hw − G_noiseless|` | Verdict |
|---|---|---|
| **M=2 (trivial, 3 qubits)** | **0.0142** | ✅ excellent agreement |
| **M=3 (full, 6 qubits)** | **0.2242** | ⚠️ degraded by depth |

The shallow M=2 circuit reproduces the noiseless characteristic function to
within `0.0142`; the deeper M=3 circuit degrades as expected from the extra
two-qubit gate depth.

## `results/`

| File | Content |
|---|---|
| `asian_ibm_G_M2.png` | Noiseless vs hardware `G` bar chart (M=2). |
| `asian_ibm_G_M3.png` | Noiseless vs hardware `G` bar chart (M=3). |
| `ibm_run_M3.log` | Full console log of the M=3 hardware run. |
| `counts_d8i9mflv8cos73f5ati0.txt` | Raw X/Y counts of the real M=2 job. |

## Notes

`p1_from_counts` was fixed to derive the bitstring length from the measured bits
(`len(bitstring)`) instead of `isa.num_qubits` (the 156 physical qubits of the
backend), which previously caused an `IndexError`.
