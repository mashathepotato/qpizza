# Running quantum programs on Q50 via LUMI

Reference for an AI coding agent (and humans) working on a quantum-computing
project that targets **Q50**, a superconducting quantum computer accessed through
the **LUMI** supercomputer, while connected to LUMI over **SSH** (e.g. VS Code /
VSCodium Remote-SSH).

Q50 is programmed with **Qiskit** + the **IQM** provider (`iqm.qiskit_iqm`).
Circuits are submitted to the FiQCI "cortex" server; LUMI provides the classical
environment, the software module, and the Slurm allocation / accounting.

---

## 0. Fill in your project values

These are project/site specific. Replace them everywhere below.

| Variable          | Example (Junction Quantum Hack) | Notes |
|-------------------|----------------------------------|-------|
| `<PROJECT>`       | `project_465003017`              | your LUMI project / `--account` |
| `<PARTITION>`     | `debug`                          | `debug` = quick, capped runtime; see LUMI partitions for others |
| `<RESERVATION>`   | `JQH2026`                        | optional; **event-only**. Omit outside the event. |
| `<SCRATCH>`       | `/scratch/<PROJECT>/$USER`       | where your code/output live |
| Modules           | `Local-quantum`, `fiqci-vtt-qiskit-JQH` | the qiskit module name may differ per project |
| Device status     | https://fiqci.fi/status          | check before blaming your code |

> The `<RESERVATION>` gives priority QPU access during an event and usually
> implies a runtime cap (e.g. `debug` = 30 min/session). Outside an event, drop
> the `--reservation` line and pick a normal partition; jobs may queue longer.

---

## 1. Where SSH puts you

`ssh lumi` (or Remote-SSH) lands you on a **login node**, e.g. `uan02`
(`<user>@uan02`). Key consequences:

- **Login nodes HAVE internet** → do all `pip install` here.
- **Compute nodes do NOT have internet** → never `pip install` inside a batch job.
- Don't run heavy/long work on the login node itself. Quantum/CPU work goes
  through **Slurm** (`sbatch`, `srun`). Tiny smoke tests can run directly on the
  login node after loading the modules.

---

## 2. One-time setup (run on the LOGIN node)

```bash
module load Local-quantum
module load fiqci-vtt-qiskit-JQH         # the qiskit + Q50 module

# extra Python packages your project needs, into your user space.
# MUST be done on the login node (compute nodes have no internet).
pip install --user matplotlib scikit-learn   # add whatever you need
```

Notes:
- The module already provides Qiskit, NumPy, SciPy and the IQM tooling. Only
  install what is genuinely missing.
- Installing many packages strains the shared filesystem; keep it minimal. For
  larger/reproducible stacks use the LUMI container wrapper instead.
- After a notebook-cell `pip install`, restart the kernel. In a script this is
  irrelevant (fresh process).

---

## 3. The Q50 backend in Python (canonical pattern)

The module exports `Q50_CORTEX_URL` as an environment variable. Use it; do not
hardcode URLs or credentials.

```python
import os
from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm import IQMProvider

url = os.getenv("Q50_CORTEX_URL")          # set by the module
provider = IQMProvider(url, quantum_computer="q50")
backend = provider.get_backend()

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

tqc = transpile(qc, backend)               # ALWAYS transpile to the device first
job = backend.run(tqc, shots=1024)
counts = job.result().get_counts()
print(counts)                              # e.g. {'00': 491, '11': 467, '01': 18, '10': 24}
```

The spurious `01`/`10` counts above are real hardware readout/decoherence noise —
expect them; a noiseless simulator would not produce them.

---

## 4. HARD CONSTRAINTS & GOTCHAS (read before writing run code)

These are device/site limits learned the hard way. Violating them produces
confusing errors, not clear ones.

### 4.1 Max 100 circuits per `backend.run()`
The device rejects larger submissions:
`{"error":"Batch contains N circuits, exceeds maximum 100 for device Q50."}`
If you have more than 100 circuits, **submit in chunks of ≤ 100**, or run **one
circuit per job** (most robust, see 4.2).

### 4.2 Prefer one circuit per job for robustness
Batched (multi-circuit) submissions can fail at result retrieval with an IQM
`BadRequestError` from `get_job_artifact_measurements`. The single-circuit path
(identical to a basic Bell-pair run) is the most reliable:

```python
def run_circuits(circuits, backend, shots=1024):
    """Robust: one circuit per job. Returns a list of counts dicts."""
    out = []
    for i, qc in enumerate(circuits):
        tqc = transpile(qc, backend)
        out.append(backend.run(tqc, shots=shots).result().get_counts())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(circuits)} done")
    return out
```

If you do batch (for speed), chunk it:

```python
def run_batched(circuits, backend, shots=1024, max_batch=100):
    tcircs = transpile(circuits, backend)        # list in -> list out
    counts = []
    for start in range(0, len(tcircs), max_batch):
        chunk = tcircs[start:start + max_batch]   # <= 100 circuits per run
        res = backend.run(chunk, shots=shots).result()
        counts += [res.get_counts(j) for j in range(len(chunk))]
    return counts
```

Sequential single-circuit jobs are slower (per-job overhead). Keep the circuit
**count** modest so a session finishes inside the partition's time cap.

### 4.3 IQM exceptions can have non-string messages
Some IQM errors carry a `list` message and crash naive logging
(`TypeError: __str__ returned non-string`). Always log with `repr`:

```python
try:
    counts = backend.run(tqc, shots=1024).result().get_counts()
except Exception as e:
    print(f"[WARN] Q50 run failed: {repr(e)}")   # repr, not str/f-string of e
```
Wrap the whole hardware section in `try/except` so a device outage degrades
gracefully instead of killing the job.

### 4.4 No GUI in batch jobs
Matplotlib has no display on compute nodes. Set the headless backend and SAVE,
never `show`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ...
plt.savefig("out.png", dpi=150)     # NOT plt.show()
```

### 4.5 Always `transpile` to the backend
`transpile(circuit, backend)` maps your logical circuit onto Q50's native gate
set and connectivity. Measurement→classical-bit mapping is preserved, so
`get_counts()` bitstrings still correspond to your logical qubits.

### 4.6 Save to scratch
Run from and write outputs to `<SCRATCH>`. Session/temp space can be wiped when a
job ends.

---

## 5. Running via Slurm (the reliable path)

Save your program (e.g. `program.py`) in `<SCRATCH>`, then create `run.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=q50-job
#SBATCH --account=<PROJECT>
#SBATCH --reservation=<RESERVATION>     # OMIT this line outside an event
#SBATCH --partition=<PARTITION>
#SBATCH --time=00:25:00                 # keep under the partition cap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --output=q50-%j.out

module load Local-quantum
module load fiqci-vtt-qiskit-JQH
# Do NOT pip install here (compute node = no internet). Install on login node first.

python program.py
```

Submit and monitor:

```bash
sbatch run.sh            # submit; prints a <jobid>
squeue --me              # PD = pending/queued, R = running
cat q50-<jobid>.out      # read stdout once finished
scancel <jobid>          # cancel if needed
```

`--account`, `--partition` (and `--reservation` if used) must match valid values
or the job won't get the intended access.

### Quick interactive test
For a fast check without writing a batch script, after loading the modules on a
login node you can run a tiny script directly (`python smoke_test.py`) with a
single small circuit. For anything non-trivial, use `sbatch`.

---

## 6. Error mitigation (optional): FiQCI-EMS

The FiQCI Error Mitigation Service wraps the IQM backend and improves accuracy.
It is included in the qiskit module. Minimal change: wrap the backend and pass a
mitigation level.

```python
import os
from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm import IQMProvider
from fiqci.ems import FiQCIBackend

provider = IQMProvider(os.getenv("Q50_CORTEX_URL"), quantum_computer="q50")
backend = provider.get_backend()
ems_backend = FiQCIBackend(backend, mitigation_level=1)   # try level 1 first

qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure_all()
tqc = transpile(qc, ems_backend)
result = ems_backend.run(tqc, shots=1024).result()
```

EMS supports readout error mitigation (REM), Pauli twirling, dynamic decoupling
and zero-noise extrapolation (ZNE), and can compute expectation values. You can
pin a calibration file when constructing the EMS backend to cache REM data and
skip re-running calibration circuits each submission. Use EMS when raw `<Z>`-type
expectation values come back suspiciously flat (squashed toward 0) due to noise.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `exceeds maximum 100 for device Q50` | >100 circuits in one `run`. Chunk to ≤100 (4.1) or one-per-job (4.2). |
| `BadRequestError` at `.result()` on a batch | Fragile batched fetch. Use one circuit per job (4.2). |
| `TypeError: __str__ returned non-string` | Logging an IQM exception with `str`. Use `repr(e)` (4.3). |
| `ModuleNotFoundError: No module named 'qiskit'` | Modules not loaded. Ensure both `module load Local-quantum` and `module load fiqci-vtt-qiskit-JQH` ran; if cached oddly, `module --ignore_cache load fiqci-vtt-qiskit-JQH`. |
| `ModuleNotFoundError` for matplotlib/sklearn/etc. | Not installed, or installed on the wrong node. `pip install --user <pkg>` on the **login** node, then resubmit. |
| `backend.run()` hangs or fails | Q50 may be offline / calibrating. Check https://fiqci.fi/status. |
| Job stuck in `PD` (queued) | No reservation/priority, or asking for too much. Use the reservation if you have one; request fewer cores/less memory. |
| Lost work when session ended | Save often and write to `<SCRATCH>`, not temp/session space. |
| No plot / blank output in batch | Used `plt.show()`. Switch to `Agg` + `plt.savefig` (4.4). |

---

## 8. Quick reference

- Provider: `from iqm.qiskit_iqm import IQMProvider` → `IQMProvider(os.getenv("Q50_CORTEX_URL"), quantum_computer="q50").get_backend()`
- Run: `transpile(circuit, backend)` → `backend.run(tqc, shots=N).result().get_counts()`
- **Limit: ≤ 100 circuits per `run`.** Prefer one circuit per job.
- Env var (set by module): `Q50_CORTEX_URL`
- Modules: `Local-quantum`, `fiqci-vtt-qiskit-JQH`
- `pip install --user` only on the **login node**.
- Batch: headless matplotlib (`Agg` + `savefig`), write to `<SCRATCH>`, wrap hardware in `try/except` with `repr(e)`.
- Device status: https://fiqci.fi/status