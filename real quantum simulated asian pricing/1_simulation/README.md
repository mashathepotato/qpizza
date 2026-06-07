# 1 — Noiseless Simulation

Self-contained, **noiseless** validation of the Route-I (QNDM) Asian pricing
pipeline. **No quantum account or hardware is required** — everything runs on a
local statevector simulator.

## Files

| File | What it does |
|---|---|
| `asian_pricing.py` | Full **M=3** pipeline: builds the QNDM characteristic-function circuit, validates `G_circuit` against the exact enumeration, reconstructs the price with the COS method (vs exact and Monte Carlo), and shows the MLAE `1/ε` scaling. |
| `asian_trivial.py` | The shallow **M=2** circuit (3 qubits), the hardware-friendly variant. |
| `show_circuit.py` | Draws the circuit diagram(s) to PNG. |

## Run

```bash
. ../.venv/bin/activate     # or your own environment with requirements.txt
python asian_pricing.py
python asian_trivial.py
python show_circuit.py
```

## Expected output

* **Validation:** `G_circuit = G_exact` with `|diff| ≈ 6.7e-16` (machine precision).
* **COS price (M=3):** ≈ **0.398023** vs exact **0.397795**.
* **Scaling:** quantum MLAE error `~ 1/ε` vs classical Monte Carlo `~ 1/√N`.
* PNG figures: characteristic-function circuit, price comparison, query scaling.
