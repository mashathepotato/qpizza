# 3 — IQM Q50 (via LUMI / Slurm)

Runs of the Route-I (QNDM) Asian pricing circuits on the **IQM Q50** quantum
computer, accessed through the **LUMI** supercomputer with a Slurm batch job. The
same scripts validate noiselessly and then submit to the Q50 hardware via the
Cortex URL hook.

## Files

| File | What it does |
|---|---|
| `asian_pricing.py` | Full **M=3** pipeline with the Q50 hardware hook (`Q50_CORTEX_URL`). |
| `asian_trivial.py` | Shallow **M=2** circuit — recommended on Q50 because of its low depth. |
| `show_circuit.py` | Draws the circuit diagram(s). |
| `run_q50.sh` | Slurm batch script to submit the job on LUMI. |
| `how_to_use_lumi_q50.md` | Step-by-step guide to LUMI / Q50 access and submission. |

## Run (on LUMI)

```bash
# set the Q50 endpoint (see how_to_use_lumi_q50.md)
export Q50_CORTEX_URL="<cortex endpoint>"
sbatch run_q50.sh
```

For local noiseless validation only, run `python asian_pricing.py` / `asian_trivial.py`
without the Q50 hook.

## Real results (`asian-pricing-19087672.out`)

| Quantity | Value |
|---|---|
| Noiseless validation | `G_circuit = G_exact`, `|diff| = 6.7e-16` (perfect) |
| COS price (M=3) | **0.398023** vs exact **0.397795** |
| MLAE scaling | `1/ε` (quantum) vs `1/√N` (classical) |
| **Q50 hardware (M=3)** | `G = −0.1133 + 0.3438j` (degraded by circuit depth) |

The deep M=3 circuit is significantly degraded on real Q50 hardware — as on IBM,
the shallow M=2 trivial circuit is the hardware-friendly choice.

## `results/`

| File | Content |
|---|---|
| `asian-pricing-19087672.out` | Full Slurm job output (noiseless validation + Q50 run). |
| `asian_char_circuit.png` | Characteristic-function circuit diagram. |
| `asian_price_comparison.png` | COS price vs exact / Monte Carlo. |
| `asian_query_scaling.png` | MLAE `1/ε` vs Monte Carlo `1/√N` scaling. |
