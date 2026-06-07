# Real Quantum Simulated Asian Option Pricing — Route I (QNDM)

This repository reproduces the **arithmetic Asian option** pricing pipeline of the
reference paper (see [`paper_to_reproduce.tex`](paper_to_reproduce.tex)) using
**Route I — the Quantum Non-Demolition Measurement (QNDM)** estimator of the
characteristic function. It contains self-contained scripts and the **real
results** obtained on two quantum platforms: **IBM Quantum** (`ibm_boston`,
Heron r3) and **IQM Q50** (via LUMI / Slurm).

Each subfolder is autonomous (no shared library): the scripts are faithful
copies of the working code, with the only modifications being credential
handling (now read from environment variables) and a counts-readout bugfix.

---

## 1. Theory in a nutshell

We price a discretely-monitored **arithmetic Asian call** with payoff
`max(Ā − K, 0)`, where `Ā = (1/M) Σ_t S_t` is the average of the underlying over
`M` monitoring dates. The underlying is modelled on a **Cox–Ross–Rubinstein
(CRR) binomial tree**: at each step the price moves up by `u` or down by `d`
with risk-neutral probability `q`.

Instead of loading the full payoff distribution, **Route I (QNDM)** estimates the
**characteristic function**

```
G(λ) = E[ exp( i λ (Ā − K) ) ]
```

of the average price `Ā` directly. A single ancilla **detector** qubit, prepared
in `|+⟩`, accumulates the controlled phase `λ(Ā − K)` through a diagonal
operator; measuring it in the **X basis** yields `Re G(λ)` and in the **Y basis**
yields `Im G(λ)`:

```
Re G(λ) = 2·Pr[det = 0 | X] − 1
Im G(λ) = 2·Pr[det = 0 | Y] − 1
```

Given `G(λ)` sampled on a grid, the **COS method** (Fourier-cosine expansion)
reconstructs the density of `Ā` and hence the discounted option price.

Two circuit families are used:

* **TRIVIAL (M=2, 3 qubits)** — a shallow circuit where the path qubits feed an
  8-entry diagonal directly on the detector (no running-weight register). It is
  the *recommended* circuit for real hardware because of its low depth.
* **FULL (M=3, 6 qubits)** — uses a running-weight register with
  controlled increment/decrement to accumulate `S_t`, then a controlled diagonal
  per time step. Deeper, hence more sensitive to hardware noise.

**Amplitude-estimation scaling.** Estimating `G` with Maximum-Likelihood
Amplitude Estimation (MLAE) achieves the quadratic speedup, i.e. error scaling
`~ 1/ε` in the number of queries versus the classical Monte Carlo `~ 1/√N`.

---

## 2. Real results

### IQM Q50 (noiseless simulation + Q50 hardware) — from `asian-pricing-19087672.out`

| Quantity | Value |
|---|---|
| Noiseless validation | `G_circuit = G_exact`, `|diff| = 6.7e-16` (perfect) |
| COS price (M=3) | **0.398023** vs exact **0.397795** |
| MLAE scaling | `1/ε` (quantum) vs `1/√N` (classical) confirmed |
| Q50 hardware (M=3) | `G = −0.1133 + 0.3438j` (degraded by circuit depth) |

The M=3 circuit on real Q50 hardware is degraded by depth — consistent with the
recommendation to use the shallow M=2 trivial circuit on hardware.

### IBM Quantum (`ibm_boston`, Heron r3)

| Circuit | `|dG| = |G_hw − G_noiseless|` | Verdict |
|---|---|---|
| **M=2 (trivial, 3 qubits)** | **0.0142** | ✅ excellent agreement |
| **M=3 (full, 6 qubits)** | **0.2242** | ⚠️ degraded by depth |

The shallow M=2 circuit reproduces the noiseless characteristic function on real
hardware to within `0.0142`; the deeper M=3 circuit degrades as expected.

---

## 3. Repository structure

```
real quantum simulated asian pricing/
├── README.md                  # this file
├── requirements.txt           # qiskit, qiskit-ibm-runtime, iqm-qiskit, numpy, matplotlib
├── paper_to_reproduce.tex     # reference paper
│
├── 1_simulation/              # noiseless, no hardware/account required
│   ├── asian_pricing.py       # M=3: validates G, COS price vs exact/MC, MLAE 1/ε scaling
│   ├── asian_trivial.py       # M=2 shallow
│   ├── show_circuit.py
│   └── README.md
│
├── 2_ibmq/                    # IBM Quantum runs (SamplerV2 + DD/twirling)
│   ├── asian_ibm.py           # token via env (IBM_TOKEN/IBM_CRN), M=2 and --full M=3
│   ├── fetch_ibm.py           # retrieve a job by id (token via env)
│   ├── README.md
│   └── results/               # PNG + log + counts of the real ibm_boston runs
│
└── 3_iqm_q50/                 # IQM Q50 runs via LUMI (Slurm)
    ├── asian_pricing.py
    ├── asian_trivial.py
    ├── show_circuit.py
    ├── run_q50.sh
    ├── how_to_use_lumi_q50.md
    ├── README.md
    └── results/
```

---

## 4. How to run

```bash
# 1) set up the environment
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 2) noiseless simulation (no account needed)
cd 1_simulation
python asian_pricing.py
python asian_trivial.py

# 3) IBM Quantum (requires credentials via env vars)
export IBM_TOKEN="<your IBM Quantum Platform API token>"
export IBM_CRN="<your instance CRN>"
cd ../2_ibmq
python asian_ibm.py            # trivial M=2 (default)
python asian_ibm.py --full     # also the M=3 circuit
python asian_ibm.py --no-hw    # noiseless only

# 4) IQM Q50 on LUMI — see 3_iqm_q50/how_to_use_lumi_q50.md
```

> **Security note:** IBM credentials are **never hardcoded**. The scripts read
> `IBM_TOKEN` and `IBM_CRN` from environment variables and print a clear error if
> they are missing.
