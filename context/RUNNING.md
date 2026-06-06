# RUNNING.md — how to run & integrate the system

**Entry point for a fresh session (human or LLM).** Read this to get the code running and to
know which folders matter. For *what we're building and why*, read
[`VISION.md`](./VISION.md) and [`../AGENTS.md`](../AGENTS.md) first — this file is the *how-to-run*,
not the pitch.

> **Status at last edit (2026-06-06):** `quantum_investor` (the motivation demo) is the live,
> verified, runnable piece — start there. `quantum_pricer` (the technical core) is built and merged
> to `main`; its Q50/IQM hardware path exists but has only been exercised on the *fake* backend.

---

## 1. Repo map — what each top-level folder is

| Path | What it is | Runnable? |
|---|---|---|
| **`quantum_investor/`** | **The motivation demo — START HERE.** A 1-qubit quantum-cognition "investor": reproduces human question-**order effects** that a classical Bayesian model structurally cannot, and satisfies the parameter-free **QQ-equality** (PNAS 2014). Self-contained, ~30 s to run. | ✅ verified |
| `quantum_pricer/` | The technical core: a quantum **option pricer** (CRR tree loading → QNDM phase encoding → QAE/QSVT readout), classical baselines, a 3-way benchmark, and a Q50/IQM hardware runner. Merged to `main`. | ⚠️ see §4 |
| `paper/` | The LaTeX derivation. `main_V2.tex` is the option-pricing math (technical core). `main_V3.tex` is a newer in-progress draft (untracked at last edit). | — (build with LaTeX) |
| `context/` | Shared docs. **`VISION.md` = single source of truth** (pitch, science, build status, task board). This `RUNNING.md`, plus the research trail (`literature-review.md`, `pressure-test.md`, `newton-feynman.md`, …) and `quantum-investor.html` (visual explainer — open in a browser). |
| `docs/` | Superpowers plans (e.g. the quantum-pricer benchmark implementation plan). |
| `AGENTS.md`, `README.md` | Top-level orientation for agents/teammates. |

**If you only do one thing:** run `quantum_investor` (§3). That's the piece we demo.

---

## 2. Prerequisites

- **Python 3.9+** (`python3`). Each package owns its own venv — there is **no repo-wide venv**.
- `quantum_investor` uses **PennyLane**; `quantum_pricer` uses **Qiskit + qiskit-iqm**. Their
  dependencies are independent — install per package, don't share a venv.
- macOS/Linux shell assumed (`source .venv/bin/activate`).

---

## 3. Run `quantum_investor` (the demo — ~30 s, verified path)

```bash
cd quantum_investor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py          # prints the story + numbers, saves figure.png
python test_model.py    # self-checks: QQ-equality, order effect, normalisation
```

`main.py` runs the whole pipeline end-to-end and writes `figure.png` (humans vs classical vs
quantum). `test_model.py` is the fast trust check — **both must pass before you push.**

**File roles inside `quantum_investor/`:**

| file | role |
|---|---|
| `quantum_model.py` | PennyLane belief-state circuit + sequential projective measurement of two non-commuting questions. **This is the science** (produces the order effect + QQ-equality). |
| `classical_model.py` | Bayesian baseline that *structurally* cannot produce an order effect (its predicted effect is always 0). |
| `data.py` | The "human" responses. Real dataset = **Clinton/Gore, PNAS 2014**. Swap/extend here for new data. |
| `fit.py` | Fits the quantum model's two parameters (α, β) to the data. |
| `plot.py` | The headline figure. |
| `main.py` | Runs everything end-to-end. |
| `test_model.py` | Fast self-checks so you trust the demo live. |

> Honest finding (don't oversell): the single-qubit quantum model does **not** out-*fit* the
> classical one on raw accuracy — the real win is the **parameter-free QQ-equality** (q ≈ −0.003),
> a prediction the classical model can't make at all. Keep that framing.

---

## 4. Run / integrate `quantum_pricer` (technical core + Q50)

Built and merged to `main`. Set up its own venv:

```bash
cd quantum_pricer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # qiskit, qiskit-aer, qiskit-iqm, pyqsp, ...
python -m pytest tests/ -q        # full self-check suite
python demo.py                    # end-to-end 3-way benchmark + dashboard
```

**Backend / hardware integration** is centralised in `backends.py` — swapping to real hardware is
a one-flag change in `run_hardware.py`:

| backend name | what it is |
|---|---|
| `local_aer` | qiskit-aer CPU (dev) |
| `lumi_aer` | qiskit-aer GPU/MPI scale-up (CPU fallback) |
| `q50_fake` | `IQMFakeAphrodite` (54q, native `{r, cz}`, noise model) — **offline**, the dev target |
| `q50_hw` | **real VTT Q50** via `qiskit-iqm` — on-site only, needs `IQM_SERVER_URL` + `IQM_TOKEN` |

```bash
python -m quantum_pricer.run_hardware --backend q50_fake --M 1   # offline noisy run
python -m quantum_pricer.run_hardware --backend q50_hw   --M 1   # on-site, real Q50
```

**Honest caveats for whoever integrates Q50:**
- `q50_hw` has **never actually executed** — only `q50_fake` is reachable offline. The real-hardware
  path is validated *only by transpilation* (`tests/test_hardware_transpile.py` asserts circuits
  compile to `{r, cz, measure, barrier}`). First on-site run is the real integration milestone.
- Expect **noise, not exactness**: `q50_fake` shows ~15% error at M=1. Real Q50 will differ.
- The shallow **Fourier route** is the Q50-feasible one (~112 CZ at M=4); deeper routes are not.

---

## 5. Conventions (don't break these)

- **`context/VISION.md` is the single source of truth.** If direction or the task board changes,
  update it there.
- **Keep `quantum_investor` runnable:** `python main.py` and `python test_model.py` must pass
  before pushing.
- **Hard guardrail (all external-facing material):** we do **not** claim the brain is a quantum
  computer. We claim human judgement violates *classical* probability exactly as *quantum*
  probability predicts — quantum-*like* math, real quantum algorithms for pricing.
- Each package keeps its own `.venv/` (git-ignored). Don't commit venvs, `__pycache__`, or
  `.DS_Store`.
