# Overnight quantum-finance triage → demo lab — design

**Date:** 2026-06-06
**Project:** qhack (qpizza) — Junction Helsinki × OP Pohjola "Quantum Computing for Finance"
**Status:** approved design, pre-implementation

## Goal

By morning, produce two things:

1. A **ranked, evidence-backed recommendation** of which candidate idea to commit
   to for the OP demo — where "evidence" is measured quantum-vs-classical results
   plus Q50-hardware readiness, not opinion.
2. A **runnable, backend-agnostic demo shell** already wired to the frontrunner,
   so the event day is spent polishing a natural end-to-end solution rather than
   deciding what to build and scaffolding from zero.

Success criterion: we wake to a `triage/REPORT.md` that lets the team pick and
defend an idea on all four judging axes (novelty, problem formulation, technical
depth, business case) in ~10 minutes, and a `demo/` app that launches and runs an
inference against at least the local simulator.

## Context and constraints

- Judging is 4 criteria at 25% each. Dominant failure mode: "classical method
  with a quantum label." Litmus test: delete the quantum part; if the problem
  collapses to standard classical, it is weak.
- Decision log in `context/ideas.md` is empty — no idea is locked. Hence **triage
  mode**: the overnight run produces the evidence to choose.
- Hardware available: **VTT Q50** (VTT + IQM, 50-qubit superconducting, connected
  to LUMI) and **LUMI** simulators. Compatibility is a hard requirement.
- The end deliverable is an *impressive, natural, end-to-end, usable* demo — more
  than a table of numbers.
- One night of human build time; the overnight run must be unattended and robust.

## Locked decisions (from brainstorming)

1. **Triage mode** — pick the strongest idea with evidence by morning.
2. **Lightweight bespoke loop** — reuse efferents *patterns* (file-based state,
   budget discipline, popper gate, notify, simple orchestrator) but not its
   QML-hardcoded code. efferents' own CLAUDE.md states it is not yet a generic
   framework; generalizing it tonight would risk never reaching "running."
3. **We hand-build harness + thin PoCs tonight**; overnight is a deterministic
   sweep + digest. Almost nothing depends on an autonomous coder behaving.
4. **Candidate scope:** QAE family (option pricing B + VaR/CVaR E, one harness),
   QAOA portfolio (A), QML-fraud (D — elevated to co-headline + demo frontrunner).
   qGAN vol-surface (C) deferred (training instability, weakest overnight
   advantage signal).
5. **Backend layer on Qiskit as the common substrate**; Q50 targeted **offline-
   faithful via IQMFakeBackend** overnight, real hardware reserved for on-site.
6. **Scaffold a backend-agnostic demo shell tonight**, pre-wired to the likely
   winner (fraud console).

## Backend layer (the compatibility contract)

`backends/get_backend(name)` returns a Qiskit-compatible backend:

| Name | Implementation | Use |
|---|---|---|
| `local_aer` | `qiskit-aer` (CPU) | dev + tonight's builds/tests |
| `lumi_aer` | `qiskit-aer` (GPU/MPI) | scale-up sim (FiQCI has run Aer to 44 qubits on LUMI) |
| `q50_fake` | **`IQMFakeBackend`** (qiskit-on-iqm) | Q50 native gates (phased-rx + CZ) + noise model — proves Q50-compatibility offline, no queue |
| `q50_hw` | `qiskit_iqm` (VTT QX / LUMI-Q) | **guarded, on-site only** — never invoked by the overnight run |

PennyLane (QML-fraud track) reaches the same backends via `pennylane-qiskit`.
Every harness is backend-agnostic; the demo exposes a backend dropdown.

Rationale: `IQMFakeBackend` is the key unlock — the overnight loop can verify
"this circuit transpiles to Q50's native gate set and survives its noise model"
(real hardware-readiness evidence) **without** depending on a hardware queue,
auth, or latency overnight. Real Q50 runs become the on-site wow, not a
dependency.

Expected triage consequence: QAE circuits are deep (amplitude amplification) →
honestly a **LUMI-sim** story (likely fails/degrades on `q50_fake`). QAOA and
QML-fraud are shallow/variational → **Q50-runnable**. "Can it run on Europe's
first 50-qubit machine" is therefore a first-class, measured triage axis.

## Common scoring axis — AdvantageRecord

Every harness run emits one record with the same schema, so candidates compare
despite different natural metrics. Defined in `triage/rubric.py`.

| Field | QAE (B/E) | QAOA (A) | QML-fraud (D) |
|---|---|---|---|
| `quantum_metric` | samples→ε for E[payoff]/VaR/CVaR | approx. ratio vs optimum | accuracy / AUC |
| `classical_metric` | Monte-Carlo samples→ε | classical heuristic (SA / greedy / exact) | RBF / linear-kernel SVM |
| `advantage_direction` | win / tie / loss | win / tie / loss | win / tie / loss |
| `scaling_signature` | log-log slope (∼1/ε vs 1/ε²) | ratio vs depth p | kernel geometric difference |
| `quantum_native_litmus` | delete-quantum → collapses? | " | " |
| `sim_runnable` | did it run on Aer; qubits, wall-clock | " | " |
| `q50_faithful_runnable` | transpiles to native gates + survives IQMFakeBackend noise | " | " |
| `demo_naturalness` | how visceral/usable the demo is | " | " |
| `op_business_fit` | OP product line + € story | " | " |

The **scaling signature** is the core insight: even at small simulable scale, the
*slope* (QAE 1/ε vs MC 1/ε²) is the quantum-native fingerprint — a measurable
advantage signal that survives NISQ honesty. The scoring formula in `rubric.py`
combines measured-advantage, Q50-readiness, demo-naturalness, and business-fit
into a single ranking; weights mirror the four 25% judging criteria.

## Popperian spine

One interactive popper-probe pass forges a **template hypothesis** + falsifier:

> "Quantum approach X yields a meaningful, quantum-native advantage over the best
> classical baseline on metric M for problem P."
> **Falsifier (= the originality trap, operationalized):** "a classical baseline
> matches or beats it at the scale we can simulate, with no asymptotic backing."

Instantiated per candidate into `popper-corpus/<cand>/hypothesis.md` (swap
P/X/M/baseline). These are pre-registered, falsifiable claims — both demo
credibility ("we tried to disprove the advantage of each; here's what survived")
and the source of each candidate's sweep spec / test design.

## Architecture

```
qhack/
  backends/__init__.py         # get_backend(): local_aer | lumi_aer | q50_fake | q50_hw
  triage/
    orchestrator.py            # sweep runner: run each config, append record,
                               #   catch+log failures, checkpoint/resume — never crashes the night
    rubric.py                  # AdvantageRecord schema + scoring/ranking formula
    digest.py                  # deterministic REPORT.md + plots; optional LLM analyst layer
    dashboard.py               # regenerates self-contained dashboard.html (auto-refresh, per-method)
    harness/
      qae.py                   # IQAE/MLQAE — option pricing (B) + VaR/CVaR (E)
      qaoa.py                  # cardinality-constrained portfolio QUBO
      fraud_qml.py             # quantum-kernel + variational classifier
    baselines/
      mc.py                    # Monte-Carlo expectation
      classical_opt.py         # simulated annealing / exact for small portfolios
      classical_kernel.py      # RBF / linear-kernel SVM
    records.jsonl              # the ledger (file-based state, efferents-style)
    plots/                     # auto-generated advantage curves (PNG, embedded into dashboard)
    dashboard.html            # live self-contained dashboard (leave open overnight)
    REPORT.md                  # morning output
  data/                        # ULB credit-card fraud, subsampled → ≤10 features/qubits
  sweeps/all.yaml              # per-candidate config grids
  demo/
    app.py                     # Streamlit shell, backend dropdown, fraud console
    inference.py               # backend-agnostic inference fn the shell calls
  popper-corpus/<cand>/hypothesis.md
  tests/                       # known-answer unit test per harness + 30s smoke sweep
```

Tooling: `uv` for the venv (match efferents). Core deps: `qiskit`,
`qiskit-aer`, `qiskit-finance`, `qiskit-optimization`, `qiskit-algorithms`,
`qiskit-iqm` (for IQMFakeBackend), `pennylane`, `pennylane-qiskit`, `scikit-learn`,
`numpy`, `pandas`, `matplotlib`, `scipy`, `streamlit`, `pyyaml`.

## Overnight execution

```
caffeinate -i uv run python -m triage.orchestrator --sweep sweeps/all.yaml
```

Terminal-launched (macOS launchd has TCC issues reading `~/Documents`; the
`caffeinate` terminal path is the realistic agent path, per efferents notes).

- Experiments are **local, deterministic, $0, no network** → near-zero unattended
  risk. Only the optional analyst step would cost anything.
- The orchestrator **always** writes a deterministic, ranked `REPORT.md` + plots
  via `digest.py` (no API dependency, guaranteed output), and regenerates the live
  `dashboard.html` after every completed config so progress is watchable in a
  browser overnight.
- `ANTHROPIC_API_KEY` is **not set** in this environment, so the optional LLM
  analyst narrative is a no-op by default; if a key is provided later, one analyst
  call layers a narrative + draft "why quantum-native" slide on top. The
  deterministic report stands alone regardless.
- Checkpoint/resume: a crash resumes from the last completed config rather than
  restarting the night.

## Live dashboard

`triage/dashboard.py` regenerates a **single self-contained `triage/dashboard.html`**
(no server, no external assets — plots embedded as base64 PNGs, like efferents'
`progress.py`) **after every completed config**, so it can be left open in a
browser overnight and watched live.

- **Auto-refresh:** a `<meta http-equiv="refresh" content="30">` tag re-loads the
  file every 30s; opening it via `file://` is enough (no server).
- **Per-method cards:** one card per method (QAE, QAOA, QML-fraud) with:
  - a **brief plain-language description** of the method and its quantum-native claim,
  - the headline AdvantageRecord fields (advantage direction win/tie/loss,
    measured quantum vs classical metric, scaling signature),
  - **visuals**: the method's advantage curve(s) (e.g. QAE samples→ε vs MC,
    QAOA approx-ratio vs depth, fraud ROC/AUC) embedded inline,
  - **Q50-readiness badge** (`q50_faithful_runnable`) and a `demo_naturalness` /
    `op_business_fit` line.
- **Top banner:** overall progress (configs done / total), current leader per the
  ranking formula, and last-updated timestamp.

`dashboard.py` reads only `records.jsonl` + `plots/`, so it is pure presentation
and can be re-run standalone at any time. It shares the ranking formula with
`digest.py` (both import from `rubric.py`) so the live leader and the morning
`REPORT.md` never disagree.

## Testing

- **Known-answer unit test per harness** so we trust the numbers before sweeping:
  - QAE on a Bernoulli with analytic E[payoff].
  - QAOA on a 3-asset problem with brute-force optimum.
  - QML-fraud on a linearly separable toy (sanity of the classifier wiring).
- **30-second smoke sweep** (one tiny config per candidate) run before the real
  launch to confirm the orchestrator, records ledger, and digest all work
  end-to-end.
- Each harness additionally records a `q50_faithful_runnable` result by
  transpiling + running its circuit on `q50_fake` — a deliberate compatibility
  assertion, not just a metric.

## Tonight's critical path

1. popper-probe template hypothesis → per-candidate `hypothesis.md`
2. `backends/` layer with `IQMFakeBackend` wired and verified
3. QAE harness (covers B + E) + Monte-Carlo baseline + known-answer test
4. QAOA harness + classical baseline + known-answer test
5. fraud data prep (ULB) + fraud-QML harness + classical-kernel baseline + test
6. `rubric.py` + `orchestrator.py` + `digest.py` + `dashboard.py`
7. `sweeps/all.yaml`
8. demo shell wired to the fraud console (local_aer), backend dropdown
9. 30-second smoke sweep → fix → launch under `caffeinate` (open `dashboard.html`)

## Decisions made by default (no objection raised)

- **Fraud dataset:** ULB credit-card fraud (Kaggle) — PCA features (V1..V28),
  highly imbalanced; subsample + select top features → ≤10 qubits.
- **Demo UI:** Streamlit — Python-native, fastest path to a usable shell.

## Out of scope

- Real VTT Q50 / LUMI-Q hardware submission overnight (reserved for on-site).
- Generalizing the efferents framework (LabConfig refactor etc.).
- qGAN volatility-surface candidate (C).
- A production system — this is a hackathon PoC + triage, per the challenge brief.
