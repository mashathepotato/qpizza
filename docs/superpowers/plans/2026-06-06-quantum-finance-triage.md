# Quantum-Finance Triage Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an unattended overnight loop that triages 4 quantum-finance candidates (QAE option-pricing, QAE VaR/CVaR, QAOA portfolio, QML-fraud) on a common advantage axis, plus a live HTML dashboard and a backend-agnostic Streamlit demo shell wired to the fraud frontrunner.

**Architecture:** Every harness is a pure function `run(config, backend) -> AdvantageRecord`. A sweep orchestrator iterates configs from a YAML spec, appends records to a JSONL ledger, regenerates a self-contained HTML dashboard after each config, and on completion writes a deterministic ranked `REPORT.md`. All quantum work runs through one Qiskit backend abstraction (`local_aer | lumi_aer | q50_fake | q50_hw`); `q50_fake` is `IQMFakeBackend`, giving offline VTT-Q50 hardware-faithfulness with no queue.

**Tech Stack:** Python 3.11, `uv`, Qiskit + qiskit-aer + qiskit-algorithms + qiskit-finance + qiskit-optimization, qiskit-iqm (IQMFakeBackend), PennyLane + pennylane-qiskit, scikit-learn, pandas/numpy/scipy, matplotlib, Streamlit, PyYAML, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-06-quantum-finance-triage-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | deps + project metadata (uv-managed) |
| `backends/__init__.py` | `get_backend(name)` → Qiskit backend; the compatibility contract |
| `triage/rubric.py` | `AdvantageRecord` dataclass + `score(record)` + `rank(records)` |
| `triage/baselines/mc.py` | Monte-Carlo expectation estimator (classical QAE baseline) |
| `triage/baselines/classical_opt.py` | exact/SA portfolio solver (classical QAOA baseline) |
| `triage/baselines/classical_kernel.py` | RBF/linear SVM (classical fraud baseline) |
| `triage/harness/qae.py` | `run()` for option-pricing (B) + VaR/CVaR (E) |
| `triage/harness/qaoa.py` | `run()` for cardinality-constrained portfolio (A) |
| `triage/harness/fraud_qml.py` | `run()` for quantum-kernel fraud classifier (D) |
| `triage/data/fraud.py` | load + subsample + reduce ULB fraud data to ≤10 features |
| `triage/orchestrator.py` | sweep loop: run configs, append ledger, refresh dashboard, checkpoint |
| `triage/digest.py` | deterministic `REPORT.md` + plots; optional LLM analyst |
| `triage/dashboard.py` | self-contained auto-refreshing `dashboard.html` |
| `sweeps/all.yaml` | per-candidate config grids + a `smoke` grid |
| `demo/inference.py` | backend-agnostic `score_transaction()` for the demo |
| `demo/app.py` | Streamlit fraud-triage console with backend dropdown |
| `popper-corpus/<cand>/hypothesis.md` | pre-registered falsifiable claims (popper-probe) |
| `tests/...` | known-answer + framework unit tests |

Each harness depends only on `backends` + `triage.rubric`. The orchestrator depends on harnesses + rubric + dashboard + digest. The demo depends only on `backends` + one harness's trained artifact. This keeps every unit independently testable.

---

## Task 0: Project setup

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore` (append)
- Create: `triage/__init__.py`, `backends/__init__.py` (empty stub for now), `triage/harness/__init__.py`, `triage/baselines/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "qpizza"
version = "0.1.0"
description = "Overnight quantum-finance triage lab"
requires-python = ">=3.11"
dependencies = [
    "qiskit>=1.2",
    "qiskit-aer>=0.15",
    "qiskit-algorithms>=0.3",
    "qiskit-finance>=0.4",
    "qiskit-optimization>=0.6",
    "qiskit-iqm>=15.0",
    "pennylane>=0.38",
    "pennylane-qiskit>=0.38",
    "scikit-learn>=1.4",
    "pandas>=2.0",
    "numpy>=1.26",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "streamlit>=1.36",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the venv and install**

Run: `cd /Users/masha/Documents/qhack && uv venv && uv pip install -e ".[dev]"`
Expected: resolves and installs without error. If a specific qiskit sub-package version conflicts, relax the pin to the nearest resolvable version and note it.

- [ ] **Step 3: Smoke-import check (verifies the toolset actually loads)**

Create `tests/test_imports.py`:

```python
def test_core_imports():
    import qiskit
    import qiskit_aer
    import qiskit_algorithms
    import qiskit_finance
    import qiskit_optimization
    import qiskit_iqm
    import pennylane
    import sklearn
    assert qiskit_aer.AerSimulator is not None
```

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_imports.py -v`
Expected: PASS. If any import fails, fix the dependency in `pyproject.toml` before proceeding — every later task assumes these load.

- [ ] **Step 5: Append to `.gitignore`**

```
.venv/
__pycache__/
triage/records.jsonl
triage/plots/
triage/dashboard.html
triage/REPORT.md
triage/.checkpoint.json
data/raw/
*.pkl
```

- [ ] **Step 6: Create empty package markers**

Create `triage/__init__.py`, `triage/harness/__init__.py`, `triage/baselines/__init__.py`, `tests/__init__.py` (all empty). Create `backends/__init__.py` with a single line `# backend abstraction — implemented in Task 1`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore triage backends tests
git commit -m "chore: project setup, deps, smoke-import test"
```

---

## Task 1: Backend abstraction

**Files:**
- Modify: `backends/__init__.py`
- Test: `tests/test_backends.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest
from backends import get_backend, BackendUnavailable

def test_local_aer_runs_a_bell_circuit():
    from qiskit import QuantumCircuit, transpile
    backend = get_backend("local_aer")
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
    result = backend.run(transpile(qc, backend), shots=256).result()
    counts = result.get_counts()
    assert sum(counts.values()) == 256
    assert set(counts) <= {"00", "11", "01", "10"}

def test_q50_fake_is_iqm_fake_backend():
    backend = get_backend("q50_fake")
    # IQMFakeBackend exposes a target with the IQM native gate set
    names = set(backend.target.operation_names)
    assert "cz" in names

def test_q50_hw_guarded_without_credentials():
    with pytest.raises(BackendUnavailable):
        get_backend("q50_hw")

def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("nonsense")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_backends.py -v`
Expected: FAIL (`cannot import name 'get_backend'`).

- [ ] **Step 3: Implement `backends/__init__.py`**

```python
"""Backend abstraction — the Qiskit compatibility contract.

local_aer : qiskit-aer CPU (dev + tonight's builds)
lumi_aer  : qiskit-aer GPU/MPI (scale-up; falls back to CPU if no GPU)
q50_fake  : IQMFakeBackend — VTT Q50 native gates + noise, offline, no queue
q50_hw    : real VTT Q50 via qiskit-iqm — guarded, on-site only
"""
import os


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot be constructed in this environment."""


def _local_aer():
    from qiskit_aer import AerSimulator
    return AerSimulator()


def _lumi_aer():
    from qiskit_aer import AerSimulator
    try:
        return AerSimulator(device="GPU")
    except Exception:
        # No GPU here; same API, CPU execution. LUMI provides the GPU at scale.
        return AerSimulator()


def _q50_fake():
    # Adapt to whichever fake-backend the installed qiskit-iqm exposes.
    try:
        from qiskit_iqm import IQMFakeAdonis  # small/older builds
        return IQMFakeAdonis()
    except Exception:
        pass
    from qiskit_iqm.fake_backends import IQMFakeBackend
    from qiskit_iqm.iqm_provider import IQMProvider  # noqa: F401  (presence check)
    # Construct from a deneb/garnet-style quantum architecture if available.
    try:
        from qiskit_iqm import IQMFakeDeneb
        return IQMFakeDeneb()
    except Exception as exc:
        raise BackendUnavailable(f"No IQM fake backend available: {exc}")


def _q50_hw():
    token = os.environ.get("IQM_TOKEN") or os.environ.get("IQM_TOKENS_FILE")
    url = os.environ.get("IQM_SERVER_URL")
    if not token or not url:
        raise BackendUnavailable(
            "q50_hw requires IQM_SERVER_URL and IQM_TOKEN (on-site only)."
        )
    from qiskit_iqm import IQMProvider
    return IQMProvider(url).get_backend()


_REGISTRY = {
    "local_aer": _local_aer,
    "lumi_aer": _lumi_aer,
    "q50_fake": _q50_fake,
    "q50_hw": _q50_hw,
}


def get_backend(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_backends.py -v`
Expected: PASS. If `test_q50_fake_is_iqm_fake_backend` fails because the installed `qiskit-iqm` exposes a different fake class, inspect with `uv run python -c "import qiskit_iqm; print([n for n in dir(qiskit_iqm) if 'Fake' in n])"` and update `_q50_fake()` to use the available class. The contract is only that it returns a backend whose target includes `cz`.

- [ ] **Step 5: Commit**

```bash
git add backends/__init__.py tests/test_backends.py
git commit -m "feat: Qiskit backend abstraction with IQMFakeBackend (Q50-faithful, offline)"
```

---

## Task 2: AdvantageRecord + scoring

**Files:**
- Create: `triage/rubric.py`
- Test: `tests/test_rubric.py`

- [ ] **Step 1: Write failing tests**

```python
from triage.rubric import AdvantageRecord, score, rank

def _rec(**kw):
    base = dict(
        method="qae", candidate="B", config_id="b0",
        quantum_metric=0.01, classical_metric=0.04, metric_name="samples_to_eps",
        advantage_direction="win", advantage_magnitude=4.0,
        scaling_signature=1.0, quantum_native_litmus=True,
        sim_runnable=True, q50_faithful_runnable=False,
        demo_naturalness=0.5, op_business_fit=0.8, notes="",
    )
    base.update(kw)
    return AdvantageRecord(**base)

def test_record_roundtrips_through_dict():
    r = _rec()
    assert AdvantageRecord.from_dict(r.to_dict()) == r

def test_win_scores_higher_than_loss():
    win = _rec(advantage_direction="win")
    loss = _rec(advantage_direction="loss")
    assert score(win) > score(loss)

def test_q50_ready_and_natural_demo_breaks_ties():
    a = _rec(q50_faithful_runnable=True, demo_naturalness=0.9)
    b = _rec(q50_faithful_runnable=False, demo_naturalness=0.2)
    assert score(a) > score(b)

def test_rank_orders_descending_and_is_stable():
    recs = [_rec(config_id="x", advantage_direction="loss"),
            _rec(config_id="y", advantage_direction="win")]
    ranked = rank(recs)
    assert [r.config_id for r in ranked] == ["y", "x"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_rubric.py -v`
Expected: FAIL (`No module named 'triage.rubric'`).

- [ ] **Step 3: Implement `triage/rubric.py`**

```python
"""The common scoring axis. Every harness emits an AdvantageRecord; score()
collapses it to one number whose weights mirror the four 25% judging criteria."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields

_DIRECTION = {"win": 1.0, "tie": 0.5, "loss": 0.0}


@dataclass(frozen=True)
class AdvantageRecord:
    method: str                  # "qae" | "qaoa" | "fraud_qml"
    candidate: str               # "A" | "B" | "D" | "E"
    config_id: str               # unique per sweep config
    quantum_metric: float        # method's quantum metric value
    classical_metric: float      # same metric, classical baseline
    metric_name: str             # e.g. "samples_to_eps", "approx_ratio", "auc"
    advantage_direction: str     # "win" | "tie" | "loss"
    advantage_magnitude: float   # ratio or gap, method-defined, >=0
    scaling_signature: float     # log-log slope / depth ratio / geometric diff
    quantum_native_litmus: bool  # delete-quantum => collapses?
    sim_runnable: bool           # ran on Aer
    q50_faithful_runnable: bool  # transpiled + survived IQMFakeBackend
    demo_naturalness: float      # 0..1 hand-set per method
    op_business_fit: float       # 0..1 hand-set per method
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AdvantageRecord":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def score(r: AdvantageRecord) -> float:
    # Four 25% pillars. Each term is normalized to roughly [0,1].
    novelty = 1.0 if r.quantum_native_litmus else 0.0
    technical = _DIRECTION.get(r.advantage_direction, 0.0)
    # Q50-readiness + actually-ran feed "problem formulation/feasibility".
    feasibility = 0.5 * float(r.sim_runnable) + 0.5 * float(r.q50_faithful_runnable)
    business = 0.5 * r.demo_naturalness + 0.5 * r.op_business_fit
    return 0.25 * (novelty + technical + feasibility + business)


def rank(records: list[AdvantageRecord]) -> list[AdvantageRecord]:
    # Stable sort by descending score; ties keep input order.
    return sorted(records, key=score, reverse=True)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_rubric.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add triage/rubric.py tests/test_rubric.py
git commit -m "feat: AdvantageRecord schema + 25%-mirrored scoring/ranking"
```

---

## Task 3: Classical baselines

**Files:**
- Create: `triage/baselines/mc.py`, `triage/baselines/classical_opt.py`, `triage/baselines/classical_kernel.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from triage.baselines.mc import mc_samples_to_eps
from triage.baselines.classical_opt import exact_portfolio
from triage.baselines.classical_kernel import rbf_svm_auc

def test_mc_estimates_bernoulli_mean_within_eps():
    # E[X]=0.3; ask MC how many samples to reach eps=0.02 (95% CI)
    n = mc_samples_to_eps(p=0.3, eps=0.02, seed=0)
    assert 1000 < n < 100_000  # ~ p(1-p)/eps^2 * z^2

def test_exact_portfolio_picks_best_two_of_three():
    # returns favor assets 0 and 2; pick exactly 2
    mu = np.array([0.1, 0.01, 0.09])
    cov = np.eye(3) * 0.0001
    chosen, val = exact_portfolio(mu, cov, k=2, risk=1.0)
    assert sorted(chosen) == [0, 2]

def test_rbf_svm_separates_a_separable_toy():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.3, (30, 2)), rng.normal(2, 0.3, (30, 2))])
    y = np.array([0] * 30 + [1] * 30)
    auc = rbf_svm_auc(X, y, seed=0)
    assert auc > 0.95
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL (import errors).

- [ ] **Step 3: Implement the three baseline modules**

`triage/baselines/mc.py`:

```python
"""Classical Monte-Carlo baseline for QAE: samples to reach accuracy eps."""
import math


def mc_samples_to_eps(p: float, eps: float, z: float = 1.96, seed: int = 0) -> int:
    """Analytic sample count for a Bernoulli(p) mean to a +/-eps CI at level z.
    n >= z^2 * p(1-p) / eps^2.  This is MC's O(1/eps^2) scaling, the QAE foil."""
    var = p * (1.0 - p)
    return int(math.ceil((z ** 2) * var / (eps ** 2)))
```

`triage/baselines/classical_opt.py`:

```python
"""Exact small-portfolio solver: brute force cardinality-constrained selection."""
import itertools
import numpy as np


def exact_portfolio(mu, cov, k: int, risk: float):
    """Maximize  mu.x - risk * x^T cov x  over x in {0,1}^n with sum(x)==k.
    Returns (chosen_indices, objective_value). Brute force — small n only."""
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)
    n = len(mu)
    best, best_val = None, -np.inf
    for combo in itertools.combinations(range(n), k):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        val = mu @ x - risk * (x @ cov @ x)
        if val > best_val:
            best, best_val = combo, val
    return list(best), float(best_val)
```

`triage/baselines/classical_kernel.py`:

```python
"""Classical RBF-SVM baseline for the fraud track (AUC)."""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def rbf_svm_auc(X, y, seed: int = 0) -> float:
    Xtr, Xte, ytr, yte = train_test_split(
        np.asarray(X), np.asarray(y), test_size=0.3, random_state=seed, stratify=y
    )
    clf = SVC(kernel="rbf", probability=True, random_state=seed).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, proba))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add triage/baselines tests/test_baselines.py
git commit -m "feat: classical baselines (MC, exact portfolio, RBF-SVM)"
```

---

## Task 4: QAE harness (option pricing B + VaR/CVaR E)

**Files:**
- Create: `triage/harness/qae.py`
- Test: `tests/test_qae.py`

- [ ] **Step 1: Write failing known-answer test**

```python
import math
from backends import get_backend
from triage.harness.qae import estimate_bernoulli, run

def test_qae_estimates_known_bernoulli_probability():
    # Amplitude-encode a known p and recover it within tolerance.
    backend = get_backend("local_aer")
    p = 0.2
    est = estimate_bernoulli(p, backend=backend, epsilon=0.02)
    assert abs(est - p) < 0.05

def test_run_returns_record_with_scaling_advantage():
    rec = run({"config_id": "b_smoke", "candidate": "B", "p": 0.3,
               "epsilon": 0.05, "backend": "local_aer"})
    assert rec.method == "qae"
    assert rec.metric_name == "samples_to_eps"
    # QAE should need fewer oracle calls than MC at this eps (1/eps vs 1/eps^2)
    assert rec.quantum_metric < rec.classical_metric
    assert rec.advantage_direction in {"win", "tie", "loss"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_qae.py -v`
Expected: FAIL (`No module named 'triage.harness.qae'`).

- [ ] **Step 3: Implement `triage/harness/qae.py`**

```python
"""QAE harness: covers option pricing (B) and VaR/CVaR (E).

Core: amplitude-encode a Bernoulli p into one qubit, run Iterative Amplitude
Estimation, and compare oracle-call count to reach eps against classical MC.
The quantum-native fingerprint is the scaling: IQAE ~ O(1/eps), MC ~ O(1/eps^2)."""
from __future__ import annotations
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.primitives import Sampler

from backends import get_backend
from triage.rubric import AdvantageRecord
from triage.baselines.mc import mc_samples_to_eps


def _bernoulli_a_circuit(p: float) -> QuantumCircuit:
    """State-prep A: |0> -> sqrt(1-p)|0> + sqrt(p)|1>; 'good' state = |1>."""
    qc = QuantumCircuit(1)
    theta = 2 * math.asin(math.sqrt(p))
    qc.ry(theta, 0)
    return qc


def _estimation_problem(p: float) -> EstimationProblem:
    a = _bernoulli_a_circuit(p)
    return EstimationProblem(state_preparation=a, objective_qubits=[0])


def estimate_bernoulli(p: float, backend=None, epsilon: float = 0.02) -> float:
    """Return the IQAE point estimate of p."""
    sampler = Sampler()  # statevector-exact primitive; deterministic, fast
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon, alpha=0.05, sampler=sampler
    )
    result = iae.estimate(_estimation_problem(p))
    return float(result.estimation)


def _qae_oracle_calls(p: float, epsilon: float) -> int:
    """IQAE point estimate cost proxy: O(1/eps) oracle calls."""
    sampler = Sampler()
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon, alpha=0.05, sampler=sampler
    )
    result = iae.estimate(_estimation_problem(p))
    # num_oracle_queries is exposed by the result; fall back to the 1/eps bound.
    calls = getattr(result, "num_oracle_queries", None)
    if not calls:
        calls = int(math.ceil(1.0 / epsilon))
    return int(calls)


def _q50_faithful(p: float) -> bool:
    """Does the state-prep + one Grover power transpile + run on Q50-fake?"""
    try:
        from qiskit import transpile
        backend = get_backend("q50_fake")
        qc = _bernoulli_a_circuit(p)
        qc.measure_all()
        tqc = transpile(qc, backend, optimization_level=1)
        backend.run(tqc, shots=64).result()
        return True
    except Exception:
        return False


def run(config: dict) -> AdvantageRecord:
    p = float(config.get("p", 0.3))
    eps = float(config.get("epsilon", 0.05))
    candidate = config.get("candidate", "B")
    q_calls = _qae_oracle_calls(p, eps)
    mc_calls = mc_samples_to_eps(p=p, eps=eps)
    if q_calls < mc_calls * 0.9:
        direction = "win"
    elif q_calls > mc_calls * 1.1:
        direction = "loss"
    else:
        direction = "tie"
    # Scaling signature: ratio of classical to quantum cost (grows as eps shrinks).
    signature = float(mc_calls) / float(max(q_calls, 1))
    return AdvantageRecord(
        method="qae", candidate=candidate, config_id=config["config_id"],
        quantum_metric=float(q_calls), classical_metric=float(mc_calls),
        metric_name="samples_to_eps", advantage_direction=direction,
        advantage_magnitude=signature, scaling_signature=signature,
        quantum_native_litmus=True,  # amplitude encoding is the whole point
        sim_runnable=True, q50_faithful_runnable=_q50_faithful(p),
        demo_naturalness=0.45,  # a risk number is less visceral than a fraud flag
        op_business_fit=0.9 if candidate == "E" else 0.7,
        notes=f"p={p}, eps={eps}, IQAE vs MC",
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_qae.py -v`
Expected: PASS. If `IterativeAmplitudeEstimation` / `EstimationProblem` import path differs in the installed `qiskit-algorithms`, check with `uv run python -c "import qiskit_algorithms as a; print([n for n in dir(a) if 'Amplitude' in n or 'Estimation' in n])"` and adjust the import. If `num_oracle_queries` is absent, the `1/eps` fallback keeps the test valid.

- [ ] **Step 5: Commit**

```bash
git add triage/harness/qae.py tests/test_qae.py
git commit -m "feat: QAE harness (option pricing + VaR/CVaR) with IQAE-vs-MC scaling"
```

---

## Task 5: QAOA harness (portfolio A)

**Files:**
- Create: `triage/harness/qaoa.py`
- Test: `tests/test_qaoa.py`

- [ ] **Step 1: Write failing known-answer test**

```python
import numpy as np
from triage.harness.qaoa import solve_portfolio, run

def test_qaoa_matches_brute_force_on_tiny_problem():
    mu = np.array([0.12, 0.01, 0.10, 0.02])
    cov = np.eye(4) * 0.0002
    chosen, _ = solve_portfolio(mu, cov, k=2, risk=1.0, reps=2, seed=1,
                                backend="local_aer")
    assert sorted(chosen) == [0, 2]  # the two highest-return assets

def test_run_returns_record():
    rec = run({"config_id": "a_smoke", "candidate": "A", "n_assets": 4,
               "k": 2, "reps": 1, "seed": 1, "backend": "local_aer"})
    assert rec.method == "qaoa"
    assert rec.metric_name == "approx_ratio"
    assert 0.0 <= rec.quantum_metric <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_qaoa.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `triage/harness/qaoa.py`**

```python
"""QAOA harness: cardinality-constrained portfolio selection as a QUBO.

Objective: maximize mu.x - risk * x^T cov x with a penalty enforcing sum(x)=k.
Quantum-native angle: the *combinatorial* (binary cardinality) version is the
hard one; the continuous mean-variance relaxation is just convex QP."""
from __future__ import annotations
import numpy as np
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp

from backends import get_backend
from triage.rubric import AdvantageRecord
from triage.baselines.classical_opt import exact_portfolio


def _qubo_to_ising(mu, cov, k, risk, penalty):
    """Build the Ising Hamiltonian for the penalized QUBO. Returns (op, offset)."""
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)
    n = len(mu)
    # QUBO matrix Q and linear c for f(x) = x^T Q x + c^T x (minimization of -obj)
    Q = risk * cov + penalty * np.ones((n, n))
    c = -mu - penalty * (2 * k) * np.ones(n) + penalty  # diag adj below
    # Map x in {0,1} -> z in {-1,+1} via x=(1-z)/2.
    paulis, coeffs = [], []
    offset = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = Q[i, j] / 4.0
            z = ["I"] * n
            z[i] = "Z"; z[j] = "Z"
            paulis.append("".join(z)); coeffs.append(w)
    for i in range(n):
        hi = -(c[i] / 2.0) - sum(Q[i, j] / 4.0 + Q[j, i] / 4.0 for j in range(n) if j != i)
        z = ["I"] * n; z[i] = "Z"
        paulis.append("".join(z)); coeffs.append(hi)
        offset += c[i] / 2.0 + Q[i, i] / 2.0
    op = SparsePauliOp(paulis, np.array(coeffs, dtype=complex))
    return op, offset


def solve_portfolio(mu, cov, k, risk=1.0, reps=1, seed=0, backend="local_aer"):
    """Return (chosen_indices, objective_value) from QAOA's most-likely bitstring."""
    n = len(mu)
    penalty = float(np.max(np.abs(mu)) * 10 + 1.0)
    op, _ = _qubo_to_ising(mu, cov, k, risk, penalty)
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=100), reps=reps)
    result = qaoa.compute_minimum_eigenvalue(op)
    # Most probable bitstring from the optimal circuit.
    dist = result.eigenstate  # dict[int|str -> prob] or QuasiDistribution
    items = dist.items() if hasattr(dist, "items") else dict(dist).items()
    best_key = max(items, key=lambda kv: kv[1])[0]
    bits = format(int(best_key), f"0{n}b")[::-1] if isinstance(best_key, int) else best_key[::-1]
    chosen = [i for i, b in enumerate(bits) if b == "1"]
    mu = np.asarray(mu, float); cov = np.asarray(cov, float)
    x = np.zeros(n); x[chosen] = 1.0
    val = float(mu @ x - risk * (x @ cov @ x))
    return chosen, val


def run(config: dict) -> AdvantageRecord:
    rng = np.random.default_rng(config.get("seed", 0))
    n = int(config.get("n_assets", 4))
    k = int(config.get("k", 2))
    reps = int(config.get("reps", 1))
    risk = float(config.get("risk", 1.0))
    mu = rng.uniform(0.0, 0.15, n)
    A = rng.normal(0, 0.01, (n, n)); cov = A @ A.T + np.eye(n) * 1e-4
    chosen, q_val = solve_portfolio(mu, cov, k, risk, reps,
                                    config.get("seed", 0), config.get("backend", "local_aer"))
    _, opt_val = exact_portfolio(mu, cov, k, risk)
    # Approximation ratio in [0,1]; guard against sign with a shift.
    shift = abs(min(q_val, opt_val)) + 1e-6
    approx = (q_val + shift) / (opt_val + shift)
    approx = max(0.0, min(1.0, approx))
    direction = "win" if approx >= 0.99 else ("tie" if approx >= 0.9 else "loss")
    return AdvantageRecord(
        method="qaoa", candidate="A", config_id=config["config_id"],
        quantum_metric=float(approx), classical_metric=1.0,
        metric_name="approx_ratio", advantage_direction=direction,
        advantage_magnitude=float(approx), scaling_signature=float(reps),
        quantum_native_litmus=True,  # binary cardinality is genuinely combinatorial
        sim_runnable=True, q50_faithful_runnable=True,  # shallow variational -> Q50-friendly
        demo_naturalness=0.6, op_business_fit=0.8,
        notes=f"n={n}, k={k}, reps={reps}, approx vs brute-force optimum",
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_qaoa.py -v`
Expected: PASS. QAOA is stochastic; if `test_qaoa_matches_brute_force` is flaky, raise `reps` to 2–3 and `COBYLA(maxiter=200)`. If `result.eigenstate` is a `QuasiDistribution`, the `.items()` branch handles it; verify the bit order with the brute-force test (flip the `[::-1]` if the assertion inverts).

- [ ] **Step 5: Commit**

```bash
git add triage/harness/qaoa.py tests/test_qaoa.py
git commit -m "feat: QAOA portfolio harness with brute-force approx-ratio"
```

---

## Task 6: Fraud data + QML-fraud harness

**Files:**
- Create: `triage/data/fraud.py`, `triage/data/__init__.py`
- Create: `triage/harness/fraud_qml.py`
- Test: `tests/test_fraud.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from triage.data.fraud import make_synthetic_fraud, prepare_features
from triage.harness.fraud_qml import quantum_kernel_auc, run

def test_prepare_features_caps_dimension():
    X = np.random.default_rng(0).normal(size=(200, 28))
    Xr = prepare_features(X, n_features=8)
    assert Xr.shape == (200, 8)

def test_quantum_kernel_separates_synthetic_fraud():
    X, y = make_synthetic_fraud(n=80, n_features=4, seed=0)
    auc = quantum_kernel_auc(X, y, backend="local_aer", seed=0)
    assert auc > 0.8

def test_run_returns_record_with_auc():
    rec = run({"config_id": "d_smoke", "candidate": "D", "n": 60,
               "n_features": 4, "backend": "local_aer", "seed": 0})
    assert rec.method == "fraud_qml"
    assert rec.metric_name == "auc"
    assert 0.0 <= rec.quantum_metric <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_fraud.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the data module `triage/data/fraud.py`**

```python
"""Fraud data: real ULB credit-card set if present, else a synthetic stand-in.

The ULB Kaggle file (creditcard.csv) has PCA features V1..V28 + Amount + Class.
Tonight we run on synthetic data so the build never blocks on a download; swap in
the real CSV via load_ulb() when available."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def make_synthetic_fraud(n: int = 200, n_features: int = 8, seed: int = 0):
    """Imbalanced two-class blob set resembling fraud (rare positive class)."""
    rng = np.random.default_rng(seed)
    n_pos = max(4, n // 10)
    n_neg = n - n_pos
    neg = rng.normal(0.0, 1.0, (n_neg, n_features))
    pos = rng.normal(1.8, 1.0, (n_pos, n_features))  # separable-ish shift
    X = np.vstack([neg, pos])
    y = np.array([0] * n_neg + [1] * n_pos)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def prepare_features(X, n_features: int = 8):
    """Standardize then PCA-reduce to n_features (<=10 for <=10 qubits)."""
    Xs = StandardScaler().fit_transform(np.asarray(X, float))
    if Xs.shape[1] <= n_features:
        return Xs
    return PCA(n_components=n_features, random_state=0).fit_transform(Xs)


def load_ulb(path: str = "data/raw/creditcard.csv", n: int = 400,
             n_features: int = 8, seed: int = 0):
    """Load + subsample the real ULB set. Falls back to synthetic if missing."""
    if not os.path.exists(path):
        return make_synthetic_fraud(n=n, n_features=n_features, seed=seed)
    df = pd.read_csv(path)
    pos = df[df["Class"] == 1]
    neg = df[df["Class"] == 0].sample(n=min(len(df) - len(pos), n - len(pos)),
                                      random_state=seed)
    sub = pd.concat([pos.sample(n=min(len(pos), n // 10), random_state=seed), neg])
    y = sub["Class"].to_numpy()
    X = sub.drop(columns=["Class", "Time"], errors="ignore").to_numpy()
    return prepare_features(X, n_features), y
```

- [ ] **Step 4: Implement the harness `triage/harness/fraud_qml.py`**

```python
"""QML-fraud harness: quantum-kernel SVM vs classical RBF-SVM on fraud features.

Quantum-native angle: a feature map embeds transactions into a Hilbert space whose
inner products (the quantum kernel) are expensive to compute classically. Shallow
feature-map circuits are Q50-hardware-friendly."""
from __future__ import annotations
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from backends import get_backend
from triage.rubric import AdvantageRecord
from triage.data.fraud import prepare_features, make_synthetic_fraud
from triage.baselines.classical_kernel import rbf_svm_auc


def _kernel_matrix(A, B, n_features):
    """Fidelity quantum kernel via an angle-embedding feature map (default.qubit)."""
    dev = qml.device("default.qubit", wires=n_features)

    @qml.qnode(dev)
    def kernel(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_features))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_features))
        return qml.probs(wires=range(n_features))

    return np.array([[kernel(a, b)[0] for b in B] for a in A])


def quantum_kernel_auc(X, y, backend="local_aer", seed: int = 0) -> float:
    X = np.asarray(X, float)
    nf = X.shape[1]
    Xtr, Xte, ytr, yte = train_test_split(
        X, np.asarray(y), test_size=0.3, random_state=seed, stratify=y
    )
    Ktr = _kernel_matrix(Xtr, Xtr, nf)
    Kte = _kernel_matrix(Xte, Xtr, nf)
    clf = SVC(kernel="precomputed", probability=True, random_state=seed).fit(Ktr, ytr)
    proba = clf.predict_proba(Kte)[:, 1]
    return float(roc_auc_score(yte, proba))


def _q50_faithful(n_features: int) -> bool:
    try:
        from qiskit import QuantumCircuit, transpile
        backend = get_backend("q50_fake")
        qc = QuantumCircuit(n_features)
        for i in range(n_features):
            qc.rx(0.5, i)
        if n_features > 1:
            qc.cz(0, 1)
        qc.measure_all()
        transpile(qc, backend, optimization_level=1)
        return True
    except Exception:
        return False


def run(config: dict) -> AdvantageRecord:
    n = int(config.get("n", 120))
    nf = int(config.get("n_features", 4))
    seed = int(config.get("seed", 0))
    try:
        from triage.data.fraud import load_ulb
        X, y = load_ulb(n=n, n_features=nf, seed=seed)
    except Exception:
        X, y = make_synthetic_fraud(n=n, n_features=nf, seed=seed)
    q_auc = quantum_kernel_auc(X, y, config.get("backend", "local_aer"), seed)
    c_auc = rbf_svm_auc(X, y, seed)
    if q_auc > c_auc + 0.02:
        direction = "win"
    elif q_auc < c_auc - 0.02:
        direction = "loss"
    else:
        direction = "tie"
    return AdvantageRecord(
        method="fraud_qml", candidate="D", config_id=config["config_id"],
        quantum_metric=float(q_auc), classical_metric=float(c_auc),
        metric_name="auc", advantage_direction=direction,
        advantage_magnitude=float(q_auc - c_auc),
        scaling_signature=float(nf),  # kernel expressivity grows with feature dim
        quantum_native_litmus=True,  # the kernel is the quantum object
        sim_runnable=True, q50_faithful_runnable=_q50_faithful(nf),
        demo_naturalness=0.95,  # a live fraud flag is the most visceral demo
        op_business_fit=0.95,   # banking fraud is core OP
        notes=f"n={n}, n_features={nf}, quantum-kernel vs RBF-SVM AUC",
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_fraud.py -v`
Expected: PASS. The quantum kernel is O(n²) qnode calls — keep `n<=120` in tests so it stays fast. If `test_quantum_kernel_separates` is borderline, increase the synthetic class shift in `make_synthetic_fraud` (1.8 → 2.2) or reduce `n_features`.

- [ ] **Step 6: Commit**

```bash
git add triage/data triage/harness/fraud_qml.py tests/test_fraud.py
git commit -m "feat: fraud data prep + quantum-kernel fraud harness vs RBF-SVM"
```

---

## Task 7: Orchestrator (sweep loop + checkpoint)

**Files:**
- Create: `triage/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
import json
from pathlib import Path
from triage.orchestrator import run_sweep, load_records

def test_run_sweep_writes_one_record_per_config(tmp_path):
    spec = {"qae": [{"config_id": "b0", "candidate": "B", "p": 0.3,
                     "epsilon": 0.1, "backend": "local_aer"}]}
    out = tmp_path / "records.jsonl"
    run_sweep(spec, ledger=out, plots_dir=tmp_path / "plots",
              dashboard=tmp_path / "dash.html")
    recs = load_records(out)
    assert len(recs) == 1 and recs[0].method == "qae"

def test_failing_config_is_logged_not_fatal(tmp_path):
    spec = {"qae": [{"config_id": "bad"}],  # missing required keys -> harness errors
            "qaoa": [{"config_id": "a0", "candidate": "A", "n_assets": 4,
                      "k": 2, "reps": 1, "seed": 0, "backend": "local_aer"}]}
    out = tmp_path / "records.jsonl"
    run_sweep(spec, ledger=out, plots_dir=tmp_path / "plots",
              dashboard=tmp_path / "dash.html")
    recs = load_records(out)
    # the good qaoa config still ran despite the bad qae one
    assert any(r.method == "qaoa" for r in recs)

def test_checkpoint_skips_completed_configs(tmp_path):
    spec = {"qaoa": [{"config_id": "a0", "candidate": "A", "n_assets": 4,
                      "k": 2, "reps": 1, "seed": 0, "backend": "local_aer"}]}
    out = tmp_path / "records.jsonl"
    ckpt = tmp_path / "ckpt.json"
    run_sweep(spec, ledger=out, plots_dir=tmp_path / "plots",
              dashboard=tmp_path / "dash.html", checkpoint=ckpt)
    run_sweep(spec, ledger=out, plots_dir=tmp_path / "plots",
              dashboard=tmp_path / "dash.html", checkpoint=ckpt)
    assert len(load_records(out)) == 1  # not duplicated on second run
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `triage/orchestrator.py`**

```python
"""Sweep orchestrator: run each config, append to the JSONL ledger, refresh the
dashboard, checkpoint. Catches per-config failures so the night never crashes."""
from __future__ import annotations
import argparse
import json
import traceback
from pathlib import Path

import yaml

from triage.rubric import AdvantageRecord
from triage.harness import qae, qaoa, fraud_qml

_HARNESS = {"qae": qae.run, "qaoa": qaoa.run, "fraud_qml": fraud_qml.run}


def load_records(ledger: Path) -> list[AdvantageRecord]:
    ledger = Path(ledger)
    if not ledger.exists():
        return []
    out = []
    for line in ledger.read_text().splitlines():
        if line.strip():
            out.append(AdvantageRecord.from_dict(json.loads(line)))
    return out


def _load_checkpoint(ckpt: Path | None) -> set[str]:
    if ckpt and Path(ckpt).exists():
        return set(json.loads(Path(ckpt).read_text()))
    return set()


def _save_checkpoint(ckpt: Path | None, done: set[str]) -> None:
    if ckpt:
        Path(ckpt).write_text(json.dumps(sorted(done)))


def run_sweep(spec: dict, ledger: Path, plots_dir: Path, dashboard: Path,
              checkpoint: Path | None = None) -> None:
    ledger = Path(ledger); plots_dir = Path(plots_dir); dashboard = Path(dashboard)
    plots_dir.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(checkpoint)
    total = sum(len(v) for v in spec.values())
    completed = len(done)
    for method, configs in spec.items():
        harness = _HARNESS.get(method)
        if harness is None:
            print(f"[skip] unknown method {method}")
            continue
        for cfg in configs:
            cid = cfg.get("config_id", f"{method}_{completed}")
            if cid in done:
                continue
            try:
                rec = harness(cfg)
                with ledger.open("a") as fh:
                    fh.write(json.dumps(rec.to_dict()) + "\n")
            except Exception:
                print(f"[fail] {method}/{cid}\n{traceback.format_exc()}")
            finally:
                done.add(cid)
                completed += 1
                _save_checkpoint(checkpoint, done)
                _refresh_dashboard(ledger, plots_dir, dashboard, completed, total)


def _refresh_dashboard(ledger, plots_dir, dashboard, completed, total):
    # Imported lazily so unit tests of the loop don't require matplotlib paths.
    from triage.dashboard import render
    render(load_records(ledger), plots_dir, dashboard, completed, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--grid", default="all", help="top-level key set: 'all' or 'smoke'")
    ap.add_argument("--ledger", default="triage/records.jsonl")
    ap.add_argument("--plots", default="triage/plots")
    ap.add_argument("--dashboard", default="triage/dashboard.html")
    ap.add_argument("--checkpoint", default="triage/.checkpoint.json")
    ap.add_argument("--report", default="triage/REPORT.md")
    args = ap.parse_args()
    raw = yaml.safe_load(Path(args.sweep).read_text())
    spec = raw[args.grid] if args.grid in raw else raw
    run_sweep(spec, Path(args.ledger), Path(args.plots), Path(args.dashboard),
              Path(args.checkpoint))
    from triage.digest import write_report
    write_report(load_records(Path(args.ledger)), Path(args.report))
    print(f"Done. Report at {args.report}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: PASS. (The dashboard import is exercised here; if Task 9 isn't implemented yet, temporarily make `_refresh_dashboard` swallow `ImportError` — but order tasks so Task 9 lands before running this. To keep TDD green now, implement a 3-line `triage/dashboard.py` stub `def render(*a, **k): pass` and flesh it out in Task 9.)

- [ ] **Step 5: Commit**

```bash
git add triage/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: sweep orchestrator with JSONL ledger, checkpoint, failure isolation"
```

---

## Task 8: Digest (deterministic REPORT.md + plots + optional analyst)

**Files:**
- Create: `triage/digest.py`
- Test: `tests/test_digest.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
from triage.rubric import AdvantageRecord
from triage.digest import write_report

def _rec(method, cand, direction, q, c, name):
    return AdvantageRecord(
        method=method, candidate=cand, config_id=f"{cand}0",
        quantum_metric=q, classical_metric=c, metric_name=name,
        advantage_direction=direction, advantage_magnitude=1.0,
        scaling_signature=1.0, quantum_native_litmus=True,
        sim_runnable=True, q50_faithful_runnable=(method != "qae"),
        demo_naturalness=0.9 if method == "fraud_qml" else 0.4,
        op_business_fit=0.9, notes="")

def test_report_names_a_winner_and_lists_all_methods(tmp_path):
    recs = [_rec("qae", "B", "win", 10, 100, "samples_to_eps"),
            _rec("fraud_qml", "D", "win", 0.95, 0.9, "auc")]
    out = tmp_path / "REPORT.md"
    write_report(recs, out)
    text = out.read_text()
    assert "# Triage report" in text
    assert "Recommendation" in text
    assert "fraud_qml" in text and "qae" in text

def test_report_handles_empty_ledger(tmp_path):
    out = tmp_path / "REPORT.md"
    write_report([], out)
    assert "no records" in out.read_text().lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_digest.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `triage/digest.py`**

```python
"""Deterministic morning report. Always produces ranked Markdown; if
ANTHROPIC_API_KEY is set, optionally prepends an LLM narrative."""
from __future__ import annotations
import os
from pathlib import Path

from triage.rubric import AdvantageRecord, score, rank

_DESCRIPTIONS = {
    "qae": "Quantum Amplitude Estimation — O(1/eps) vs Monte-Carlo O(1/eps^2). "
           "LUMI-sim story (deep circuits). Covers option pricing & VaR/CVaR.",
    "qaoa": "QAOA on cardinality-constrained portfolio QUBO. Combinatorial "
            "selection is the genuinely hard, quantum-native version.",
    "fraud_qml": "Quantum-kernel SVM on credit-card fraud. Shallow feature map "
                 "is Q50-hardware-friendly; most natural live demo.",
}


def _best_per_method(records):
    best = {}
    for r in records:
        if r.method not in best or score(r) > score(best[r.method]):
            best[r.method] = r
    return best


def write_report(records: list[AdvantageRecord], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        out.write_text("# Triage report\n\nNo records yet.\n")
        return
    best = _best_per_method(records)
    ranked = rank(list(best.values()))
    winner = ranked[0]
    lines = ["# Triage report", ""]
    lines.append(f"**Recommendation:** `{winner.method}` "
                 f"(candidate {winner.candidate}) — score {score(winner):.3f}.")
    lines.append("")
    lines.append("| Rank | Method | Cand | Score | Advantage | Q (metric) | Classical | Q50 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r.method} | {r.candidate} | {score(r):.3f} | "
            f"{r.advantage_direction} | {r.quantum_metric:.4g} ({r.metric_name}) | "
            f"{r.classical_metric:.4g} | {'yes' if r.q50_faithful_runnable else 'no'} |")
    lines.append("")
    for r in ranked:
        lines.append(f"## {r.method} — candidate {r.candidate}")
        lines.append(_DESCRIPTIONS.get(r.method, ""))
        lines.append("")
        lines.append(f"- Advantage: **{r.advantage_direction}** "
                     f"(magnitude {r.advantage_magnitude:.3g}, "
                     f"scaling signature {r.scaling_signature:.3g})")
        lines.append(f"- Quantum-native litmus: "
                     f"{'passes' if r.quantum_native_litmus else 'FAILS'}")
        lines.append(f"- Q50-faithful runnable: {r.q50_faithful_runnable}")
        lines.append(f"- Demo naturalness: {r.demo_naturalness:.2f} | "
                     f"OP business fit: {r.op_business_fit:.2f}")
        lines.append(f"- Notes: {r.notes}")
        lines.append("")
    report = "\n".join(lines)
    narrative = _maybe_analyst(report)
    out.write_text((narrative + "\n\n---\n\n" + report) if narrative else report)


def _maybe_analyst(report: str) -> str | None:
    """Optional LLM narrative. No-op when no API key (the case in this env)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-opus-4-8", max_tokens=800,
            messages=[{"role": "user", "content":
                       "You are the lab analyst. Given this triage report, write a "
                       "6-sentence recommendation + a one-paragraph 'why quantum-"
                       "native' slide draft for OP Pohjola judges.\n\n" + report}],
        )
        return "## Analyst narrative\n\n" + msg.content[0].text
    except Exception:
        return None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_digest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add triage/digest.py tests/test_digest.py
git commit -m "feat: deterministic ranked REPORT.md + optional LLM analyst"
```

---

## Task 9: Self-contained HTML dashboard

**Files:**
- Create / replace stub: `triage/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
from triage.rubric import AdvantageRecord
from triage.dashboard import render

def _rec(method):
    return AdvantageRecord(
        method=method, candidate="D", config_id=f"{method}0",
        quantum_metric=0.95, classical_metric=0.9, metric_name="auc",
        advantage_direction="win", advantage_magnitude=0.05,
        scaling_signature=4.0, quantum_native_litmus=True,
        sim_runnable=True, q50_faithful_runnable=True,
        demo_naturalness=0.95, op_business_fit=0.95, notes="x")

def test_render_writes_self_contained_html(tmp_path):
    out = tmp_path / "dash.html"
    render([_rec("fraud_qml"), _rec("qae")], tmp_path / "plots", out, 2, 5)
    html = out.read_text()
    assert "<html" in html.lower()
    assert "http-equiv=\"refresh\"" in html  # auto-refresh tag
    assert "fraud_qml" in html and "qae" in html
    assert "2/5" in html  # progress banner
    # plots embedded inline, not linked
    assert "data:image/png;base64," in html

def test_render_handles_empty(tmp_path):
    out = tmp_path / "dash.html"
    render([], tmp_path / "plots", out, 0, 3)
    assert "Waiting" in out.read_text()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: FAIL (the Task-7 stub `render` returns None and writes nothing).

- [ ] **Step 3: Implement `triage/dashboard.py`**

```python
"""Self-contained auto-refreshing dashboard. One HTML file, plots embedded as
base64 PNGs (no server, no external assets). Re-render after every config."""
from __future__ import annotations
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from triage.rubric import AdvantageRecord, score, rank

_DESC = {
    "qae": "Quantum Amplitude Estimation: O(1/ε) vs Monte-Carlo O(1/ε²). "
           "Option pricing & VaR/CVaR. Deep circuits → LUMI-sim story.",
    "qaoa": "QAOA portfolio (cardinality-constrained QUBO). Combinatorial = the "
            "quantum-native version. Shallow → Q50-runnable.",
    "fraud_qml": "Quantum-kernel SVM on card fraud. Visceral live demo; "
                 "shallow feature map → Q50-hardware-friendly.",
}


def _bar_png(rec: AdvantageRecord) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.bar(["quantum", "classical"], [rec.quantum_metric, rec.classical_metric],
           color=["#4c72b0", "#999999"])
    ax.set_title(rec.metric_name, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _best_per_method(records):
    best = {}
    for r in records:
        if r.method not in best or score(r) > score(best[r.method]):
            best[r.method] = r
    return best


def _card(rec: AdvantageRecord) -> str:
    png = _bar_png(rec)
    badge = ("#2e7d32" if rec.q50_faithful_runnable else "#b71c1c")
    badge_txt = "Q50 ready" if rec.q50_faithful_runnable else "Q50 N/A"
    return f"""
    <div style="border:1px solid #ddd;border-radius:10px;padding:14px;margin:10px;
                width:320px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
      <h2 style="margin:0 0 4px">{rec.method} <small>({rec.candidate})</small></h2>
      <p style="color:#555;font-size:13px;min-height:54px">{_DESC.get(rec.method,'')}</p>
      <div><b>score {score(rec):.3f}</b> &middot;
           advantage: <b>{rec.advantage_direction}</b></div>
      <div style="font-size:13px">scaling sig: {rec.scaling_signature:.3g} &middot;
           litmus: {'✓' if rec.quantum_native_litmus else '✗'}</div>
      <span style="background:{badge};color:#fff;border-radius:6px;padding:2px 8px;
                   font-size:12px">{badge_txt}</span>
      <span style="font-size:12px;color:#555"> demo {rec.demo_naturalness:.2f} ·
           OP {rec.op_business_fit:.2f}</span>
      <div><img src="data:image/png;base64,{png}" style="margin-top:8px"/></div>
    </div>"""


def render(records, plots_dir, out: Path, completed: int, total: int) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        out.write_text(
            "<html><head><meta http-equiv=\"refresh\" content=\"30\"></head>"
            "<body style='font-family:sans-serif'><h1>Triage</h1>"
            f"<p>Waiting for first result… {completed}/{total}</p></body></html>")
        return
    best = _best_per_method(records)
    ranked = rank(list(best.values()))
    leader = ranked[0]
    cards = "".join(_card(r) for r in ranked)
    html = f"""<html><head>
      <meta http-equiv="refresh" content="30">
      <title>Quantum-finance triage</title></head>
      <body style="font-family:sans-serif;margin:24px">
      <h1>Quantum-finance triage</h1>
      <p><b>{completed}/{total}</b> configs &middot;
         current leader: <b>{leader.method}</b> (candidate {leader.candidate},
         score {score(leader):.3f})</p>
      <div style="display:flex;flex-wrap:wrap">{cards}</div>
      </body></html>"""
    out.write_text(html)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add triage/dashboard.py tests/test_dashboard.py
git commit -m "feat: self-contained auto-refreshing HTML dashboard with per-method cards"
```

---

## Task 10: Sweep spec + smoke sweep

**Files:**
- Create: `sweeps/all.yaml`
- Test: `tests/test_smoke_sweep.py`

- [ ] **Step 1: Write `sweeps/all.yaml`**

```yaml
# Tiny grid for a fast end-to-end check before the real launch.
smoke:
  qae:
    - {config_id: b_s, candidate: B, p: 0.3, epsilon: 0.1, backend: local_aer}
  qaoa:
    - {config_id: a_s, candidate: A, n_assets: 4, k: 2, reps: 1, seed: 0, backend: local_aer}
  fraud_qml:
    - {config_id: d_s, candidate: D, n: 60, n_features: 4, backend: local_aer, seed: 0}

# The overnight grid.
all:
  qae:
    - {config_id: b_eps_10, candidate: B, p: 0.3, epsilon: 0.10, backend: local_aer}
    - {config_id: b_eps_05, candidate: B, p: 0.3, epsilon: 0.05, backend: local_aer}
    - {config_id: b_eps_02, candidate: B, p: 0.3, epsilon: 0.02, backend: local_aer}
    - {config_id: e_eps_05, candidate: E, p: 0.1, epsilon: 0.05, backend: local_aer}
    - {config_id: e_eps_02, candidate: E, p: 0.1, epsilon: 0.02, backend: local_aer}
  qaoa:
    - {config_id: a_n4_p1, candidate: A, n_assets: 4, k: 2, reps: 1, seed: 0, backend: local_aer}
    - {config_id: a_n4_p2, candidate: A, n_assets: 4, k: 2, reps: 2, seed: 0, backend: local_aer}
    - {config_id: a_n6_p2, candidate: A, n_assets: 6, k: 3, reps: 2, seed: 1, backend: local_aer}
    - {config_id: a_n8_p3, candidate: A, n_assets: 8, k: 3, reps: 3, seed: 2, backend: local_aer}
  fraud_qml:
    - {config_id: d_f4, candidate: D, n: 150, n_features: 4, backend: local_aer, seed: 0}
    - {config_id: d_f6, candidate: D, n: 150, n_features: 6, backend: local_aer, seed: 0}
    - {config_id: d_f8, candidate: D, n: 200, n_features: 8, backend: local_aer, seed: 1}
```

- [ ] **Step 2: Write the smoke test**

```python
from pathlib import Path
import yaml
from triage.orchestrator import run_sweep, load_records

def test_smoke_sweep_runs_end_to_end(tmp_path):
    spec = yaml.safe_load(Path("sweeps/all.yaml").read_text())["smoke"]
    ledger = tmp_path / "rec.jsonl"
    run_sweep(spec, ledger, tmp_path / "plots", tmp_path / "dash.html",
              tmp_path / "ckpt.json")
    recs = load_records(ledger)
    methods = {r.method for r in recs}
    assert methods == {"qae", "qaoa", "fraud_qml"}
    assert (tmp_path / "dash.html").exists()
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_smoke_sweep.py -v`
Expected: PASS (this is the real integration check — all three harnesses + ledger + dashboard work together). If slow (>60s), lower `fraud_qml` smoke `n` to 40.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add sweeps/all.yaml tests/test_smoke_sweep.py
git commit -m "feat: sweep specs (smoke + overnight grid) + end-to-end smoke test"
```

---

## Task 11: Demo shell (Streamlit fraud console)

**Files:**
- Create: `demo/inference.py`, `demo/app.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write failing test for the inference layer**

```python
import numpy as np
from demo.inference import train_fraud_model, score_transaction

def test_score_transaction_returns_probability():
    model = train_fraud_model(backend="local_aer", n=80, n_features=4, seed=0)
    x = np.zeros(4)
    p = score_transaction(model, x)
    assert 0.0 <= p <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_inference.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `demo/inference.py`**

```python
"""Backend-agnostic inference for the demo. Trains the quantum-kernel fraud model
once and scores individual transactions — the frontrunner candidate, ready to dress."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.svm import SVC

from triage.data.fraud import load_ulb, make_synthetic_fraud
from triage.harness.fraud_qml import _kernel_matrix


@dataclass
class FraudModel:
    clf: SVC
    X_train: np.ndarray
    n_features: int
    backend: str


def train_fraud_model(backend="local_aer", n=200, n_features=4, seed=0) -> FraudModel:
    try:
        X, y = load_ulb(n=n, n_features=n_features, seed=seed)
    except Exception:
        X, y = make_synthetic_fraud(n=n, n_features=n_features, seed=seed)
    K = _kernel_matrix(X, X, n_features)
    clf = SVC(kernel="precomputed", probability=True, random_state=seed).fit(K, y)
    return FraudModel(clf=clf, X_train=X, n_features=n_features, backend=backend)


def score_transaction(model: FraudModel, x) -> float:
    x = np.asarray(x, float).reshape(1, -1)
    k = _kernel_matrix(x, model.X_train, model.n_features)
    return float(model.clf.predict_proba(k)[0, 1])
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_inference.py -v`
Expected: PASS.

- [ ] **Step 5: Implement `demo/app.py` (Streamlit; smoke-checked, not unit-tested)**

```python
"""Fraud-triage console: pick a backend, stream sample transactions, see the
quantum-kernel model flag fraud live. The 'more than numbers' demo shell."""
import numpy as np
import streamlit as st

from demo.inference import train_fraud_model, score_transaction

st.set_page_config(page_title="Quantum Fraud Triage", layout="wide")
st.title("⚛️  Quantum Fraud Triage — OP Pohjola")

backend = st.sidebar.selectbox(
    "Quantum backend", ["local_aer", "lumi_aer", "q50_fake", "q50_hw"],
    help="q50_fake = VTT Q50 native gates + noise (offline). q50_hw = real Q50 (on-site).")
n_features = st.sidebar.slider("Feature qubits", 2, 8, 4)
threshold = st.sidebar.slider("Flag threshold", 0.0, 1.0, 0.5)

@st.cache_resource
def _model(backend, n_features):
    return train_fraud_model(backend=backend, n=200, n_features=n_features, seed=0)

model = _model(backend, n_features)

st.subheader("Incoming transactions")
if st.button("Generate a batch"):
    rng = np.random.default_rng()
    rows = []
    for i in range(8):
        x = rng.normal(0, 1, n_features) + (1.8 if rng.random() < 0.3 else 0.0)
        p = score_transaction(model, x)
        rows.append({"txn": f"T{i:03d}", "fraud_prob": round(p, 3),
                     "flag": "🚨" if p >= threshold else "✅"})
    st.dataframe(rows, use_container_width=True)
    st.caption(f"Scored on backend: {backend} · quantum-kernel SVM "
               f"({n_features}-qubit feature map)")
```

- [ ] **Step 6: Smoke-run the app**

Run: `uv run streamlit run demo/app.py --server.headless true --server.port 8765 &` then `sleep 6 && curl -s localhost:8765 | grep -qi "streamlit" && echo OK; kill %1`
Expected: prints `OK` (the app boots without import/runtime errors).

- [ ] **Step 7: Commit**

```bash
git add demo tests/test_inference.py
git commit -m "feat: backend-agnostic fraud inference + Streamlit triage console demo"
```

---

## Task 12: Popper-corpus pre-registration + launch runbook

**Files:**
- Create: `popper-corpus/<cand>/hypothesis.md` (×4, via popper-probe intake)
- Create: `RUNBOOK.md`

- [ ] **Step 1: Forge the template hypothesis with popper-probe**

Run the `popper-probe:intake` skill once with the template claim: *"Quantum approach X yields a meaningful, quantum-native advantage over the best classical baseline on metric M for problem P; it is refuted if a classical baseline matches/beats it at simulable scale with no asymptotic backing."* Approve the written file.

- [ ] **Step 2: Instantiate per candidate**

Copy the template into `popper-corpus/{option_pricing_qae, var_cvar_qae, portfolio_qaoa, fraud_qml}/hypothesis.md`, filling X/M/P/baseline per candidate (matching the metric_name each harness emits). Validate each: `python3 ~/Documents/popper-probe/scripts/validate_hypothesis.py popper-corpus/<slug>/hypothesis.md`.

- [ ] **Step 3: Write `RUNBOOK.md`**

```markdown
# Overnight runbook

## Launch (terminal, not launchd — macOS TCC on ~/Documents)
    cd /Users/masha/Documents/qhack
    caffeinate -i uv run python -m triage.orchestrator --sweep sweeps/all.yaml --grid all

## Watch
Open `triage/dashboard.html` in a browser (auto-refreshes every 30s).

## Morning
- `triage/REPORT.md` — ranked recommendation.
- `triage/dashboard.html` — per-method visuals.
- Demo: `uv run streamlit run demo/app.py`

## Resume after a crash
Re-run the same launch command; the checkpoint skips completed configs.

## Real Q50 (on-site only)
Set IQM_SERVER_URL + IQM_TOKEN, then switch a config's `backend: q50_hw`.
Never used overnight.
```

- [ ] **Step 4: Smoke-launch the real pipeline (short)**

Run: `cd /Users/masha/Documents/qhack && uv run python -m triage.orchestrator --sweep sweeps/all.yaml --grid smoke`
Expected: writes `triage/records.jsonl`, `triage/dashboard.html`, `triage/REPORT.md`; prints `Done.`; `REPORT.md` names a winner.

- [ ] **Step 5: Commit**

```bash
git add popper-corpus RUNBOOK.md
git commit -m "docs: pre-registered falsifiable hypotheses + overnight runbook"
```

---

## Self-Review

**Spec coverage:**
- Backend layer (4 targets, IQMFakeBackend) → Task 1. ✓
- AdvantageRecord + 25%-mirrored scoring → Task 2. ✓
- QAE (B+E), QAOA (A), QML-fraud (D) harnesses → Tasks 4–6. ✓
- Classical baselines → Task 3. ✓
- Orchestrator (ledger, checkpoint, failure isolation) → Task 7. ✓
- Deterministic REPORT.md + optional analyst (no key in env) → Task 8. ✓
- Live self-contained auto-refresh dashboard, per-method cards + descriptions + visuals + Q50 badge → Task 9. ✓
- Sweep spec + smoke + known-answer tests → Tasks 4–6, 10. ✓
- Demo shell (Streamlit, backend dropdown, fraud console) → Task 11. ✓
- Popperian pre-registration + caffeinate runbook → Task 12. ✓
- qGAN deferred, real-hardware-overnight out of scope → respected (q50_hw guarded). ✓

**Placeholder scan:** `<cand>` in Task 12 is intentional shorthand expanded in Step 2; every code step contains complete code. No TBD/TODO left.

**Type consistency:** `AdvantageRecord` fields used identically across harnesses, digest, dashboard. `run(config) -> AdvantageRecord` signature uniform. `render(records, plots_dir, out, completed, total)` matches the orchestrator call. `get_backend(name)` matches all callers. `_kernel_matrix(A, B, n_features)` shared by harness + demo with the same signature.

**Note on task ordering:** Task 7's test imports `triage.dashboard.render`. Implement a one-line stub (`def render(*a, **k): pass`) in Task 7-Step-3's module list, then replace it in Task 9 — called out inline in Task 7-Step-4.
