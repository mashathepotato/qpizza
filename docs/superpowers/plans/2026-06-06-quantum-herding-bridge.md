# Quantum Herding bridge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tunable "herding" entangler `J` to the binomial path loader, producing a non-product price measure whose fat tails reprice the European option (`J=0` = classical), and surface it in the results dashboard + a narrative bridge.

**Architecture:** One new module `quantum_pricer/herding.py` that swaps the *probabilities* into the existing pricing path (reusing `tree.payoff_variable_values`). A `CRY(J)` nearest-neighbour ladder correlates consecutive moves; a multiplicative forward-recenter keeps it arbitrage-free; `J` is calibrated to market kurtosis. A demo driver produces a `J`-sweep panel embedded into `results/dashboard.html`.

**Tech Stack:** Python 3.12, Qiskit 1.2.4 (`QuantumCircuit.cry`, `qiskit.quantum_info.Statevector`), NumPy, SciPy, Matplotlib, pytest. Run everything with the pricer venv: `quantum_pricer/.venv/bin/python`.

**Reference:** `docs/superpowers/specs/2026-06-06-quantum-herding-bridge-design.md`

**Conventions to know (from `quantum_pricer/tree.py`):**
- `crr_params(S0,K,r,sigma,T,M) -> (u,d,q)`; `loading_angles(...) -> [theta]*M` (constant per step).
- `payoff_variable_values(...,option) -> np.array len 2**M` of `f(x)`: terminal `S_T` (european) or path-average (asian), indexed by integer path `x` with **qubit 0 = LSB**.
- `path_probabilities(...) -> np.array len 2**M` of the product measure `p(x)`.
- `exact_tree_price(...,option,kind) -> float` discounted expected payoff (the ground truth).
- `Statevector(qc).probabilities()` is indexed the same way (qubit 0 = LSB) — so it lines up with `payoff_variable_values` directly.

---

### Task 1: Herded state + herded probabilities (with `J=0` regression)

**Files:**
- Create: `quantum_pricer/herding.py`
- Test: `quantum_pricer/tests/test_herding.py`

- [ ] **Step 1: Write the failing test**

Create `quantum_pricer/tests/test_herding.py`:

```python
import numpy as np
from quantum_pricer import herding, tree

PARAMS = dict(S0=4.20, K=4.20, r=0.03, sigma=0.30, T=1.0, M=3)


def test_herded_probs_J0_equals_product_measure():
    p_herded = herding.herded_path_probabilities(J=0.0, **PARAMS)
    p_tree = tree.path_probabilities(**PARAMS)
    assert p_herded.shape == p_tree.shape == (2 ** PARAMS["M"],)
    np.testing.assert_allclose(p_herded, p_tree, atol=1e-9)


def test_herded_probs_normalised_and_changed_when_J_positive():
    p = herding.herded_path_probabilities(J=0.6, **PARAMS)
    assert abs(p.sum() - 1.0) < 1e-9
    # J>0 must actually change the measure (else the entangler is inert)
    assert not np.allclose(p, tree.path_probabilities(**PARAMS), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantum_pricer.herding'`.

- [ ] **Step 3: Write minimal implementation**

Create `quantum_pricer/herding.py`:

```python
"""Herding (non-product) path measure: a tunable controlled-RY entangler on top
of the CRR path loader. J=0 recovers the classical independent product measure
exactly; J>0 correlates consecutive moves (momentum/herding) -> fatter terminal
tails -> the European option reprices.

Spec: docs/superpowers/specs/2026-06-06-quantum-herding-bridge-design.md
NOTE: the entangler MUST be non-diagonal. RZZ(J) is diagonal in the computational
basis -> it only adds phases and leaves p(x) unchanged. Controlled-RY does change
the joint distribution (momentum), which is the whole point.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_pricer import tree


def herded_state(angles, J):
    """Circuit: R_y(theta_i) loading + a controlled-RY(J) nearest-neighbour ladder.
    J=0 -> the CRY gates are identities -> the exact product loader."""
    M = len(angles)
    qc = QuantumCircuit(M)
    for i, th in enumerate(angles):
        qc.ry(th, i)
    for i in range(M - 1):
        qc.cry(J, i, i + 1)   # if step i is up, nudge step i+1's up-amplitude up
    return qc


def herded_path_probabilities(S0, K, r, sigma, T, M, J):
    """Length-2**M array p(x) of the herded measure (qubit 0 = LSB, matches tree)."""
    angles = tree.loading_angles(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    qc = herded_state(angles, J)
    return np.asarray(Statevector(qc).probabilities())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add quantum_pricer/herding.py quantum_pricer/tests/test_herding.py
git commit -m "feat(herding): herded state + non-product path probabilities (J=0 = classical)"
```

---

### Task 2: Forward recenter (arbitrage-free martingale correction)

**Files:**
- Modify: `quantum_pricer/herding.py`
- Test: `quantum_pricer/tests/test_herding.py`

- [ ] **Step 1: Write the failing test**

Append to `quantum_pricer/tests/test_herding.py`:

```python
def test_recenter_restores_forward():
    M = PARAMS["M"]
    p = herding.herded_path_probabilities(J=0.6, **PARAMS)
    ST = tree.payoff_variable_values(option="european", **PARAMS)
    forward = herding.forward_price(PARAMS["S0"], PARAMS["r"], PARAMS["T"])
    c = herding.recenter_factor(p, ST, forward)
    assert abs(float(np.sum(p * (ST * c))) - forward) < 1e-9


def test_recenter_factor_is_one_at_J0():
    # the CRR product measure is already risk-neutral: E[S_T] = forward, so c == 1
    p = herding.herded_path_probabilities(J=0.0, **PARAMS)
    ST = tree.payoff_variable_values(option="european", **PARAMS)
    forward = herding.forward_price(PARAMS["S0"], PARAMS["r"], PARAMS["T"])
    assert abs(herding.recenter_factor(p, ST, forward) - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -k recenter -v`
Expected: FAIL with `AttributeError: module 'quantum_pricer.herding' has no attribute 'forward_price'`.

- [ ] **Step 3: Write minimal implementation**

Append to `quantum_pricer/herding.py`:

```python
def forward_price(S0, r, T):
    """Risk-neutral forward F = S0 * e^{rT} (the martingale target for E[S_T])."""
    return float(S0 * np.exp(r * T))


def recenter_factor(probs, terminal_ST, forward):
    """Multiplicative martingale correction c so that E[c * S_T] = forward.
    Scaling all prices by a constant preserves log-return dispersion (and hence
    the herding-induced fat tails) while restoring the arbitrage-free mean."""
    mean_ST = float(np.sum(probs * terminal_ST))
    return forward / mean_ST
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -k recenter -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add quantum_pricer/herding.py quantum_pricer/tests/test_herding.py
git commit -m "feat(herding): forward recenter (arbitrage-free martingale correction)"
```

---

### Task 3: `price_herded` for the European call

**Files:**
- Modify: `quantum_pricer/herding.py`
- Test: `quantum_pricer/tests/test_herding.py`

- [ ] **Step 1: Write the failing test**

Append to `quantum_pricer/tests/test_herding.py`:

```python
def test_price_herded_J0_equals_exact_tree():
    got = herding.price_herded(J=0.0, option="european", kind="call", **PARAMS)
    want = tree.exact_tree_price(option="european", kind="call", **PARAMS)
    assert abs(got - want) < 1e-9


def test_herding_raises_otm_call_value():
    # fat tails (J>0), mean held at forward by recenter -> OTM call worth MORE
    otm = dict(PARAMS); otm["K"] = round(PARAMS["S0"] * 1.10, 4)
    p0 = herding.price_herded(J=0.0, option="european", kind="call", **otm)
    pJ = herding.price_herded(J=0.6, option="european", kind="call", **otm)
    assert pJ > p0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -k price_herded -v`
Expected: FAIL with `AttributeError: ... has no attribute 'price_herded'`.

- [ ] **Step 3: Write minimal implementation**

Append to `quantum_pricer/herding.py`:

```python
def price_herded(S0, K, r, sigma, T, M, J, option="european", kind="call",
                 recenter=True):
    """Discounted E[payoff] under the herded measure, by direct statevector
    expectation (exact at small M). recenter=True applies the forward correction
    so the price is arbitrage-free; at J=0 the correction is a no-op and this
    equals tree.exact_tree_price."""
    probs = herded_path_probabilities(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, J=J)
    terminal_ST = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T,
                                              M=M, option="european")
    values = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                         option=option)
    if recenter:
        c = recenter_factor(probs, terminal_ST, forward_price(S0, r, T))
        values = values * c   # scales terminal price (and the average, linearly)
    if kind == "call":
        payoff = np.maximum(values - K, 0.0)
    elif kind == "put":
        payoff = np.maximum(K - values, 0.0)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return float(np.exp(-r * T) * np.sum(probs * payoff))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -k price_herded -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add quantum_pricer/herding.py quantum_pricer/tests/test_herding.py
git commit -m "feat(herding): price_herded for European call (J=0 == exact tree; fat tails raise OTM)"
```

---

### Task 4: Kurtosis calibration to market data

**Files:**
- Modify: `quantum_pricer/herding.py`
- Test: `quantum_pricer/tests/test_herding.py`

- [ ] **Step 1: Write the failing test**

Append to `quantum_pricer/tests/test_herding.py`:

```python
def test_herding_increases_terminal_kurtosis():
    k0 = herding.terminal_logreturn_excess_kurtosis(J=0.0, **PARAMS)
    kJ = herding.terminal_logreturn_excess_kurtosis(J=0.6, **PARAMS)
    assert kJ > k0   # herding fattens the tails


def test_calibrate_J_picks_grid_point_matching_target():
    rng = np.random.default_rng(0)
    # heavy-tailed synthetic daily returns (Student-t-like) -> positive excess kurtosis
    returns = rng.standard_t(df=4, size=2000) * 0.02
    out = herding.calibrate_J_to_kurtosis(returns, J_grid=np.linspace(0, 1.2, 13),
                                          **PARAMS)
    assert set(out) >= {"J", "target_excess_kurtosis", "model_excess_kurtosis"}
    assert 0.0 <= out["J"] <= 1.2
    # the chosen J is the grid point closest in kurtosis to the target
    assert abs(out["model_excess_kurtosis"] - out["target_excess_kurtosis"]) < \
        abs(herding.terminal_logreturn_excess_kurtosis(J=0.0, **PARAMS)
            - out["target_excess_kurtosis"]) + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -k "kurtosis or calibrate" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'terminal_logreturn_excess_kurtosis'`.

- [ ] **Step 3: Write minimal implementation**

Add the SciPy import near the top of `quantum_pricer/herding.py` (below the existing imports):

```python
from scipy.stats import kurtosis as _sample_excess_kurtosis  # fisher=True -> excess
```

Append to `quantum_pricer/herding.py`:

```python
def terminal_logreturn_excess_kurtosis(S0, K, r, sigma, T, M, J):
    """Excess kurtosis of log(S_T/S0) under the herded measure (probability-weighted)."""
    probs = herded_path_probabilities(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, J=J)
    ST = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                     option="european")
    x = np.log(ST / S0)
    mu = float(np.sum(probs * x))
    var = float(np.sum(probs * (x - mu) ** 2))
    if var <= 0:
        return 0.0
    m4 = float(np.sum(probs * (x - mu) ** 4))
    return m4 / var ** 2 - 3.0


def calibrate_J_to_kurtosis(daily_log_returns, S0, K, r, sigma, T, M, J_grid=None):
    """Pick J in J_grid whose model terminal-log-return excess kurtosis best matches
    the market's realised excess kurtosis (from daily returns). J is anchored to
    market data -- NOT fitted to the option, NOT derived from the cognition model."""
    if J_grid is None:
        J_grid = np.linspace(0.0, 1.2, 25)
    target = float(_sample_excess_kurtosis(np.asarray(daily_log_returns),
                                           fisher=True, bias=False))
    best_J, best_err, best_k = 0.0, np.inf, 0.0
    for J in J_grid:
        k = terminal_logreturn_excess_kurtosis(S0=S0, K=K, r=r, sigma=sigma,
                                               T=T, M=M, J=float(J))
        err = abs(k - target)
        if err < best_err:
            best_J, best_err, best_k = float(J), err, k
    return dict(J=best_J, target_excess_kurtosis=target, model_excess_kurtosis=best_k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -k "kurtosis or calibrate" -v`
Expected: PASS (2 passed).

> If `test_herding_increases_terminal_kurtosis` FAILS (kurtosis not increasing with J), the entangler sign/form is wrong for momentum — flip to `qc.cry(-J, ...)` or widen the J range, re-run, and confirm before proceeding. Do NOT weaken the assertion to hide a non-effect.

- [ ] **Step 5: Commit**

```bash
git add quantum_pricer/herding.py quantum_pricer/tests/test_herding.py
git commit -m "feat(herding): terminal-kurtosis helper + calibrate_J_to_kurtosis (market anchor)"
```

---

### Task 5: Asian support (stretch) + full-suite regression

**Files:**
- Test: `quantum_pricer/tests/test_herding.py`

- [ ] **Step 1: Write the failing test**

Append to `quantum_pricer/tests/test_herding.py`:

```python
def test_price_herded_asian_J0_equals_exact_tree():
    got = herding.price_herded(J=0.0, option="asian", kind="call", **PARAMS)
    want = tree.exact_tree_price(option="asian", kind="call", **PARAMS)
    assert abs(got - want) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -k asian -v`
Expected: PASS immediately (Task 3's `price_herded` already accepts `option="asian"` and recenters the average via the same factor `c`). This task confirms Asian works end-to-end; if it FAILS, the recenter is mis-scaling the average — verify `values = values * c` applies to the asian array too.

> **Spec §6.4 note (order-sensitivity):** the spec lists "Asian fractional move > European" as a validation item. It is an *empirical* claim about the model, not guaranteed, so it is **surfaced as a dashboard observation** — the §6 herding table (Task 7) shows European *and* Asian price vs `J` side by side — rather than a hard unit test that could falsely fail. Do not assert it as a test unless you first confirm it holds for the calibrated `J`.

- [ ] **Step 3: Run the whole herding suite**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding.py -v`
Expected: PASS (all tests from Tasks 1–5).

- [ ] **Step 4: Commit**

```bash
git add quantum_pricer/tests/test_herding.py
git commit -m "test(herding): Asian J=0 regression + full-suite green"
```

---

### Task 6: `J`-sweep demo driver (data + plot)

**Files:**
- Create: `quantum_pricer/herding_demo.py`
- Test: `quantum_pricer/tests/test_herding_demo.py`

- [ ] **Step 1: Write the failing test**

Create `quantum_pricer/tests/test_herding_demo.py`:

```python
import os
from quantum_pricer import herding_demo


def test_sweep_has_classical_anchor_at_J0(tmp_path):
    out = herding_demo.run_sweep(S0=4.20, K=4.20, r=0.03, sigma=0.30, T=1.0, M=3,
                                 daily_log_returns=None, n_J=9)
    rows = out["rows"]
    assert len(rows) == 9
    j0 = [row for row in rows if row["J"] == 0.0][0]
    # J=0 European price equals the classical tree price (the dashboard anchor)
    assert abs(j0["price_eu"] - out["classical_price_eu"]) < 1e-9
    assert out["rows"][-1]["J"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding_demo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantum_pricer.herding_demo'`.

- [ ] **Step 3: Write minimal implementation**

Create `quantum_pricer/herding_demo.py`:

```python
"""J-sweep demo: European (and Asian) price + terminal kurtosis as the herding
knob J rises. J=0 reproduces the classical tree price. Saves results/herding.json
and results/herding.png for the dashboard. Run: python -m quantum_pricer.herding_demo
"""
import json
import os

import numpy as np

from quantum_pricer import herding, tree

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(os.path.dirname(_HERE), "results")


def run_sweep(S0, K, r, sigma, T, M, daily_log_returns=None, n_J=13, J_max=1.2):
    classical_eu = tree.exact_tree_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                         option="european", kind="call")
    classical_as = tree.exact_tree_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                         option="asian", kind="call")
    J_grid = np.linspace(0.0, J_max, n_J)
    rows = []
    for J in J_grid:
        rows.append(dict(
            J=float(J),
            price_eu=herding.price_herded(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                          J=float(J), option="european", kind="call"),
            price_as=herding.price_herded(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                          J=float(J), option="asian", kind="call"),
            excess_kurtosis=herding.terminal_logreturn_excess_kurtosis(
                S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, J=float(J))))
    calib = None
    if daily_log_returns is not None and len(daily_log_returns) > 30:
        calib = herding.calibrate_J_to_kurtosis(daily_log_returns, S0=S0, K=K, r=r,
                                                sigma=sigma, T=T, M=M, J_grid=J_grid)
    return dict(rows=rows, classical_price_eu=classical_eu,
                classical_price_as=classical_as, calibrated=calib,
                params=dict(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M))


def save_plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["rows"]
    J = [row["J"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(J, [row["price_eu"] for row in rows], "o-", color="#7c5cff",
             label="European price")
    ax1.axhline(out["classical_price_eu"], ls="--", color="#9fb0d4",
                label="classical (J=0)")
    if out.get("calibrated"):
        ax1.axvline(out["calibrated"]["J"], ls=":", color="#23d5c8",
                    label=f"kurtosis-calibrated J={out['calibrated']['J']:.2f}")
    ax1.set_xlabel("herding strength J"); ax1.set_ylabel("European call price (EUR)")
    ax2 = ax1.twinx()
    ax2.plot(J, [row["excess_kurtosis"] for row in rows], "s-", color="#ffb454",
             alpha=0.6, label="excess kurtosis")
    ax2.set_ylabel("terminal excess kurtosis")
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    os.makedirs(_RESULTS, exist_ok=True)
    from quantum_pricer.data import nokia_params
    params, meta = nokia_params(allow_network=True)
    S0, sigma, r = params["S0"], params["sigma"], params["r"]
    K = round(S0, 2)
    returns = None
    if meta.get("source") == "yfinance":
        try:
            import yfinance as yf
            hist = yf.Ticker(meta["ticker"]).history(period="1y")["Close"].dropna()
            returns = np.diff(np.log(hist.values))
        except Exception:
            returns = None
    out = run_sweep(S0=S0, K=K, r=r, sigma=sigma, T=1.0, M=3,
                    daily_log_returns=returns)
    out["data_source"] = meta.get("source", "synthetic")
    with open(os.path.join(_RESULTS, "herding.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    save_plot(out, os.path.join(_RESULTS, "herding.png"))
    print("wrote results/herding.json + results/herding.png")
    print("classical EU=%.6f; J-sweep EU[last]=%.6f; calibrated=%s"
          % (out["classical_price_eu"], out["rows"][-1]["price_eu"], out["calibrated"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes, then run the driver**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding_demo.py -v`
Expected: PASS (1 passed).
Run: `quantum_pricer/.venv/bin/python -m quantum_pricer.herding_demo`
Expected: prints "wrote results/herding.json + results/herding.png" and the summary line; both files exist.

- [ ] **Step 5: Commit**

```bash
git add quantum_pricer/herding_demo.py quantum_pricer/tests/test_herding_demo.py results/herding.json results/herding.png
git commit -m "feat(herding): J-sweep demo driver (price + kurtosis vs J, calibrated J) with plot"
```

---

### Task 7: Embed the herding panel into the labeled dashboard

**Files:**
- Modify: `quantum_pricer/make_results.py` (function `compute` and `build_html`)
- Test: `quantum_pricer/tests/test_herding_dashboard.py`

- [ ] **Step 1: Write the failing test**

Create `quantum_pricer/tests/test_herding_dashboard.py`:

```python
import json
import os

from quantum_pricer import make_results


def test_dashboard_has_herding_section(tmp_path):
    # build the herding sweep artifact the dashboard reads
    from quantum_pricer import herding_demo
    out = herding_demo.run_sweep(S0=4.20, K=4.20, r=0.03, sigma=0.30, T=1.0, M=3,
                                 daily_log_returns=None, n_J=7)
    html = make_results.herding_section_html(out, png_b64="")
    assert "Quantum Herding" in html
    assert "J=0" in html and "classical" in html.lower()
    assert "modeling choice" in html.lower()  # honesty note: not a quantum advantage
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding_dashboard.py -v`
Expected: FAIL with `AttributeError: module 'quantum_pricer.make_results' has no attribute 'herding_section_html'`.

- [ ] **Step 3: Write minimal implementation**

Add to `quantum_pricer/make_results.py` (a standalone function near `build_html`):

```python
def herding_section_html(sweep, png_b64):
    """Render the Quantum Herding panel (price-vs-J + honesty). `sweep` is the dict
    from herding_demo.run_sweep; png_b64 is the base64 herding.png (may be empty)."""
    rows = sweep["rows"]
    cls = sweep["classical_price_eu"]
    calib = sweep.get("calibrated")
    H = ["<h2>6 · Quantum Herding — the non-classical market model (NEW)</h2>"]
    H.append("<div class='callout'><b>What this adds vs the benchmark above.</b> The benchmark "
             "compares <i>algorithms</i> that all price the SAME classical (independent) measure. "
             "Herding changes the <i>measure itself</i>: a controlled-RY(J) entangler correlates "
             "consecutive moves (the crowd's interference) → fatter tails → the European reprices. "
             "<b>J=0 reproduces the classical price above.</b> "
             "<b>Honesty:</b> herding is a <b>modeling choice</b> — at this M it is classically "
             "simulable, so it is NOT itself a quantum advantage; the speed-up remains QAE's "
             "query-complexity (see §5). J is calibrated to market kurtosis, not the option, not "
             "the cognition model.</div>")
    if png_b64:
        H.append(f"<div class='card'><img src='data:image/png;base64,{png_b64}' "
                 "alt='price vs J'></div>")
    H.append("<div class='card'><table><thead><tr>"
             "<th>Herding J<span class='dir'>0 = classical</span></th>"
             "<th>European price<span class='dir'>EUR, SIMULATED</span></th>"
             "<th>Δ vs classical<span class='dir'>EUR</span></th>"
             "<th>Asian price<span class='dir'>EUR, SIMULATED</span></th>"
             "<th>Excess kurtosis<span class='dir'>tail fatness ↑ with J</span></th>"
             "</tr></thead><tbody>")
    for row in rows:
        tag = " (classical anchor)" if row["J"] == 0.0 else ""
        H.append("<tr>"
                 f"<td class='num'>{row['J']:.3f}{tag}</td>"
                 f"<td class='num'>{row['price_eu']:.6f}</td>"
                 f"<td class='num'>{row['price_eu']-cls:+.6f}</td>"
                 f"<td class='num'>{row['price_as']:.6f}</td>"
                 f"<td class='num'>{row['excess_kurtosis']:+.4f}</td></tr>")
    H.append("</tbody></table>")
    if calib:
        H.append(f"<p class='note'>Kurtosis-calibrated <b>J={calib['J']:.3f}</b> "
                 f"(market excess kurtosis {calib['target_excess_kurtosis']:.3f}, "
                 f"model {calib['model_excess_kurtosis']:.3f}). "
                 f"{_badge('MARKET')} anchor.</p>")
    H.append("<p class='note'>All prices " + _badge("SIMULATED") +
             " (statevector, noiseless); J=0 column is the " + _badge("REFERENCE") +
             " classical tree price.</p></div>")
    return "".join(H)
```

Then wire it into `build_html`: locate the line that appends the honesty-notes section header (`"<h2>6 · Honesty notes`) and insert the herding section **before** it, so numbering becomes 1–5 (benchmark), **6 (herding), 7 (honesty)**. Change the honesty header text from `"6 · Honesty notes"` to `"7 · Honesty notes"`, then insert this block immediately before that honesty header line:

```python
    # ── herding panel (if the sweep artifact exists) ───────────────────────────
    import base64 as _b64
    _hjson = os.path.join(_RESULTS, "herding.json")
    if os.path.exists(_hjson):
        with open(_hjson) as fh:
            _sweep = json.load(fh)
        _hpng = os.path.join(_RESULTS, "herding.png")
        _b = ""
        if os.path.exists(_hpng):
            with open(_hpng, "rb") as fh:
                _b = _b64.b64encode(fh.read()).decode("ascii")
        H.append(herding_section_html(_sweep, _b))
```

(Place that block immediately before the `H.append("<h2>... Honesty notes` line in `build_html`.)

- [ ] **Step 4: Run test, then regenerate the dashboard**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_herding_dashboard.py -v`
Expected: PASS (1 passed).
Run: `quantum_pricer/.venv/bin/python -m quantum_pricer.herding_demo && quantum_pricer/.venv/bin/python -m quantum_pricer.make_results`
Expected: dashboard regenerates; `grep -c "Quantum Herding" results/dashboard.html` returns ≥ 1.

- [ ] **Step 5: Commit**

```bash
git add quantum_pricer/make_results.py quantum_pricer/tests/test_herding_dashboard.py results/dashboard.html results/results.json
git commit -m "feat(herding): embed Quantum Herding price-vs-J panel into the labeled dashboard"
```

---

### Task 8: Narrative bridge in the one-pager + VISION

**Files:**
- Modify: `context/quantum-investor.html`
- Modify: `context/VISION.md`

> **Coordination note:** a parallel session edits these files. **Re-read both files immediately before editing** and place the additions without disturbing other in-flight changes. If a section anchor below is gone, append the content at the end of the relevant section instead.

- [ ] **Step 1: Add the bridge card to the one-pager**

Re-read `context/quantum-investor.html`. Find the `<!-- THE DEMO -->`/`What we actually build` section (the build section). Immediately after its closing content (before the next `<h2>`), insert:

```html
  <div class="card" style="border-color:rgba(124,92,255,.5)">
    <h3>The bridge: the crowd's interference → a non-classical price</h3>
    <p class="muted">The classical pricer assumes price moves are <b>independent</b>
    (a product measure). But the "madness of people" is exactly <b>non-independence</b> —
    the crowd interferes and herds. We add a tunable <b>herding entangler J</b>
    (a controlled-R<sub>y</sub> ladder on the path qubits): J=0 is the classical tree,
    J&gt;0 correlates consecutive moves → <b>fatter tails</b> → the European option reprices
    (a volatility-smile effect). J is calibrated to the stock's realised kurtosis.</p>
    <p class="muted" style="font-size:13px"><b>Honest:</b> herding is the <i>model</i>
    (classically simulable at this scale) — the quantum <i>speed-up</i> is QAE's query
    complexity, separate. We do not derive J from the cognition model; it is motivated by it
    and calibrated to market data.</p>
  </div>
```

- [ ] **Step 2: Add a section to VISION.md**

Re-read `context/VISION.md`. After §5 ("What we are building"), insert:

```markdown
## 5b. The bridge — herding makes the market non-classical
The classical pricer loads an **independent product** measure. The cognition motivation says
crowds are **not** independent — they interfere/herd. We add a tunable **herding entangler `J`**
(controlled-`R_y` ladder on the path qubits) → a **non-product** measure: `J=0` = classical tree,
`J>0` correlates moves → **fat tails** → the European reprices (volatility-smile effect). `J` is
**calibrated to the stock's realised kurtosis** (market anchor), recentered to the forward for
arbitrage-freeness. **Honest:** herding is the *model* (classically simulable at demo scale) — the
quantum *speed-up* is QAE's query complexity, a separate axis; we do **not** derive `J` from the
cognition model. Code: `quantum_pricer/herding.py`; demo panel in `results/dashboard.html`.
```

- [ ] **Step 3: Verify the additions render / are present**

Run: `grep -c "herding entangler" context/quantum-investor.html context/VISION.md`
Expected: each file returns ≥ 1.

- [ ] **Step 4: Commit**

```bash
git add context/quantum-investor.html context/VISION.md
git commit -m "docs(bridge): narrative — herding makes the market non-classical (cognition <-> pricer)"
```

---

## Final verification

- [ ] Run the full pricer + herding suite:

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests -v`
Expected: all pass (existing pricer tests + new herding tests).

- [ ] Regenerate artifacts end to end:

Run: `quantum_pricer/.venv/bin/python -m quantum_pricer.herding_demo && quantum_pricer/.venv/bin/python -m quantum_pricer.make_results`
Expected: `results/dashboard.html` contains the "Quantum Herding" section with the price-vs-J table and the honesty note.

- [ ] Push (use the autostash-rebase pattern if the parallel session pushed first):

```bash
git push origin main || { git rebase --autostash origin/main && git push origin main; }
```
