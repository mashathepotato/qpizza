# Pricing-Race Frontend — Design Spec

**Date:** 2026-06-06
**Project:** qpizza — "The Madness of People Is Quantum" (Junction Helsinki × OP Pohjola)
**Author:** brainstorm session (Masha + Claude)

## 1. Purpose

A clean, animated frontend for the hackathon judging table that shows our real Nokia
option-pricing result as a **pricing-accuracy race**: real stock data loads into a quantum
superposition of all price paths, then every pricing model estimates the option price at once
against the ground-truth fair price — with the quantum routes shown winning in the two honest
senses (free/exact path loading, and quadratically fewer queries to reach a target accuracy).

It is the demo artifact, not new science: every moving element is backed by a number we already
compute. It doubles as the fallback video.

## 2. The honest framing (non-negotiable — this is a physics-literate judging panel)

There are **two distinct wins**, kept visually separate so no conflation is possible:

- **Win 1 — exact, free path loading (novelty).** `M` single-qubit `R_y` rotations load all `2^M`
  CRR price paths into one superposition; the Fourier/exact route reads the price off the
  statevector exactly (error ~`6e-11`). No expensive distribution-loading oracle — which is the
  bulk of the textbook SOTA (Qiskit-Finance oracle-QAE) circuit. Visual: all paths load at once,
  price snaps to truth.
- **Win 2 — quadratically fewer queries (scaling).** On an x-axis of **oracle queries / samples
  spent (never wall-clock seconds)**: Monte Carlo error ~ `1/√N` (log-log slope −½, i.e. `1/ε²`);
  QAE error ~ `1/queries` (slope −1, i.e. `1/ε`). So QAE reaches a target accuracy with
  quadratically fewer queries (Montanaro 2015).

**Caveats baked into the UI (all already enforced in `quantum_pricer/benchmark.py`):**
1. The race axis is labelled **"oracle queries / samples"**, never "time." On current hardware
   quantum is slower in wall-clock; the win is error-per-resource-spent.
2. The QAE lead appears **only at tight accuracy**; at coarse ε the fixed per-Grover-round cost
   lets classical win. The guided race runs to tight accuracy; the cockpit (mode A) shows the
   crossover honestly.
3. **QSVT plateaus** at its polynomial-approximation floor (~1–2% at degree 60) — its line
   flattens, never reaches zero. Honest, not a bug.
4. Ground truth for every error = **exact CRR tree price** (`tree.exact_tree_price`), not
   Black–Scholes. Headline ground truth in the demo instance: **€2.7752** (Nokia, M=3 reference;
   the convergence series uses a larger M where the QAE descent is clean).
5. SOTA comparison uses the real numbers: ours abs-err **0.0003** vs Qiskit oracle-QAE **0.057**.

The guardrail (§4 of `context/VISION.md`) stands: quantum-*like* markets are the motivation; the
pricer is real quantum computing. No claim about quantum brains.

## 3. Layout & interaction

**Primary mode B — guided story.** Five full-screen acts, advanced by space-bar / arrow keys
(also auto-advance option for the fallback video). Cinematic, hard to fumble.

**Toggle mode A — explore cockpit.** Pressing **E** at any time drops into an all-panels-at-once
dashboard (stock+tree · race leaderboard · convergence plot · resource/qubit table · Q50 status)
so a judge can poke any number live. Pressing E again returns to the story.

### The five acts
1. **The madness of people** — real NOKIA.HE path draws in, jagged. Newton ("madness of people")
   + Feynman (classical intuition fails) hook → markets are non-classical.
2. **Load every path at once** (Win 1) — the chart fans into the `2^M` CRR tree; all paths light
   up, then collapse into one superposition cloud. Caption: `M` rotations load `2^M` paths, free
   and exact, no loading oracle.
3. **Read the price without collapsing it — the race** — all models estimate the option price
   simultaneously against the ground-truth line (€2.7752). Exact/Fourier snaps to truth; Monte
   Carlo creeps up; SOTA oracle-QAE lags (expensive loading). Live query/sample counters.
4. **Quadratic speedup** (Win 2) — the error-vs-**queries** log-log plot draws itself from real
   points: QAE slope −1 vs MC −½; QSVT plateaus at its floor. Annotation: quadratically fewer
   queries at tight accuracy.
5. **Beat the textbook + run on real hardware** — ours err 0.0003 vs SOTA 0.057; a **Q50 hardware**
   badge that reads "pending" until the teammate's real counts land, then lights up with the result.

## 4. Architecture — components

Each unit has one purpose, a defined interface, and is independently testable.

### 4.1 Data exporters (Python)
`results/export_demo_data.py` (or co-located with `quantum_pricer/make_results.py`).
- **Produces:**
  - `convergence.json` — from `benchmark.error_vs_queries_rms(...)`: rows
    `{method, budget_x (queries/samples), rms_error, note}` for MC and QAE, with the honest
    saturation/floor flags preserved.
  - `prices.json` — the real NOKIA.HE price path for act 1 (from the yfinance window already
    recorded in `results.json` meta; a small standalone file).
  - `hardware.json` — placeholder `{"status":"pending"}` the teammate overwrites with Q50 counts.
  - Reuses existing `results/results.json` for final prices, queries, errors, SOTA row, ground truth.
- **Depends on:** `quantum_pricer/benchmark.py`, `tree.py`, existing `results.json`.
- **Testable:** JSON schema assertions; assert fitted QAE slope ≈ −1 and MC slope ≈ −½ (within
  tolerance) on the exported `convergence.json`; assert ground-truth and SOTA numbers match
  `results.json`.

### 4.2 Animation frontend (single self-contained HTML)
`results/demo_animation.html` — vanilla JS + inline SVG/canvas, no framework, no bundler.
- **Behaviour:** loads the JSONs at startup (relative paths, works from `file://`); runs the 5
  acts; E-toggles into the cockpit. All animation driven by the real JSON values.
- **Degradation:** fully functional standalone with the prebaked JSON. If the live server is
  absent, the "re-run live" buttons are silently hidden.
- **Testable:** opens with zero console errors; each act renders from fixture JSON; toggling E
  shows/hides the cockpit; with `hardware.json` = pending the badge reads "pending".

### 4.3 Optional live server (small Flask)
`results/live_server.py`.
- **Adds exactly one capability:** a "⟳ re-run live" button per route that re-runs that route
  (small M) and streams the fresh result back to the frontend. Additive only.
- **Depends on:** the pricer route modules (`qae.py`, `fourier.py`, `qsvt.py`, `classical.py`).
- **Testable:** frontend renders and runs identically with the server down (buttons absent);
  with the server up, a re-run returns a price within tolerance of the prebaked value.

## 5. Data flow

```
quantum_pricer/{benchmark,tree,...}.py
        │  export_demo_data.py
        ▼
results/{results.json, convergence.json, prices.json, hardware.json}
        │  (static fetch, relative paths)
        ▼
results/demo_animation.html  ──(optional)──►  results/live_server.py  ──► pricer routes
   (5 acts + E→cockpit)         re-run live        fresh price
```

## 6. Error handling / robustness

- Missing/malformed JSON → frontend shows a small inline "data not exported yet — run
  `export_demo_data.py`" notice in that panel, never a blank/broken screen.
- `hardware.json` pending or absent → Q50 badge reads "pending", rest of demo unaffected.
- Live server unreachable → re-run buttons hidden; prebaked animation unchanged.
- Conference wifi / hardware failure → demo runs entirely from `file://` + prebaked JSON.

## 7. Testing strategy

- **Exporters:** pytest in `quantum_pricer/tests/` — schema + slope assertions on
  `convergence.json`; consistency of ground-truth/SOTA numbers with `results.json`.
- **Frontend:** a lightweight check that the HTML loads fixtures with no console errors and each
  act renders (manual + a smoke check); verify graceful degradation with server down and with
  `hardware.json` pending.
- **Server:** test that a re-run returns a price within tolerance; test the frontend is unchanged
  when the server is absent.

## 8. Out of scope (YAGNI)

- No JS framework / bundler / build step.
- No real-time market-data feed (the Nokia path is the recorded window).
- No trading-decision / buy-sell agent (we price options; we do not trade).
- No auth, no deployment infra beyond a `file://` open or `python live_server.py`.

## 9. Success criteria

- Opens from `file://` with zero console errors and plays all 5 acts on real data.
- E-toggle into the cockpit works; a judge can read any number live.
- Every moving element traces to a real exported number; the three honesty caveats are visible.
- Degrades gracefully with the live server down and with Q50 hardware pending.
- Doubles as a clean 2-minute screen-recorded fallback video.
