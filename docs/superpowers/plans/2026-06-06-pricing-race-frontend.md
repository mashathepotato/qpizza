# Pricing-Race Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an animated, self-contained HTML frontend that stages our real Nokia option-pricing result as a pricing-accuracy race (5 guided acts + an explore cockpit), driven entirely by real numbers exported from the existing `quantum_pricer` code, with an optional additive Flask "re-run live" server.

**Architecture:** Three layers. (1) Python **data exporters** turn existing `quantum_pricer` computations into four small JSON files (`prices.json`, `convergence.json`, `hardware.json`, plus the existing `results.json`). (2) A single **self-contained HTML** (`results/demo_animation.html`, vanilla JS + inline SVG) fetches those JSONs and renders the acts; it works from `file://` with no server. (3) An **optional Flask server** (`results/live_server.py`) adds "re-run live" buttons that the frontend silently hides when the server is absent.

**Tech Stack:** Python 3.12 (venv at `quantum_pricer/.venv`), Qiskit/qiskit-algorithms (already wired), numpy, pytest; Flask (new dep) for the optional server; vanilla JS + SVG for the frontend (no framework, no bundler).

**Conventions:**
- Run all tests with the project venv from the repo root: `quantum_pricer/.venv/bin/python -m pytest <path> -v`.
- Ground truth for every error is the **exact CRR tree price** (`tree.exact_tree_price`), never Black–Scholes.
- The race/convergence axis is **always** labelled "oracle queries / samples", never wall-clock time.
- Honesty caveats (QAE lead only at tight accuracy; QSVT plateau; queries-not-time) must appear in UI captions.

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `quantum_pricer/data.py` | add `nokia_price_series()` — the real price *path* for act 1 (the existing `nokia_params` discards it) | Modify |
| `quantum_pricer/tests/test_data.py` | test for `nokia_price_series` | Modify |
| `results/export_demo_data.py` | pure builders (`build_prices`, `build_convergence`, `hardware_placeholder`) + `main()` writer | Create |
| `results/test_export_demo.py` | tests for the exporters | Create |
| `results/demo_animation.html` | the self-contained 5-act + cockpit frontend | Create |
| `results/test_demo_animation.py` | structural smoke tests on the HTML | Create |
| `results/live_server.py` | optional Flask re-run server | Create |
| `results/test_live_server.py` | Flask test-client tests | Create |
| `quantum_pricer/requirements.txt` | add `flask` | Modify |
| `results/README.md` | document how to run the demo | Modify |

Two M values on purpose (per spec §2.4): **M=3** for the visually-clean 8-path superposition + race instance, **M=5** for the convergence series where the QAE descent is genuinely visible.

---

## Task 1: Real Nokia price *path* exporter in `data.py`

The existing `nokia_params` returns only `{S0, sigma, r}` and throws away `hist.values`. Act 1 needs the actual closing-price series. Add a sibling function that returns the path, with a labelled synthetic GBM fallback.

**Files:**
- Modify: `quantum_pricer/data.py`
- Test: `quantum_pricer/tests/test_data.py`

- [ ] **Step 1: Write the failing test**

Append to `quantum_pricer/tests/test_data.py`:

```python
def test_nokia_price_series_offline_is_labelled_synthetic_and_well_formed():
    from quantum_pricer import data
    series, meta = data.nokia_price_series(allow_network=False, n_synth=120, seed=0)
    # provenance is explicit
    assert meta["source"] == "synthetic"
    # series is internally consistent and plottable
    assert len(series["dates"]) == len(series["closes"]) == 120
    assert all(c > 0 for c in series["closes"])
    assert series["S0"] == series["closes"][-1]
    assert series["sigma"] > 0
    # deterministic given the seed
    series2, _ = data.nokia_price_series(allow_network=False, n_synth=120, seed=0)
    assert series2["closes"] == series["closes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_data.py::test_nokia_price_series_offline_is_labelled_synthetic_and_well_formed -v`
Expected: FAIL with `AttributeError: module 'quantum_pricer.data' has no attribute 'nokia_price_series'`

- [ ] **Step 3: Write minimal implementation**

Add to `quantum_pricer/data.py` (after `nokia_params`):

```python
def _synthetic_series(n, r, seed):
    """Labelled GBM stand-in path from the synthetic params (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    S0, sigma = _SYNTHETIC["S0"], _SYNTHETIC["sigma"]
    dt = 1.0 / 252
    shocks = rng.normal((r - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), size=n - 1)
    closes = [S0]
    for s in shocks:
        closes.append(closes[-1] * float(np.exp(s)))
    closes = [float(c) for c in closes]
    logret = np.diff(np.log(closes))
    return dict(dates=["d%03d" % i for i in range(n)], closes=closes,
                S0=closes[-1], sigma=annualized_vol(logret), r=r)


def nokia_price_series(ticker="NOKIA.HE", lookback="1y", r=DEFAULT_R,
                       allow_network=True, n_synth=252, seed=0):
    """Return (series, meta). series = {dates, closes, S0, sigma, r} — the real daily
    closing path for act 1. Falls back to a LABELLED synthetic GBM path offline."""
    if allow_network:
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period=lookback)["Close"].dropna()
            if len(hist) > 30:
                closes = [float(x) for x in hist.values]
                sigma = annualized_vol(np.diff(np.log(hist.values)))
                series = dict(dates=[str(d.date()) for d in hist.index], closes=closes,
                              S0=closes[-1], sigma=sigma, r=r)
                meta = dict(source="yfinance", ticker=ticker, lookback=lookback,
                            n_obs=len(closes), start=str(hist.index[0].date()),
                            end=str(hist.index[-1].date()))
                return series, meta
        except Exception as exc:  # offline / rate-limited / delisted
            return _synthetic_series(n_synth, r, seed), dict(source="synthetic", reason=str(exc))
    return _synthetic_series(n_synth, r, seed), dict(source="synthetic", reason="network disabled")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest quantum_pricer/tests/test_data.py::test_nokia_price_series_offline_is_labelled_synthetic_and_well_formed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add quantum_pricer/data.py quantum_pricer/tests/test_data.py
git commit -m "feat(data): export real Nokia price path for the demo frontend"
```

---

## Task 2: Convergence-series builder

Turn `benchmark.error_vs_queries_rms` (rows `{method, budget_x, rms_error, n_seeds, note}`) into the JSON the act-4 log-log plot draws from, with fitted slopes and the honesty notes preserved.

**Files:**
- Create: `results/export_demo_data.py`
- Test: `results/test_export_demo.py`

- [ ] **Step 1: Write the failing test**

Create `results/test_export_demo.py`:

```python
from results import export_demo_data as ex

# small/fast params; M=4, seeds=2 keeps the QAE runs quick but still descending
FAST = dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)


def test_build_convergence_has_real_descending_series_and_slopes():
    conv = ex.build_convergence(M=4, seeds=2, **FAST)
    assert conv["ground_truth"] > 0
    assert conv["axis_label"].lower().startswith("oracle queries")
    for method in ("classical_mc", "qae"):
        pts = conv["series"][method]
        assert len(pts) >= 2
        assert all(p["x"] > 0 and p["y"] > 0 for p in pts)
        assert pts == sorted(pts, key=lambda p: p["x"])  # sorted by budget
    # classical MC log-log slope ~ -1/2 (error ~ 1/sqrt(N))
    assert -0.75 <= conv["slopes"]["classical_mc"] <= -0.25
    # QAE descends faster than MC (slope more negative) OR is flagged as saturated-theory
    qae_saturated = any("saturat" in n for n in conv["notes"]["qae"])
    assert qae_saturated or conv["slopes"]["qae"] < conv["slopes"]["classical_mc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_export_demo.py::test_build_convergence_has_real_descending_series_and_slopes -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'results.export_demo_data'`

- [ ] **Step 3: Write minimal implementation**

Create `results/export_demo_data.py`:

```python
"""Export the real numbers the demo_animation.html frontend renders.

Every moving element in the frontend traces to a value produced here:
  prices.json       -- the real Nokia price path (act 1) + the 2^M tree (act 2)
  convergence.json  -- seed-averaged error-vs-QUERIES descent (act 4): MC slope ~-1/2,
                       QAE slope ~-1; honesty notes preserved. Ground truth = exact tree.
  hardware.json     -- a 'pending' Q50 placeholder a teammate overwrites with real counts.
results/results.json (already produced by make_results.py) supplies the final prices/queries.

Run:  python -m results.export_demo_data
"""
import json
import os

import numpy as np

from quantum_pricer import benchmark, tree
from quantum_pricer.data import nokia_price_series

_HERE = os.path.dirname(os.path.abspath(__file__))


def build_convergence(S0, K, r, sigma, T, M=5, seeds=6):
    """error_vs_queries_rms -> {ground_truth, M, axis_label, series, slopes, notes}."""
    rows = benchmark.error_vs_queries_rms(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                          seeds=seeds)
    gt = tree.exact_tree_price(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    series, notes, slopes = {}, {}, {}
    for method in ("classical_mc", "qae"):
        pts = [dict(x=float(rw["budget_x"]), y=float(rw["rms_error"]))
               for rw in rows if rw["method"] == method and rw["budget_x"] > 0
               and np.isfinite(rw["rms_error"]) and rw["rms_error"] > 0]
        pts.sort(key=lambda p: p["x"])
        series[method] = pts
        notes[method] = sorted({rw.get("note", "") for rw in rows
                                if rw["method"] == method} - {""})
        if len(pts) >= 2:
            xs = np.log([p["x"] for p in pts])
            ys = np.log([p["y"] for p in pts])
            slopes[method] = float(np.polyfit(xs, ys, 1)[0])
        else:
            slopes[method] = None
    return dict(ground_truth=gt, M=M,
                axis_label="oracle queries / samples (resource spent, NOT wall-clock)",
                series=series, slopes=slopes, notes=notes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_export_demo.py::test_build_convergence_has_real_descending_series_and_slopes -v`
Expected: PASS (may take ~30–60s — real QAE runs)

- [ ] **Step 5: Commit**

```bash
git add results/export_demo_data.py results/test_export_demo.py
git commit -m "feat(export): convergence-series builder for the demo frontend"
```

---

## Task 3: Prices + tree builder and `main()` writer

Add the price-path/tree builder and the hardware placeholder, then a `main()` that writes all three JSON files. `main()` must NOT clobber an existing real `hardware.json` (the teammate may have already dropped a Q50 result).

**Files:**
- Modify: `results/export_demo_data.py`
- Test: `results/test_export_demo.py`

- [ ] **Step 1: Write the failing test**

Append to `results/test_export_demo.py`:

```python
import json
import os


def test_build_prices_carries_path_and_tree():
    series_meta = ({"dates": ["a", "b", "c"], "closes": [4.0, 4.2, 4.1],
                    "S0": 4.1, "sigma": 0.3, "r": 0.03}, {"source": "synthetic"})
    p = ex.build_prices(series_meta, K=4.1, T=1.0, M_paths=3)
    assert p["closes"] == [4.0, 4.2, 4.1] and p["source"] == "synthetic"
    assert p["tree"]["M"] == 3
    # 2^M terminal values + matching path probabilities that sum to 1
    assert len(p["tree"]["terminal_values"]) == 8
    assert len(p["tree"]["path_probs"]) == 8
    assert abs(sum(p["tree"]["path_probs"]) - 1.0) < 1e-9


def test_hardware_placeholder_is_pending():
    hw = ex.hardware_placeholder()
    assert hw["status"] == "pending" and hw["backend"] == "Q50"


def test_main_writes_three_jsons_and_preserves_real_hardware(tmp_path):
    # a real hardware result already present must NOT be overwritten
    hw = tmp_path / "hardware.json"
    hw.write_text(json.dumps({"status": "done", "price": 2.78}))
    ex.main(out_dir=str(tmp_path), allow_network=False, M_conv=4, seeds=2)
    for name in ("prices.json", "convergence.json", "hardware.json"):
        assert (tmp_path / name).exists()
    assert json.loads(hw.read_text())["status"] == "done"  # preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_export_demo.py -k "build_prices or hardware_placeholder or main_writes" -v`
Expected: FAIL with `AttributeError: module 'results.export_demo_data' has no attribute 'build_prices'`

- [ ] **Step 3: Write minimal implementation**

Append to `results/export_demo_data.py`:

```python
def build_prices(series_meta, K, T, M_paths=3):
    """Real price path (act 1) + the exact 2^M CRR tree (act 2 superposition)."""
    series, meta = series_meta
    S0, sigma, r = series["S0"], series["sigma"], series["r"]
    vals = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_paths)
    probs = tree.path_probabilities(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M_paths)
    return dict(ticker=meta.get("ticker", "NOKIA.HE"), currency="EUR",
                source=meta.get("source", "synthetic"),
                window="%s..%s" % (series["dates"][0], series["dates"][-1]),
                dates=series["dates"], closes=series["closes"],
                S0=S0, sigma=sigma, r=r, K=K, T=T,
                tree=dict(M=M_paths,
                          terminal_values=[float(v) for v in vals],
                          path_probs=[float(p) for p in probs]))


def hardware_placeholder():
    """Q50 slot a teammate overwrites with real counts. Frontend shows 'pending'."""
    return dict(status="pending", backend="Q50", route="fourier",
                price=None, abs_error=None, shots=None, note="awaiting teammate Q50 run")


def main(out_dir=None, allow_network=True, M_paths=3, M_conv=5, seeds=6):
    out_dir = out_dir or os.path.join(_HERE)
    os.makedirs(out_dir, exist_ok=True)
    series, meta = nokia_price_series(allow_network=allow_network)
    S0, sigma, r = series["S0"], series["sigma"], series["r"]
    K = round(S0, 2)

    prices = build_prices((series, meta), K=K, T=1.0, M_paths=M_paths)
    conv = build_convergence(S0=S0, K=K, r=r, sigma=sigma, T=1.0, M=M_conv, seeds=seeds)

    with open(os.path.join(out_dir, "prices.json"), "w") as fh:
        json.dump(prices, fh, indent=2)
    with open(os.path.join(out_dir, "convergence.json"), "w") as fh:
        json.dump(conv, fh, indent=2)
    hw_path = os.path.join(out_dir, "hardware.json")
    if not os.path.exists(hw_path):          # never clobber a real Q50 result
        with open(hw_path, "w") as fh:
            json.dump(hardware_placeholder(), fh, indent=2)
    return dict(prices=prices, convergence=conv)


if __name__ == "__main__":
    main()
    print("wrote prices.json, convergence.json, hardware.json to results/")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_export_demo.py -k "build_prices or hardware_placeholder or main_writes" -v`
Expected: PASS

- [ ] **Step 5: Generate the real data files and commit**

```bash
quantum_pricer/.venv/bin/python -m results.export_demo_data
git add results/export_demo_data.py results/test_export_demo.py results/prices.json results/convergence.json results/hardware.json
git commit -m "feat(export): prices+tree+hardware writer; generate real demo JSONs"
```

---

## Task 4: Frontend skeleton — data load, act framework, graceful degradation

Create the self-contained HTML with: CSS, a data-loading layer (fetch the 4 JSONs, show an inline notice on failure instead of a blank screen), keyboard navigation (←/→/space across acts, `E` toggles cockpit), and empty act containers wired to render functions added in later tasks.

**Files:**
- Create: `results/demo_animation.html`
- Test: `results/test_demo_animation.py`

- [ ] **Step 1: Write the failing test**

Create `results/test_demo_animation.py`:

```python
import os

HTML = os.path.join(os.path.dirname(__file__), "demo_animation.html")


def _html():
    with open(HTML) as fh:
        return fh.read()


def test_html_is_self_contained_and_loads_all_four_jsons():
    h = _html()
    assert h.lstrip().lower().startswith("<!doctype html")
    for f in ("results.json", "prices.json", "convergence.json", "hardware.json"):
        assert f in h, f"frontend must fetch {f}"
    # no framework / CDN dependency -> demo survives dead wifi
    assert "http://" not in h and "https://" not in h


def test_html_has_navigation_and_cockpit_toggle():
    h = _html()
    assert "ArrowRight" in h and "ArrowLeft" in h
    assert "'e'" in h.lower() or '"e"' in h.lower()  # E toggles cockpit
    assert "data not exported yet" in h.lower()       # degradation notice
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py -v`
Expected: FAIL with `FileNotFoundError: .../demo_animation.html`

- [ ] **Step 3: Write minimal implementation**

Create `results/demo_animation.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Madness of People Is Quantum — Pricing Race</title>
<style>
  :root{--bg:#0b1020;--ink:#eaf0ff;--muted:#9fb0d4;--line:#2a3760;
        --quantum:#3ec98a;--classical:#9fb0d4;--sota:#c9603e;--qsvt:#9b6cff;--gt:#e0b341;}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Arial,sans-serif}
  #stage{height:100vh;display:flex;align-items:center;justify-content:center;padding:4vh 6vw}
  .act{display:none;width:100%;max-width:1100px}
  .act.live{display:block;animation:fade .5s ease}
  @keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
  h1{font-size:34px;margin:0 0 6px;background:linear-gradient(135deg,#7c5cff,#23d5c8);
    -webkit-background-clip:text;background-clip:text;color:transparent;font-weight:800}
  .cap{color:var(--muted);font-size:15px;max-width:760px}
  .num{color:var(--gt);font-weight:700}
  #nav{position:fixed;bottom:14px;left:0;right:0;text-align:center;color:var(--muted);font-size:13px}
  #dots span{display:inline-block;width:9px;height:9px;border-radius:50%;
    background:#33406a;margin:0 4px}
  #dots span.on{background:var(--quantum)}
  #notice{position:fixed;top:12px;left:12px;right:12px;background:#3a1f1f;color:#ffb4b4;
    border:1px solid #6b2b2b;border-radius:8px;padding:10px 14px;font-size:13px;display:none}
  #cockpit{display:none;grid-template-columns:1fr 1fr;gap:14px;width:100%;max-width:1180px}
  #cockpit.live{display:grid}
  .panel{background:#141d3a;border:1px solid var(--line);border-radius:12px;padding:14px}
  .label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
  svg{width:100%;height:auto;overflow:visible}
</style>
</head>
<body>
<div id="notice"></div>
<div id="stage">
  <section class="act" data-act="0"></section>
  <section class="act" data-act="1"></section>
  <section class="act" data-act="2"></section>
  <section class="act" data-act="3"></section>
  <section class="act" data-act="4"></section>
  <div id="cockpit"></div>
</div>
<div id="nav"><span id="dots"></span><div>← → / space to move · press <b>E</b> for explore cockpit</div></div>

<script>
const DATA = {};            // filled by loadData(); render functions read from here
let ACT = 0, COCKPIT = false;
const ACTS = document.querySelectorAll('.act');
const RENDERERS = [];       // RENDERERS[i] draws act i; added in later tasks

async function getJSON(name){
  try{ const r = await fetch(name); if(!r.ok) throw 0; return await r.json(); }
  catch(e){ return null; }
}
async function loadData(){
  const [results, prices, conv, hw] = await Promise.all(
    ['results.json','prices.json','convergence.json','hardware.json'].map(getJSON));
  Object.assign(DATA, {results, prices, conv, hw});
  const missing = [];
  if(!prices) missing.push('prices.json');
  if(!conv) missing.push('convergence.json');
  if(missing.length){
    const n = document.getElementById('notice');
    n.style.display='block';
    n.textContent = 'Data not exported yet — run  python -m results.export_demo_data  '
                  + '(missing: '+missing.join(', ')+'). Showing what is available.';
  }
}
function renderDots(){
  const d = document.getElementById('dots'); d.innerHTML='';
  ACTS.forEach((_,i)=>{ const s=document.createElement('span');
    if(i===ACT && !COCKPIT) s.className='on'; d.appendChild(s); });
}
function show(){
  document.getElementById('cockpit').classList.toggle('live', COCKPIT);
  ACTS.forEach((el,i)=>el.classList.toggle('live', !COCKPIT && i===ACT));
  if(!COCKPIT && RENDERERS[ACT]) RENDERERS[ACT](ACTS[ACT]);
  if(COCKPIT && typeof renderCockpit==='function') renderCockpit(document.getElementById('cockpit'));
  renderDots();
}
document.addEventListener('keydown', e=>{
  const k = e.key.toLowerCase();
  if(k==='e'){ COCKPIT=!COCKPIT; show(); return; }
  if(COCKPIT) return;
  if(e.key==='ArrowRight'||e.key===' '){ ACT=Math.min(ACT+1, ACTS.length-1); show(); }
  if(e.key==='ArrowLeft'){ ACT=Math.max(ACT-1, 0); show(); }
});
loadData().then(show);
</script>
</body>
</html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add results/demo_animation.html results/test_demo_animation.py
git commit -m "feat(frontend): self-contained demo skeleton with nav + graceful degradation"
```

---

## Task 5: Acts 1–2 — the madness chart + load-all-paths superposition

Render act 0 (real Nokia path drawing in) and act 1 (the chart fanning into the 2^M tree, captioned with the free/exact loading win). All geometry comes from `prices.json`.

**Files:**
- Modify: `results/demo_animation.html`
- Test: `results/test_demo_animation.py`

- [ ] **Step 1: Write the failing test**

Append to `results/test_demo_animation.py`:

```python
def test_acts_1_2_have_madness_and_loading_content():
    h = _html()
    assert "madness of people" in h.lower()
    assert "2^M" in h or "2<sup>M" in h          # 2^M paths messaging
    assert "no loading oracle" in h.lower() or "no distribution-loading" in h.lower()
    assert "renderAct0" in h and "renderAct1" in h
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py::test_acts_1_2_have_madness_and_loading_content -v`
Expected: FAIL (markers not present yet)

- [ ] **Step 3: Write minimal implementation**

Insert the following just before `loadData().then(show);` in `results/demo_animation.html`:

```javascript
// ---- helpers ----------------------------------------------------------------
function polyline(pts, color, w){
  return '<polyline points="'+pts.map(p=>p[0]+','+p[1]).join(' ')
       + '" fill="none" stroke="'+color+'" stroke-width="'+(w||2)+'"/>';
}
function scalePath(closes, W, H){
  const lo=Math.min(...closes), hi=Math.max(...closes), span=(hi-lo)||1;
  return closes.map((c,i)=>[ i*(W/(closes.length-1)), H-(c-lo)/span*H ]);
}

// ---- Act 0: the madness of people ------------------------------------------
RENDERERS[0] = function(el){
  const p = DATA.prices;
  const closes = p ? p.closes : [];
  const W=900,H=240, pts = closes.length ? scalePath(closes,W,H) : [];
  el.innerHTML =
    '<h1>The madness of people is quantum</h1>'
   +'<p class="cap">Newton could not predict "the madness of people"; Feynman said classical '
   +'intuition fails on the quantum world. Both say the same thing: <b>markets are non-classical.</b> '
   +'Here is real '+(p? p.ticker : 'NOKIA.HE')+', jagged and unpredictable'
   +(p && p.source!=='yfinance' ? ' <i>(synthetic stand-in — offline)</i>' : '')+'.</p>'
   +'<svg viewBox="0 0 '+W+' '+H+'">'+ (pts.length? polyline(pts,'#6c8cff',2):'') +'</svg>';
};

// ---- Act 1: load every path at once ----------------------------------------
RENDERERS[1] = function(el){
  const p = DATA.prices; const M = p && p.tree ? p.tree.M : 3;
  const W=420,H=240, x0=10, dx=(W-20)/M;
  // draw all 2^M up/down paths from S0 (CRR: up=bit 1)
  let lines='';
  const N=Math.pow(2,M);
  for(let path=0; path<N; path++){
    let y=H/2, pts=[[x0,y]];
    for(let step=0; step<M; step++){
      const up=(path>>step)&1; y += up? -H/(2*(M+1)) : H/(2*(M+1));
      pts.push([x0+(step+1)*dx, y]);
    }
    lines += polyline(pts,'rgba(108,140,255,0.45)',1);
  }
  el.innerHTML =
    '<h1>Load every path at once</h1>'
   +'<p class="cap"><span class="num">M</span> single-qubit rotations load all '
   +'<span class="num">2<sup>M</sup></span> price paths into one superposition — '
   +'<b>free and exact, no loading oracle</b> (the cost that bottlenecks textbook quantum pricers).</p>'
   +'<div style="display:flex;gap:30px;align-items:center">'
   +'<svg viewBox="0 0 '+W+' '+H+'" style="max-width:440px">'+lines+'</svg>'
   +'<div style="font-size:40px;color:var(--muted)">&rarr;</div>'
   +'<div class="panel" style="width:260px;height:160px;background:'
   +'radial-gradient(circle at 40% 40%,#6c8cff55,transparent 70%),'
   +'radial-gradient(circle at 65% 60%,#9b6cff44,transparent 70%)">'
   +'<div class="label">superposition of '+N+' paths</div></div></div>';
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py::test_acts_1_2_have_madness_and_loading_content -v`
Expected: PASS

- [ ] **Step 5: Manual browser check**

Run: `quantum_pricer/.venv/bin/python -m results.export_demo_data` (if JSONs not fresh), then open `results/demo_animation.html` in a browser. Verify act 0 shows the jagged Nokia path, → act 1 shows the fan of paths into a superposition cloud, zero console errors.

- [ ] **Step 6: Commit**

```bash
git add results/demo_animation.html results/test_demo_animation.py
git commit -m "feat(frontend): acts 1-2 madness chart + load-all-paths superposition"
```

---

## Task 6: Acts 3–4 — the race + the quadratic-speedup plot

Act 2 (the race: all models' prices vs the €-ground-truth line, exact route snaps, MC creeps, SOTA lags; counters from `results.json`). Act 3 (error-vs-**queries** log-log drawn from `convergence.json`, with slopes and the QSVT-floor / queries-not-time caveats).

**Files:**
- Modify: `results/demo_animation.html`
- Test: `results/test_demo_animation.py`

- [ ] **Step 1: Write the failing test**

Append to `results/test_demo_animation.py`:

```python
def test_acts_3_4_have_race_and_speedup_with_honesty_labels():
    h = _html()
    assert "renderAct2" in h and "renderAct3" in h
    assert "ground truth" in h.lower()
    # axis must be queries/samples, never time
    assert "queries" in h.lower()
    assert "not wall-clock" in h.lower() or "not time" in h.lower()
    # QSVT honesty: plateau / floor mentioned
    assert "floor" in h.lower() or "plateau" in h.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py::test_acts_3_4_have_race_and_speedup_with_honesty_labels -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Insert before `loadData().then(show);` in `results/demo_animation.html`:

```javascript
// ---- Act 2: the race -------------------------------------------------------
function routeColor(name){
  const s=name.toLowerCase();
  if(s.includes('sota')) return 'var(--sota)';
  if(s.includes('monte')) return 'var(--classical)';
  if(s.includes('qsvt')) return 'var(--qsvt)';
  return 'var(--quantum)';
}
RENDERERS[2] = function(el){
  const R = DATA.results;
  const gt = R ? R.references.tree_price : (DATA.conv? DATA.conv.ground_truth : 1);
  const routes = R ? R.routes : [];
  let bars = routes.map(rt=>{
    const acc = 1 - Math.min(1, Math.abs(rt.abs_err)/ (0.06)); // closeness to truth (0..1)
    const w = Math.max(8, acc*100);
    const q = rt.queries==null ? 'exact (statevector)' : (rt.queries.toLocaleString()+' queries');
    return '<div style="margin:6px 0"><div class="label">'+rt.name+' — €'+rt.price.toFixed(4)
         + ' · '+q+'</div><div style="height:14px;border-radius:7px;border-left:3px dashed var(--gt);'
         + 'width:'+w+'%;background:'+routeColor(rt.name)+'"></div></div>';
  }).join('');
  el.innerHTML =
    '<h1>Read the price without collapsing it</h1>'
   +'<p class="cap">Every model estimates the option price at once against the ground-truth line '
   +'(exact CRR tree = <span class="num">€'+gt.toFixed(4)+'</span>). Our exact/Fourier route '
   +'<b>snaps</b> to truth; Monte Carlo creeps; the textbook SOTA lags on its costly loading oracle.</p>'
   + bars;
};

// ---- Act 3: quadratic speedup (error vs QUERIES) ---------------------------
RENDERERS[3] = function(el){
  const c = DATA.conv;
  const W=820,H=320, pad=40;
  function logbox(series){
    const all=[].concat(...Object.values(series));
    const xs=all.map(p=>Math.log10(p.x)), ys=all.map(p=>Math.log10(p.y));
    const xlo=Math.min(...xs),xhi=Math.max(...xs),ylo=Math.min(...ys),yhi=Math.max(...ys);
    const sx=x=>pad+(Math.log10(x)-xlo)/((xhi-xlo)||1)*(W-2*pad);
    const sy=y=>H-pad-(Math.log10(y)-ylo)/((yhi-ylo)||1)*(H-2*pad);
    return {sx,sy};
  }
  let body='';
  if(c && c.series.classical_mc.length && c.series.qae.length){
    const {sx,sy}=logbox(c.series);
    const mk=(pts,col)=>polyline(pts.map(p=>[sx(p.x),sy(p.y)]),col,2.4);
    body = '<svg viewBox="0 0 '+W+' '+H+'">'
         + mk(c.series.classical_mc,'var(--classical)')
         + mk(c.series.qae,'var(--quantum)')
         + '<text x="'+(W-pad)+'" y="'+(H-12)+'" fill="var(--muted)" font-size="12" '
         + 'text-anchor="end">'+c.axis_label+'</text></svg>'
         + '<p class="cap">Monte Carlo slope <span class="num">'
         + (c.slopes.classical_mc!=null?c.slopes.classical_mc.toFixed(2):'~-0.5')
         + '</span> (error ~ 1/&radic;N) vs QAE slope <span class="num">'
         + (c.slopes.qae!=null?c.slopes.qae.toFixed(2):'~-1.0')
         + '</span> (error ~ 1/queries) &rarr; <b>quadratically fewer queries at tight accuracy.</b></p>';
  } else {
    body = '<p class="cap">convergence.json not loaded.</p>';
  }
  el.innerHTML =
    '<h1>Quadratic speedup</h1>'
   + body
   + '<p class="cap" style="opacity:.75">Honesty: the axis is <b>oracle queries / samples — '
   + 'NOT wall-clock time</b> (on today\'s hardware quantum is slower in seconds; the win is '
   + 'error-per-query). The QAE lead appears only at tight accuracy. QSVT carries a polynomial '
   + 'approximation <b>floor</b> (~1–2%) and plateaus rather than reaching zero.</p>';
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py::test_acts_3_4_have_race_and_speedup_with_honesty_labels -v`
Expected: PASS

- [ ] **Step 5: Manual browser check**

Open `results/demo_animation.html`; arrow to act 3 (race bars with €prices + query counts vs the ground-truth dashed line) and act 4 (two descending log-log lines, QAE steeper). Confirm the queries-not-time and QSVT-floor captions render. Zero console errors.

- [ ] **Step 6: Commit**

```bash
git add results/demo_animation.html results/test_demo_animation.py
git commit -m "feat(frontend): acts 3-4 pricing race + quadratic-speedup plot"
```

---

## Task 7: Act 5 + the explore cockpit + Q50 hardware badge

Act 4 (beat-the-textbook numbers from `results.json` SOTA row + a Q50 badge from `hardware.json` that reads "pending" until a real result lands). Plus `renderCockpit` — the all-panels mode-A view reachable via `E`.

**Files:**
- Modify: `results/demo_animation.html`
- Test: `results/test_demo_animation.py`

- [ ] **Step 1: Write the failing test**

Append to `results/test_demo_animation.py`:

```python
def test_act5_and_cockpit_and_hardware_badge():
    h = _html()
    assert "renderAct4" in h and "renderCockpit" in h
    assert "q50" in h.lower()
    assert "pending" in h.lower()       # badge default state
    assert "beat the textbook" in h.lower() or "vs sota" in h.lower() or "oracle-qae" in h.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py::test_act5_and_cockpit_and_hardware_badge -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Insert before `loadData().then(show);` in `results/demo_animation.html`:

```javascript
function sotaRow(){ const R=DATA.results; if(!R) return null;
  return R.routes.find(rt=>rt.name.toLowerCase().includes('sota')); }
function oursBest(){ const R=DATA.results; if(!R) return null;
  return R.routes.filter(rt=>rt.group==='ours')
          .sort((a,b)=>Math.abs(a.abs_err)-Math.abs(b.abs_err))[0]; }
function hwBadge(){
  const hw = DATA.hw || {status:'pending', backend:'Q50'};
  const done = hw.status==='done';
  const color = done? 'var(--quantum)':'var(--muted)';
  const txt = done ? (hw.backend+' ✓ €'+(hw.price!=null?hw.price.toFixed(4):'?')
                      +' (abs err '+(hw.abs_error!=null?hw.abs_error.toFixed(4):'?')+')')
                   : (hw.backend+' · pending (awaiting hardware run)');
  return '<span class="label" style="color:'+color+'">● '+txt+'</span>';
}

// ---- Act 4: beat the textbook + real hardware ------------------------------
RENDERERS[4] = function(el){
  const sota=sotaRow(), ours=oursBest();
  let cmp = (sota&&ours)
    ? 'Our best route abs-error <span class="num">'+Math.abs(ours.abs_err).toFixed(4)
      +'</span> vs Qiskit-Finance oracle-QAE (SOTA) <span class="num">'
      +Math.abs(sota.abs_err).toFixed(4)+'</span> — same ground truth, fewer assumptions.'
    : 'results.json not loaded.';
  el.innerHTML =
    '<h1>Beat the textbook — and run it on real hardware</h1>'
   +'<p class="cap">'+cmp+'</p>'
   +'<div class="panel" style="display:inline-block">'
   +'<div class="label">Q50 hardware</div>'+hwBadge()+'</div>';
};

// ---- Cockpit (mode A): everything at once ----------------------------------
function renderCockpit(el){
  const R=DATA.results, c=DATA.conv;
  const rows = R ? R.routes.map(rt=>'<div>'+rt.name+' — €'+rt.price.toFixed(4)
        +' ('+(rt.queries==null?'exact':rt.queries.toLocaleString()+'q')+')</div>').join('') : '';
  el.innerHTML =
    '<div class="panel"><div class="label">stock + 2^M tree</div>'
   +'<div id="ck-stock"></div></div>'
   +'<div class="panel"><div class="label">race leaderboard</div>'+rows+'</div>'
   +'<div class="panel"><div class="label">convergence (queries → error)</div>'
   +'<div id="ck-conv"></div></div>'
   +'<div class="panel"><div class="label">Q50 hardware</div>'+hwBadge()
   +'<div class="label" style="margin-top:8px">slopes: MC '
   +(c&&c.slopes.classical_mc!=null?c.slopes.classical_mc.toFixed(2):'?')+' · QAE '
   +(c&&c.slopes.qae!=null?c.slopes.qae.toFixed(2):'?')+'</div></div>';
  // reuse the act renderers inside the panels
  if(RENDERERS[0]) RENDERERS[0](document.getElementById('ck-stock'));
  if(RENDERERS[3]) RENDERERS[3](document.getElementById('ck-conv'));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py::test_act5_and_cockpit_and_hardware_badge -v`
Expected: PASS

- [ ] **Step 5: Manual browser check**

Open the HTML: act 5 shows the ours-vs-SOTA error comparison + a grey "Q50 · pending" badge. Press `E` → 2×2 cockpit with all panels; press `E` again → back to the story. Then edit `results/hardware.json` to `{"status":"done","backend":"Q50","route":"fourier","price":2.7752,"abs_error":0.0003,"shots":4096}`, reload, and confirm the badge turns green with the number. Revert the file afterward.

- [ ] **Step 6: Commit**

```bash
git add results/demo_animation.html results/test_demo_animation.py
git commit -m "feat(frontend): act 5 beat-SOTA + Q50 badge + explore cockpit toggle"
```

---

## Task 8: Optional Flask live server + re-run buttons (additive, degrades silently)

Add a small Flask server exposing `/health` and `/api/rerun/<route>` that re-prices on demand. The frontend probes `/health` at load: if reachable it shows "⟳ re-run live" buttons; if not, it shows nothing extra and the prebaked demo is unchanged.

**Files:**
- Modify: `quantum_pricer/requirements.txt`
- Create: `results/live_server.py`
- Test: `results/test_live_server.py`
- Modify: `results/demo_animation.html`

- [ ] **Step 1: Add Flask dependency and install it**

Append to `quantum_pricer/requirements.txt`:

```
flask>=3.0          # optional live "re-run a route" server (results/live_server.py)
```

Run: `quantum_pricer/.venv/bin/python -m pip install "flask>=3.0"`
Expected: Flask installs into the venv.

- [ ] **Step 2: Write the failing test**

Create `results/test_live_server.py`:

```python
from results import live_server


def test_health_ok():
    client = live_server.app.test_client()
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_rerun_qae_returns_price_near_truth():
    client = live_server.app.test_client()
    # small fast instance; route must return a price + query count near the tree truth
    r = client.get("/api/rerun/qae?S0=100&K=100&r=0.05&sigma=0.2&T=1&M=3&eps=0.05")
    assert r.status_code == 200
    body = r.get_json()
    assert body["route"] == "qae"
    assert body["price"] > 0 and body["queries"] >= 0
    assert abs(body["price"] - body["ground_truth"]) < 0.5


def test_rerun_unknown_route_is_400():
    client = live_server.app.test_client()
    r = client.get("/api/rerun/nope")
    assert r.status_code == 400
```

- [ ] **Step 3: Run test to verify it fails**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_live_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'results.live_server'`

- [ ] **Step 4: Write minimal implementation**

Create `results/live_server.py`:

```python
"""Optional, ADDITIVE live server: re-run a single pricing route on demand for judges.

The demo_animation.html frontend works fully WITHOUT this server. When it is running,
the frontend shows '⟳ re-run live' buttons; when it is not, those buttons are hidden and
the prebaked animation is unchanged. Never required for the demo.

Run:  quantum_pricer/.venv/bin/python -m results.live_server   (serves on :5057)
"""
from flask import Flask, jsonify, request

from quantum_pricer import classical, qae, tree

app = Flask(__name__)


@app.after_request
def _cors(resp):                      # allow the file:// frontend to call us
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.get("/health")
def health():
    return jsonify(ok=True)


def _params():
    g = request.args
    return dict(S0=float(g.get("S0", 100)), K=float(g.get("K", 100)),
                r=float(g.get("r", 0.05)), sigma=float(g.get("sigma", 0.2)),
                T=float(g.get("T", 1.0)), M=int(g.get("M", 3)))


@app.get("/api/rerun/<route>")
def rerun(route):
    p = _params()
    gt = tree.exact_tree_price(**p)
    eps = float(request.args.get("eps", 0.05))
    if route == "qae":
        res = qae.price(epsilon_target=eps, **p)
        price, queries = res["price"], res["num_oracle_queries"]
    elif route == "mc":
        n = int(request.args.get("n", 20000))
        price, _ = classical.monte_carlo_price(n_paths=n, **p)
        queries = n
    else:
        return jsonify(error="unknown route '%s' (use qae|mc)" % route), 400
    return jsonify(route=route, price=float(price), queries=int(queries),
                   ground_truth=float(gt), abs_error=abs(float(price) - float(gt)))


if __name__ == "__main__":
    app.run(port=5057)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_live_server.py -v`
Expected: PASS

- [ ] **Step 6: Wire the (silently-degrading) re-run buttons into the frontend**

Insert before `loadData().then(show);` in `results/demo_animation.html`:

```javascript
const LIVE = {url:'http://localhost:5057', on:false};
async function probeLive(){
  try{ const r=await fetch(LIVE.url+'/health',{signal:AbortSignal.timeout(600)});
       LIVE.on = r.ok; }catch(e){ LIVE.on=false; }
}
async function rerun(route){
  if(!LIVE.on) return;
  const p=DATA.prices||{};
  const qs='S0='+(p.S0||100)+'&K='+(p.K||100)+'&r='+(p.r||0.05)
          +'&sigma='+(p.sigma||0.2)+'&T='+(p.T||1)+'&M=3&eps=0.05';
  try{
    const res=await (await fetch(LIVE.url+'/api/rerun/'+route+'?'+qs)).json();
    const box=document.getElementById('live-out');
    if(box) box.textContent = route+' → €'+res.price.toFixed(4)+' ('+res.queries+' queries, err '
                              +res.abs_error.toFixed(4)+')';
  }catch(e){}
}
function liveControls(){
  if(!LIVE.on) return '';                 // server absent → nothing rendered
  return '<div class="panel" style="margin-top:10px"><div class="label">live (server up)</div>'
       + '<button onclick="rerun(\'qae\')">⟳ re-run QAE</button> '
       + '<button onclick="rerun(\'mc\')">⟳ re-run Monte Carlo</button>'
       + '<div id="live-out" class="label" style="margin-top:6px"></div></div>';
}
```

Then, inside `RENDERERS[2]` (act 2, the race), change the final `+ bars;` line to:

```javascript
   + bars + liveControls();
```

And change the bottom of `loadData()` — replace the final line `loadData().then(show);` with:

```javascript
Promise.all([loadData(), probeLive()]).then(show);
```

- [ ] **Step 7: Verify graceful degradation (server down) and the frontend tests still pass**

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py -v`
Expected: PASS (HTML still self-contained; `http://localhost:5057` is the only allowed local URL — update the Task 4 assertion if needed: see Step 8).

- [ ] **Step 8: Loosen the "no URL" assertion to allow the localhost live probe**

In `results/test_demo_animation.py`, replace:

```python
    assert "http://" not in h and "https://" not in h
```

with:

```python
    # only the optional localhost live-server probe may reference a URL; no CDNs
    urls = [l for l in h.splitlines() if "http://" in l or "https://" in l]
    assert all("localhost:5057" in l for l in urls), urls
```

Run: `quantum_pricer/.venv/bin/python -m pytest results/test_demo_animation.py -v`
Expected: PASS

- [ ] **Step 9: Manual check — both modes**

(a) Open `results/demo_animation.html` with NO server running → act 3 shows no live buttons, demo plays normally. (b) Start `quantum_pricer/.venv/bin/python -m results.live_server`, reload → "⟳ re-run QAE / Monte Carlo" buttons appear on the race act; clicking shows a fresh price near ground truth.

- [ ] **Step 10: Commit**

```bash
git add quantum_pricer/requirements.txt results/live_server.py results/test_live_server.py results/demo_animation.html results/test_demo_animation.py
git commit -m "feat(frontend): optional Flask re-run server with silent degradation"
```

---

## Task 9: README + full-suite verification

Document how to run the demo and verify the whole thing end to end.

**Files:**
- Modify: `results/README.md`

- [ ] **Step 1: Append run instructions to `results/README.md`**

Append:

```markdown
## Animated pricing-race demo

1. Export the real data (once, or after new pricer runs):
   `quantum_pricer/.venv/bin/python -m results.export_demo_data`
   → writes `prices.json`, `convergence.json`, `hardware.json`.
2. Open `results/demo_animation.html` in a browser. ←/→/space move through the 5 acts;
   press **E** for the explore cockpit. Works fully offline from `file://`.
3. (Optional) live mode: `quantum_pricer/.venv/bin/python -m results.live_server`,
   then reload — adds "⟳ re-run" buttons. If the server is down the demo is unchanged.
4. Q50 hardware: a teammate overwrites `results/hardware.json` with
   `{"status":"done","backend":"Q50","route":"fourier","price":<p>,"abs_error":<e>,"shots":<n>}`;
   the act-5 badge and cockpit then light up green. Until then it reads "pending".

Honesty: the convergence/race axis is **oracle queries / samples, not wall-clock time**;
the QAE lead appears at tight accuracy; QSVT plateaus at its polynomial-approx floor;
ground truth is the exact CRR tree price.
```

- [ ] **Step 2: Run the full affected test suite**

Run:
```bash
quantum_pricer/.venv/bin/python -m pytest \
  quantum_pricer/tests/test_data.py \
  results/test_export_demo.py \
  results/test_demo_animation.py \
  results/test_live_server.py -v
```
Expected: all PASS.

- [ ] **Step 3: Regenerate fresh real data and final manual smoke**

Run: `quantum_pricer/.venv/bin/python -m results.export_demo_data`, open `results/demo_animation.html`, walk all 5 acts + cockpit, confirm zero console errors and every number renders.

- [ ] **Step 4: Commit**

```bash
git add results/README.md results/prices.json results/convergence.json results/hardware.json
git commit -m "docs(results): document the animated pricing-race demo + regenerate data"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** §3 acts → Tasks 5–7; §2 honesty caveats → asserted in Tasks 5/6 tests + captions; §4.1 exporters → Tasks 1–3; §4.2 frontend + degradation → Tasks 4–7; §4.3 live server → Task 8; §6 error handling → Task 4 notice + Task 8 silent hide; §7 testing → every task; §9 success criteria → Task 9 verification.
- **Type consistency:** `qae.price` returns `num_oracle_queries` (used in Task 8 server); `error_vs_queries_rms` rows use `budget_x`/`rms_error`/`note` (used in Task 2); `results.json` routes use `name`/`group`/`price`/`abs_err`/`queries` (used in race/cockpit); `tree.exact_tree_price` is the ground truth everywhere.
- **No placeholders:** every code step is complete and runnable.
```
