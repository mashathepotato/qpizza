"""Live, auto-refreshing HTML dashboard for the quantum option pricer.

Collects the CURRENT state (market params, ground-truth + 4 route prices, resource
table, fitted benchmark slopes, optional q50_fake hardware run, git activity),
regenerates both money-slide figures, persists a JSON state + an append-only
activity log, and renders a self-contained dark-theme `dashboard.html` that the
browser reloads every 15 s.

Usage
-----
  python -m quantum_pricer.dashboard            # one full build, prints HTML path
  python -m quantum_pricer.dashboard --loop 15  # live: first tick full, rest light
  python -m quantum_pricer.dashboard --loop 15 --recompute   # every tick full
  python -m quantum_pricer.dashboard --loop 15 --light       # every tick light

Open `quantum_pricer/dashboard.html` in a browser; with --loop it stays current.
The HTML works offline (no CDN/JS); it just <meta>-refreshes every 15 s.
"""
import argparse
import datetime
import html as _html
import json
import os
import subprocess
import time
import traceback

from quantum_pricer import (
    benchmark,
    classical,
    data,
    fourier,
    qae,
    qsvt,
    tree,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(_HERE, "dashboard.html")
STATE_PATH = os.path.join(_HERE, "dashboard_state.json")
LOG_PATH = os.path.join(_HERE, "dashboard_log.jsonl")
COMPLEXITY_PNG = os.path.join(_HERE, "complexity.png")
SPEEDUP_PNG = os.path.join(_HERE, "speedup.png")


# --------------------------------------------------------------------------- utils
def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _try(fn, default=None):
    """Run fn(); on any failure return (default, error_string)."""
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - dashboard must never blank out
        return default, f"{type(exc).__name__}: {exc}"


def _git_activity():
    """Return (commit_count, [last 8 subjects]) via git, or (None, [])."""
    try:
        count = subprocess.run(["git", "rev-list", "--count", "HEAD"],
                               cwd=_HERE, capture_output=True, text=True,
                               timeout=10).stdout.strip()
        log = subprocess.run(["git", "log", "--oneline", "-8"],
                             cwd=_HERE, capture_output=True, text=True,
                             timeout=10).stdout.strip().splitlines()
        return (int(count) if count.isdigit() else None), log
    except Exception:  # noqa: BLE001
        return None, []


# --------------------------------------------------------------------------- collect
def collect_state(allow_network=True, run_hw=True, M=4, T=1.0):
    """Gather the full dashboard state. Every heavy call is isolated so one failure
    records an error rather than blanking the whole dashboard."""
    errors = {}
    state = {"timestamp": _now(), "M": int(M), "T": float(T), "errors": errors}

    # --- market params + provenance -----------------------------------------------
    (pm, err) = _try(lambda: data.nokia_params(allow_network=allow_network))
    if err:
        errors["params"] = err
        params = dict(S0=4.20, sigma=0.30, r=0.03)
        meta = dict(source="synthetic", reason="params fetch failed")
    else:
        params, meta = pm
    state["params"] = params
    state["meta"] = meta

    S0 = params["S0"]
    sigma = params["sigma"]
    r = params["r"]
    K = round(S0, 2)
    state["K"] = K
    bp = dict(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)

    # --- ground truth: exact tree + Black-Scholes ---------------------------------
    exact, err = _try(lambda: tree.exact_tree_price(**bp))
    if err:
        errors["exact_tree"] = err
    state["exact_tree"] = exact
    bs, err = _try(lambda: classical.black_scholes_call(S0=S0, K=K, r=r,
                                                        sigma=sigma, T=T))
    if err:
        errors["black_scholes"] = err
    state["black_scholes"] = bs

    def _err_vs_exact(price):
        if price is None or exact is None:
            return None
        return float(price - exact)

    prices = {}
    state["prices"] = prices

    # --- classical Monte Carlo ----------------------------------------------------
    mc, err = _try(lambda: classical.monte_carlo_price(**bp, n_paths=200_000,
                                                       seed=0))
    if err:
        errors["classical_mc"] = err
        prices["classical_mc"] = dict(price=None, err=None, note="Monte Carlo failed")
    else:
        mc_price, mc_se = mc
        prices["classical_mc"] = dict(
            price=float(mc_price), stderr=float(mc_se),
            err=_err_vs_exact(mc_price),
            note="200k paths, O(1/eps^2) statistical error")

    # --- Fourier (QNDM characteristic function) -----------------------------------
    fp, err = _try(lambda: fourier.price(**bp))
    if err:
        errors["fourier"] = err
        prices["fourier"] = dict(price=None, err=None, note="Fourier failed")
    else:
        prices["fourier"] = dict(
            price=float(fp), err=_err_vs_exact(fp),
            note="O(1/eps^2) in shots; Q50 anchor (shallow, exact loading)")

    # --- QAE (genuine O(1/eps) queries) -------------------------------------------
    qres, err = _try(lambda: qae.price(**bp, epsilon_target=0.01))
    if err:
        errors["qae"] = err
        prices["qae"] = dict(price=None, err=None, note="QAE failed")
    else:
        prices["qae"] = dict(
            price=float(qres["price"]), err=_err_vs_exact(qres["price"]),
            queries=int(qres["num_oracle_queries"]),
            note="O(1/eps) oracle queries (saturates at small M)")

    # --- QSVT (straddle + put-call parity) ----------------------------------------
    qsv, err = _try(lambda: qsvt.price(**bp, degree=60, return_meta=True))
    if err:
        errors["qsvt"] = err
        prices["qsvt"] = dict(price=None, err=None, note="QSVT failed")
    else:
        prices["qsvt"] = dict(
            price=float(qsv["price"]), err=_err_vs_exact(qsv["price"]),
            degree=int(qsv["degree"]),
            probe_residual=float(qsv["probe_residual"]),
            note="straddle E[|f-K|] + parity, ~1.4% approx floor")

    # --- resource table (qubits, IQM cz_depth) ------------------------------------
    rt, err = _try(lambda: benchmark.resource_table(**bp, qsvt_degree=20), [])
    if err:
        errors["resource_table"] = err
    state["resource_table"] = rt or []

    # --- benchmark fitted slopes (small grid) -------------------------------------
    slopes = {}
    state["slopes"] = slopes
    qa_rows, err = _try(lambda: benchmark.queries_to_accuracy(
        **bp, epsilons=(0.04, 0.02, 0.01)), [])
    if err:
        errors["slopes"] = err
    else:
        import numpy as np

        def _fit(method, kind):
            pts = sorted((1.0 / x["epsilon"], x["queries"]) for x in qa_rows
                         if x["method"] == method and x["kind"] == kind
                         and x["epsilon"] > 0 and x["queries"] > 0)
            if len(pts) < 2:
                return None
            xs, ys = zip(*pts)
            return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])

        slopes["mc_analytic"] = _fit("classical_mc", "theoretical")  # ~2
        slopes["qae_theory"] = _fit("qae", "theoretical")            # ~1

    # empirical RMS slopes: read cheaply from a cached state if present (the 8-seed
    # sweep is too expensive to rerun every tick). Otherwise leave None.
    state["rms_slopes"] = _cached_rms_slopes()

    # --- hardware run (q50_fake, M=1) ---------------------------------------------
    if run_hw:
        from quantum_pricer import run_hardware
        hw, err = _try(lambda: run_hardware.run(backend_name="q50_fake", M=1,
                                                shots=20000, S0=S0, K=K, r=r,
                                                sigma=sigma, T=T))
        if err:
            errors["hardware"] = err
            state["hardware"] = None
        else:
            hw_price, hw_exact = hw
            state["hardware"] = dict(
                backend="q50_fake (IQMFakeAphrodite)", M=1,
                price=float(hw_price), exact=float(hw_exact),
                abs_err=float(abs(hw_price - hw_exact)))
    else:
        state["hardware"] = None

    # --- git activity --------------------------------------------------------------
    commit_count, commits = _git_activity()
    state["git_commit_count"] = commit_count
    state["git_commits"] = commits

    return state


def _cached_rms_slopes():
    """Best-effort: read previously computed empirical RMS slopes from the state file
    so loop ticks don't rerun the expensive seed sweep."""
    try:
        with open(STATE_PATH) as fh:
            prev = json.load(fh)
        return prev.get("rms_slopes")
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- log
def _summary_line(state):
    parts = []
    qsv = state.get("prices", {}).get("qsvt", {})
    exact = state.get("exact_tree")
    if qsv.get("err") is not None and exact:
        parts.append(f"QSVT err {100 * qsv['err'] / exact:+.2f}%")
    sl = state.get("slopes", {})
    if sl.get("mc_analytic") is not None:
        parts.append(f"MC slope {sl['mc_analytic']:.2f}")
    if sl.get("qae_theory") is not None:
        parts.append(f"QAE slope {sl['qae_theory']:.2f}")
    if state.get("errors"):
        parts.append(f"{len(state['errors'])} err")
    return "rebuilt: " + ", ".join(parts) if parts else "rebuilt"


def append_log(state, kind="full"):
    entry = dict(t=state.get("timestamp", _now()), kind=kind,
                 summary=_summary_line(state))
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: BLE001
        pass
    return entry


def read_log(n=20):
    try:
        with open(LOG_PATH) as fh:
            lines = fh.readlines()
        out = []
        for ln in lines[-n:]:
            try:
                out.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                continue
        return out
    except Exception:  # noqa: BLE001
        return []


def write_state(state):
    try:
        with open(STATE_PATH, "w") as fh:
            json.dump(state, fh, indent=2, default=str)
    except Exception:  # noqa: BLE001
        pass


def load_state():
    with open(STATE_PATH) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- render
def _e(x):
    return _html.escape(str(x))


def _fmt(x, prec=4):
    if x is None:
        return "&mdash;"
    try:
        return f"{float(x):.{prec}f}"
    except Exception:  # noqa: BLE001
        return _e(x)


def _price_card(name, label, info, exact, note_override=None):
    price = info.get("price")
    err = info.get("err")
    note = note_override or info.get("note", "")
    extra = ""
    if "queries" in info:
        extra += f"<div class='meta'>oracle queries: {_e(info['queries'])}</div>"
    if "stderr" in info:
        extra += f"<div class='meta'>stderr: {_fmt(info['stderr'])}</div>"
    if "degree" in info:
        extra += f"<div class='meta'>QSP degree: {_e(info['degree'])}"
        if "probe_residual" in info:
            extra += f", probe resid: {_fmt(info['probe_residual'], 4)}"
        extra += "</div>"

    err_html = ""
    if err is not None and exact:
        rel = 100.0 * err / exact
        cls = "good" if abs(rel) < 2.0 else ("warn" if abs(rel) < 6.0 else "bad")
        err_html = (f"<div class='err {cls}'>err vs tree: {err:+.4f} "
                    f"({rel:+.2f}%)</div>")
    return f"""
      <div class="card price">
        <div class="route">{_e(label)}</div>
        <div class="big">{_fmt(price)}</div>
        {err_html}
        {extra}
        <div class="note">{_e(note)}</div>
      </div>"""


def render_html(state, log_lines):
    ts = state.get("timestamp", _now())
    cb = _e(ts.replace(":", "").replace("-", ""))  # cache-buster
    meta = state.get("meta", {})
    source = str(meta.get("source", "synthetic")).lower()
    live = source == "yfinance"
    badge_cls = "live" if live else "synth"
    badge_txt = "LIVE (yfinance)" if live else "SYNTHETIC"
    params = state.get("params", {})
    exact = state.get("exact_tree")
    prices = state.get("prices", {})

    # market params card
    market = f"""
      <div class="card">
        <h2>Market parameters</h2>
        <table class="kv">
          <tr><td>S0 (spot)</td><td>{_fmt(params.get('S0'))}</td></tr>
          <tr><td>sigma (annualized)</td><td>{_fmt(params.get('sigma'))}</td></tr>
          <tr><td>r (risk-free)</td><td>{_fmt(params.get('r'))}</td></tr>
          <tr><td>K (ATM strike)</td><td>{_fmt(state.get('K'))}</td></tr>
          <tr><td>T (years)</td><td>{_fmt(state.get('T'), 2)}</td></tr>
          <tr><td>M (tree steps)</td><td>{_e(state.get('M'))}</td></tr>
          <tr><td>ticker</td><td>{_e(meta.get('ticker', 'NOKIA.HE'))}</td></tr>
          <tr><td>calibration window</td><td>{_e(meta.get('calib_start', '?'))} &rarr; {_e(meta.get('calib_end', '?'))} (&le; asof {_e(meta.get('asof', '?'))})</td></tr>
          <tr><td>n_obs</td><td>{_e(meta.get('n_obs', '?'))}</td></tr>
        </table>
      </div>"""

    # ground-truth card
    truth = f"""
      <div class="card truth">
        <h2>Ground truth</h2>
        <div class="route">Exact binomial tree</div>
        <div class="big gt">{_fmt(exact)}</div>
        <div class="note">Discounted expectation over all 2^M paths &mdash; the
        quantity every quantum route estimates.</div>
        <div class="route" style="margin-top:14px">Black-Scholes (continuum)</div>
        <div class="big">{_fmt(state.get('black_scholes'))}</div>
      </div>"""

    # route cards
    route_cards = "".join([
        _price_card("classical_mc", "Classical Monte Carlo",
                    prices.get("classical_mc", {}), exact),
        _price_card("fourier", "QNDM Fourier", prices.get("fourier", {}), exact),
        _price_card("qae", "QNDM QAE", prices.get("qae", {}), exact),
        _price_card("qsvt", "Novel QSVT", prices.get("qsvt", {}), exact),
    ])

    # resource table
    rt_rows = ""
    for row in state.get("resource_table", []):
        deep = row.get("method") == "qsvt"
        cls = "deep" if deep else ("shallow" if row.get("method") in
                                   ("fourier", "qae") else "")
        rt_rows += (f"<tr class='{cls}'><td>{_e(row.get('method'))}</td>"
                    f"<td>{_e(row.get('qubits'))}</td>"
                    f"<td>{_e(row.get('cz_depth'))}</td></tr>")
    resource = f"""
      <div class="card">
        <h2>Resource table (transpiled to IQM {{r, cz}})</h2>
        <table class="kv res">
          <tr><th>method</th><th>qubits</th><th>IQM cz depth</th></tr>
          {rt_rows}
        </table>
        <div class="note">Fourier / QAE stay shallow (Q50-feasible); QSVT is deep
        (high cz depth from the QET-U phase schedule).</div>
      </div>"""

    # slopes
    sl = state.get("slopes", {})
    rms = state.get("rms_slopes") or {}
    rms_txt = ""
    if rms.get("mc") is not None and rms.get("qae") is not None:
        rms_txt = (f" Empirical RMS slopes: MC {rms['mc']:.2f}, "
                   f"QAE {rms['qae']:.2f}.")
    slope_line = (f"MC analytic slope ~ {_fmt(sl.get('mc_analytic'), 2)} "
                  f"(quadratic), QAE theory slope ~ {_fmt(sl.get('qae_theory'), 2)} "
                  f"(linear).{rms_txt}")

    figures = f"""
      <div class="card">
        <h2>The quadratic advantage</h2>
        <div class="figrow">
          <figure>
            <img src="complexity.png?t={cb}" alt="query complexity">
            <figcaption>Query complexity: classical MC O(1/eps^2) (slope ~2) vs
            quantum amplitude estimation O(1/eps) (slope ~1).</figcaption>
          </figure>
          <figure>
            <img src="speedup.png?t={cb}" alt="empirical RMS error vs queries">
            <figcaption>Empirical RMS error vs queries: MC descends ~1/sqrt(N),
            QAE descends steeper toward 1/N.</figcaption>
          </figure>
        </div>
        <div class="note">{slope_line}</div>
      </div>"""

    # hardware
    hw = state.get("hardware")
    if hw:
        rel = 100.0 * hw["abs_err"] / hw["exact"] if hw["exact"] else 0.0
        hw_html = f"""
          <div class="card">
            <h2>Hardware (NISQ noise)</h2>
            <div class="route">{_e(hw['backend'])}, M={_e(hw['M'])}</div>
            <div class="hwgrid">
              <div><div class="big">{_fmt(hw['price'])}</div><div class="meta">noisy price</div></div>
              <div><div class="big">{_fmt(hw['exact'])}</div><div class="meta">exact</div></div>
              <div><div class="big">{rel:+.2f}%</div><div class="meta">abs error</div></div>
            </div>
            <div class="note">Real NISQ noise on the shallow Fourier route.</div>
          </div>"""
    else:
        hwerr = state.get("errors", {}).get("hardware")
        hw_html = f"""
          <div class="card">
            <h2>Hardware (NISQ noise)</h2>
            <div class="note">q50_fake run not available this build
            {('(' + _e(hwerr) + ')') if hwerr else '(run_hw disabled or qiskit-iqm missing)'}.</div>
          </div>"""

    honesty = """
      <div class="card honesty">
        <h2>Honesty notes</h2>
        <ul>
          <li><b>Ground truth</b> = exact binomial tree price, not Black-Scholes.</li>
          <li><b>QSVT</b> measures the straddle E[|f-K|] and recovers the call by
          put-call parity; it carries a ~1.4% polynomial-approximation residual near
          the kink. NO answer-calibration: every constant comes from the value range
          and the swept QSP phases.</li>
          <li><b>Fourier</b> sampling is O(1/eps^2) in shots (parallel to MC); its win
          is qubit count / shallow depth / Q50 feasibility, not eps-scaling.</li>
          <li><b>QAE</b> has genuine O(1/eps) query scaling but its IAE Grover schedule
          SATURATES at small M (several eps targets report the same query count).</li>
          <li><b>Hardware</b> shows real NISQ noise on q50_fake; errors are expected.</li>
        </ul>
      </div>"""

    # errors banner
    err_banner = ""
    if state.get("errors"):
        items = "".join(f"<li>{_e(k)}: {_e(v)}</li>"
                        for k, v in state["errors"].items())
        err_banner = f"<div class='card errbanner'><h2>Build warnings</h2><ul>{items}</ul></div>"

    # activity feed
    feed_items = ""
    for ln in reversed(log_lines[-20:]):
        feed_items += (f"<li><span class='t'>{_e(ln.get('t', ''))}</span> "
                       f"<span class='k'>[{_e(ln.get('kind', ''))}]</span> "
                       f"{_e(ln.get('summary', ''))}</li>")
    if not feed_items:
        feed_items = "<li>(no log entries yet)</li>"
    commit_items = "".join(f"<li>{_e(c)}</li>"
                           for c in state.get("git_commits", []))
    feed = f"""
      <div class="card">
        <h2>Activity feed</h2>
        <div class="feedcols">
          <div>
            <h3>Rebuilds (last 20)</h3>
            <ul class="feed">{feed_items}</ul>
          </div>
          <div>
            <h3>Recent commits ({_e(state.get('git_commit_count', '?'))} total)</h3>
            <ul class="feed">{commit_items or '<li>(none)</li>'}</ul>
          </div>
        </div>
      </div>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quantum Option Pricer &mdash; Live Dashboard</title>
<style>
  :root {{
    --bg:#0c1018; --card:#161c28; --card2:#1d2433; --fg:#e6ecf5; --muted:#8b97ad;
    --accent:#4ea1ff; --good:#37d399; --warn:#f5b942; --bad:#ff6b6b;
    --amber:#3a2f12; --amberb:#f5b942;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    line-height:1.45; }}
  header {{ padding:22px 28px; background:linear-gradient(120deg,#13192a,#0c1018);
    border-bottom:1px solid #222b3d; display:flex; align-items:center;
    justify-content:space-between; flex-wrap:wrap; gap:12px; }}
  header h1 {{ margin:0; font-size:22px; letter-spacing:.3px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-top:4px; }}
  .badge {{ padding:6px 14px; border-radius:20px; font-weight:700; font-size:13px;
    letter-spacing:.5px; }}
  .badge.live {{ background:#10331f; color:var(--good); border:1px solid #1f6b43; }}
  .badge.synth {{ background:#332a10; color:var(--warn); border:1px solid #6b561f; }}
  main {{ padding:22px 28px; max-width:1500px; margin:0 auto; }}
  .grid {{ display:grid; gap:18px; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); }}
  .card {{ background:var(--card); border:1px solid #222b3d; border-radius:14px;
    padding:18px 20px; }}
  .card h2 {{ margin:0 0 12px; font-size:15px; color:var(--accent);
    text-transform:uppercase; letter-spacing:.6px; }}
  .card h3 {{ margin:0 0 8px; font-size:13px; color:var(--muted); }}
  .price {{ background:var(--card2); }}
  .route {{ font-size:13px; color:var(--muted); }}
  .big {{ font-size:30px; font-weight:700; margin:2px 0; }}
  .big.gt {{ color:var(--good); }}
  .truth {{ background:linear-gradient(160deg,#14241c,#161c28);
    border-color:#1f6b43; grid-column:span 2; }}
  .err {{ font-size:13px; font-weight:600; margin-bottom:4px; }}
  .err.good {{ color:var(--good); }}
  .err.warn {{ color:var(--warn); }}
  .err.bad {{ color:var(--bad); }}
  .note {{ font-size:12px; color:var(--muted); margin-top:8px; }}
  .meta {{ font-size:12px; color:var(--muted); }}
  table.kv {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.kv td, table.kv th {{ padding:5px 6px; border-bottom:1px solid #222b3d;
    text-align:left; }}
  table.kv td:first-child {{ color:var(--muted); }}
  table.res tr.shallow td {{ color:var(--good); }}
  table.res tr.deep td {{ color:var(--warn); }}
  .figrow {{ display:flex; gap:18px; flex-wrap:wrap; }}
  .figrow figure {{ margin:0; flex:1 1 360px; }}
  .figrow img {{ width:100%; border-radius:10px; background:#fff; }}
  .figrow figcaption {{ font-size:12px; color:var(--muted); margin-top:6px; }}
  .hwgrid {{ display:flex; gap:24px; margin:10px 0; }}
  .hwgrid .big {{ font-size:24px; }}
  .honesty {{ background:var(--amber); border:1px solid var(--amberb); }}
  .honesty h2 {{ color:var(--amberb); }}
  .honesty ul {{ margin:0; padding-left:18px; font-size:13px; }}
  .honesty li {{ margin-bottom:7px; }}
  .errbanner {{ border-color:var(--bad); }}
  .errbanner h2 {{ color:var(--bad); }}
  .errbanner ul {{ font-size:12px; color:var(--muted); }}
  .feedcols {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  ul.feed {{ list-style:none; margin:0; padding:0; font-size:12px;
    font-family:ui-monospace,Menlo,monospace; max-height:240px; overflow:auto; }}
  ul.feed li {{ padding:3px 0; border-bottom:1px solid #1c2433; }}
  ul.feed .t {{ color:var(--muted); }}
  ul.feed .k {{ color:var(--accent); }}
  .full {{ grid-column:1 / -1; }}
  @media (max-width:720px) {{ .truth, .feedcols {{ grid-column:auto;
    grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Quantum Option Pricer &mdash; Live Dashboard</h1>
    <div class="sub">last updated {_e(ts)} &middot; auto-refresh 15s</div>
  </div>
  <div class="badge {badge_cls}">{_e(badge_txt)}</div>
</header>
<main>
  {err_banner}
  <div class="grid">
    {truth}
    {market}
  </div>
  <h2 style="margin:24px 0 6px;color:var(--accent);font-size:15px;letter-spacing:.6px;text-transform:uppercase">Route prices</h2>
  <div class="grid">
    {route_cards}
  </div>
  <div class="grid" style="margin-top:18px">
    <div class="full">{figures}</div>
  </div>
  <div class="grid" style="margin-top:18px">
    {resource}
    {hw_html}
  </div>
  <div class="grid" style="margin-top:18px">
    <div class="full">{honesty}</div>
  </div>
  <div class="grid" style="margin-top:18px">
    <div class="full">{feed}</div>
  </div>
</main>
</body>
</html>"""


# --------------------------------------------------------------------------- builds
def _regenerate_figures(state, M, T):
    """Recompute the two money-slide figures from the current params."""
    params = state.get("params", {})
    S0 = params.get("S0", 4.20)
    sigma = params.get("sigma", 0.30)
    r = params.get("r", 0.03)
    K = state.get("K", round(S0, 2))
    bp = dict(S0=S0, K=K, r=r, sigma=sigma, T=T)

    qa_rows, err = _try(lambda: benchmark.queries_to_accuracy(M=M, **bp))
    if not err and qa_rows:
        _try(lambda: benchmark.save_complexity_plot(qa_rows, path=COMPLEXITY_PNG))

    # RMS sweep: keep it modest so a full build is not glacial; record slopes so
    # light ticks can show them.
    rms_rows, err = _try(lambda: benchmark.error_vs_queries_rms(
        M=max(M, 4), seeds=4,
        mc_budgets=(250, 1000, 4000, 16000, 64000),
        qae_eps=(0.1, 0.05, 0.025, 0.012, 0.006), **bp))
    if not err and rms_rows:
        _try(lambda: benchmark.save_speedup_plot_rms(rms_rows, path=SPEEDUP_PNG))
        import numpy as np

        def _fit(method):
            pts = sorted((x["budget_x"], x["rms_error"]) for x in rms_rows
                         if x["method"] == method and x["budget_x"] > 0
                         and np.isfinite(x["rms_error"]) and x["rms_error"] > 0)
            if len(pts) < 2:
                return None
            xs, ys = zip(*pts)
            return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])

        state["rms_slopes"] = dict(mc=_fit("classical_mc"), qae=_fit("qae"))


def build_once(allow_network=True, run_hw=True, M=4, T=1.0, regen_figures=True):
    """Full build: collect state, regenerate both figures, persist state + log,
    render and write the HTML. Returns the state dict."""
    state = collect_state(allow_network=allow_network, run_hw=run_hw, M=M, T=T)
    if regen_figures:
        _try(lambda: _regenerate_figures(state, M, T))
    write_state(state)
    append_log(state, kind="full")
    log_lines = read_log(20)
    html = render_html(state, log_lines)
    with open(HTML_PATH, "w") as fh:
        fh.write(html)
    return state


def build_light(M=4, T=1.0):
    """Light tick: reuse the last full state's prices/figures, just refresh the
    timestamp + git activity + log feed and re-render the HTML."""
    try:
        state = load_state()
    except Exception:  # noqa: BLE001 - no prior full build; fall back to full
        return build_once(allow_network=False, run_hw=False, M=M, T=T)
    state["timestamp"] = _now()
    commit_count, commits = _git_activity()
    state["git_commit_count"] = commit_count
    state["git_commits"] = commits
    write_state(state)
    append_log(state, kind="light")
    html = render_html(state, read_log(20))
    with open(HTML_PATH, "w") as fh:
        fh.write(html)
    return state


# --------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description="Live quantum-pricer dashboard.")
    ap.add_argument("--once", action="store_true",
                    help="one full build (default).")
    ap.add_argument("--loop", type=float, metavar="SECONDS",
                    help="rebuild forever every SECONDS (first full, rest light).")
    ap.add_argument("--light", action="store_true",
                    help="with --loop, make EVERY tick light.")
    ap.add_argument("--recompute", action="store_true",
                    help="with --loop, make EVERY tick a full recompute.")
    ap.add_argument("--no-network", action="store_true",
                    help="force synthetic params (no yfinance).")
    ap.add_argument("--no-hw", action="store_true",
                    help="skip the q50_fake hardware run.")
    ap.add_argument("-M", type=int, default=4, help="tree steps (default 4).")
    args = ap.parse_args(argv)

    allow_network = not args.no_network
    run_hw = not args.no_hw

    if args.loop:
        first = True
        while True:
            t0 = time.time()
            try:
                if args.recompute or (first and not args.light):
                    build_once(allow_network=allow_network, run_hw=run_hw, M=args.M)
                    mode = "full"
                elif args.light:
                    build_light(M=args.M)
                    mode = "light"
                else:
                    build_light(M=args.M)
                    mode = "light"
                print(f"[{_now()}] tick ({mode}) -> {HTML_PATH}")
            except Exception as exc:  # noqa: BLE001 - keep looping
                print(f"[{_now()}] tick FAILED: {exc}")
                traceback.print_exc()
            first = False
            elapsed = time.time() - t0
            time.sleep(max(0.0, args.loop - elapsed))
    else:
        build_once(allow_network=allow_network, run_hw=run_hw, M=args.M)
        print(HTML_PATH)


if __name__ == "__main__":
    main()
