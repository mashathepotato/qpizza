"""Self-contained auto-refreshing dashboard. One HTML file, plots embedded as
base64 PNGs (no server, no external assets). Re-render after every config."""
from __future__ import annotations
import base64
import collections
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from triage.rubric import AdvantageRecord, score, rank

# ---------------------------------------------------------------------------
# Per-method structured copy
# Each entry: what, good
# ---------------------------------------------------------------------------
_METHOD_COPY = {
    "qae": {
        "what": (
            "Quantum Amplitude Estimation loads a probability distribution into "
            "qubit amplitudes and estimates an expectation — here a European-call "
            "price and tail risk (VaR/CVaR). Its error falls as O(1/eps), a "
            "quadratic speedup over Monte-Carlo's O(1/eps²)."
        ),
        "good": (
            "Good result: far fewer quantum oracle queries than classical "
            "Monte-Carlo samples to reach the same accuracy eps — and a "
            "scaling-curve slope near −1 (quantum) vs −2 (classical). "
            "Caveat: only at TIGHT accuracy; at coarse eps the fixed per-round "
            "cost lets classical win."
        ),
    },
    "qaoa": {
        "what": (
            "QAOA picks exactly k of n assets to maximise return minus "
            "risk×variance — a combinatorial (NP-hard-family) problem encoded "
            "as a QUBO/Ising and solved with a shallow variational circuit. "
            "It is quantum-native because the discrete cardinality constraint is "
            "genuinely combinatorial, unlike the convex continuous mean-variance "
            "problem."
        ),
        "good": (
            "Good result: matches or beats a classical simulated-annealing "
            "heuristic's solution quality, with the edge expected at larger n "
            "where exact search is infeasible."
        ),
    },
    "fraud_qml": {
        "what": (
            "A quantum-kernel SVM embeds each transaction into a quantum feature "
            "map and uses the quantum state overlap (fidelity) as the similarity "
            "kernel — a measure that can be classically expensive to compute."
        ),
        "good": (
            "Good result: ROC-AUC meaningfully ABOVE the classical RBF-SVM on "
            "the same data. Honest finding so far: on tabular fraud it TIES "
            "classical — strong on demo/business naturalness, but no measured "
            "quantum edge."
        ),
    },
}

# Fallback for unknown methods
def _method_what(method: str) -> str:
    return _METHOD_COPY.get(method, {}).get(
        "what",
        f"{method}: quantum vs classical comparison."
    )

def _method_good(method: str) -> str:
    return _METHOD_COPY.get(method, {}).get(
        "good",
        "Good result: quantum measurably beats the classical baseline."
    )


# ---------------------------------------------------------------------------
# Plain-language interpretation of the best record
# ---------------------------------------------------------------------------
def _interpret(rec: AdvantageRecord) -> str:
    q = rec.quantum_metric
    c = rec.classical_metric
    direction = rec.advantage_direction
    if rec.metric_name == "samples_to_eps":
        # lower is better; ratio = c/q means quantum is that many times cheaper
        if direction == "win" and q > 0:
            ratio = c / q
            return (
                f"Quantum used {q:.0f} queries vs {c:.0f} classical samples "
                f"&#8212; {ratio:.1f}x fewer (advantage)."
            )
        elif direction == "loss":
            return (
                f"Classical cheaper here ({c:.0f} vs {q:.0f}) "
                f"&#8212; advantage only appears at tighter accuracy."
            )
        else:
            return f"About even ({q:.0f} vs {c:.0f})."
    elif rec.metric_name == "approx_ratio":
        return (
            f"QAOA reached {q:.2f} of the optimal objective "
            f"(1.0&nbsp;=&nbsp;optimal)."
        )
    elif rec.metric_name == "auc":
        if q > c + 0.01:
            verdict = "ahead"
        elif abs(q - c) <= 0.01:
            verdict = "about even"
        else:
            verdict = "behind"
        return f"Quantum AUC {q:.3f} vs classical {c:.3f} &#8212; {verdict}."
    else:
        return f"Quantum {q:.4g} vs classical {c:.4g} ({direction})."


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
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


def _scaling_png(method_records) -> str | None:
    """Log-log scaling curve over the swept axis, built from ALL of a method's
    records. Returns base64 PNG, or None when there's nothing to plot (fewer
    than 2 records, or every sweep_value is NaN)."""
    recs = sorted(method_records, key=lambda r: r.sweep_value)
    if len(recs) < 2 or all(math.isnan(r.sweep_value) for r in recs):
        return None
    xs = [r.sweep_value for r in recs]
    q = [r.quantum_metric for r in recs]
    c = [r.classical_metric for r in recs]
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.loglog(xs, q, "o-", color="#4c72b0", label="quantum")
    ax.loglog(xs, c, "s-", color="#999999", label="classical")
    ax.set_xlabel(recs[0].sweep_label, fontsize=8)
    ax.set_ylabel(recs[0].metric_name, fontsize=8)
    ax.set_title(f"{recs[0].method}: quantum vs classical scaling", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Best-per-method
# ---------------------------------------------------------------------------
def _best_per_method(records):
    best = {}
    for r in records:
        if r.method not in best or score(r) > score(best[r.method]):
            best[r.method] = r
    return best


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------
def _card(best_rec: AdvantageRecord, method_records) -> str:
    rec = best_rec
    png = _bar_png(rec)
    badge = ("#2e7d32" if rec.q50_faithful_runnable else "#b71c1c")
    badge_txt = "Q50 ready" if rec.q50_faithful_runnable else "Q50 N/A"
    scaling = _scaling_png(method_records)
    if scaling:
        scaling_html = (
            f'<div style="margin-top:8px">'
            f'<img src="data:image/png;base64,{scaling}"/>'
            f'<div style="font-size:11px;color:#777">scaling across {rec.sweep_label}</div>'
            f'</div>'
        )
    else:
        scaling_html = ""

    interpretation = _interpret(rec)
    what_text = _method_what(rec.method)
    good_text = _method_good(rec.method)

    return f"""
    <div style="border:1px solid #ddd;border-radius:10px;padding:16px;margin:10px;
                width:360px;box-shadow:0 1px 4px rgba(0,0,0,.08);
                font-family:sans-serif">
      <h2 style="margin:0 0 4px;font-size:18px">{rec.method} <small style="font-weight:normal;color:#555">({rec.candidate})</small></h2>

      <p style="color:#444;font-size:13px;line-height:1.5;margin:6px 0 4px">{what_text}</p>

      <p style="color:#1a5c1a;font-size:12px;font-style:italic;margin:4px 0 8px;
                background:#f0faf0;border-left:3px solid #2e7d32;padding:4px 8px;
                border-radius:0 4px 4px 0">{good_text}</p>

      <div style="background:#fffbe6;border:1px solid #ffe082;border-radius:6px;
                  padding:8px 10px;margin:8px 0;font-size:13px">
        <b>Result:</b> {interpretation}
      </div>

      <div style="margin:4px 0"><b>Score {score(rec):.3f}</b> &middot;
           advantage: <b>{rec.advantage_direction}</b></div>
      <div style="font-size:12px;color:#555">scaling sig: {rec.scaling_signature:.3g} &middot;
           litmus: {'OK' if rec.quantum_native_litmus else 'X'}</div>
      <div style="margin:6px 0">
        <span style="background:{badge};color:#fff;border-radius:6px;padding:2px 8px;
                     font-size:12px">{badge_txt}</span>
        <span style="font-size:12px;color:#555"> demo {rec.demo_naturalness:.2f} /
             OP {rec.op_business_fit:.2f}</span>
      </div>
      <div><img src="data:image/png;base64,{png}" style="margin-top:8px"/></div>
      {scaling_html}
    </div>"""


# ---------------------------------------------------------------------------
# How-to-read panel
# ---------------------------------------------------------------------------
_HOW_TO_READ = """
<div style="background:#f0f4ff;border:1px solid #b0c0e8;border-radius:10px;
            padding:20px 24px;margin:16px 0 20px;max-width:860px;font-size:14px;
            line-height:1.7;font-family:sans-serif">
  <h2 style="margin:0 0 10px;font-size:17px;color:#1a2a6c">How to read this dashboard</h2>
  <p style="margin:0 0 8px">
    <b>Adversarial triage:</b> each candidate makes a falsifiable claim of quantum
    advantage; we try to <em>refute</em> it by measuring against the best classical
    baseline. A method &ldquo;wins&rdquo; only if it measurably beats classical.
  </p>
  <p style="margin:0 0 8px">
    <b>Reading a card:</b> the bar shows quantum vs classical on that method&rsquo;s
    metric (best config); the scaling curve shows how they scale as the problem
    gets harder (smaller eps / larger n / more features).
  </p>
  <p style="margin:0 0 8px">
    <b>Advantage labels:</b>
    <span style="color:#2e7d32;font-weight:bold">WIN</span> = quantum measurably
    beats classical &nbsp;|&nbsp;
    <span style="color:#888;font-weight:bold">TIE</span> = within noise &nbsp;|&nbsp;
    <span style="color:#b71c1c;font-weight:bold">LOSS</span> = classical wins
    (the falsifier fired &mdash; an honest negative result).
  </p>
  <p style="margin:0 0 8px">
    <b>Score (0&ndash;1)</b> = 0.45&thinsp;&times;&thinsp;measured-advantage
    &nbsp;+&nbsp; 0.25&thinsp;&times;&thinsp;quantum-native-litmus
    &nbsp;+&nbsp; 0.20&thinsp;&times;&thinsp;feasibility (simulator + Q50)
    &nbsp;+&nbsp; 0.10&thinsp;&times;&thinsp;business (demo + OP-fit).
    Measured signals dominate; demo/business are only tiebreakers.
  </p>
  <p style="margin:0 0 8px">
    <b>Q50 badge</b> = the circuit transpiles to VTT Q50&rsquo;s native gates
    (phased-Rx + CZ) and survives its noise model &mdash; i.e. runnable on
    Europe&rsquo;s first 50-qubit machine.
  </p>
  <p style="margin:0">
    <b>What you want to see:</b> a candidate worth pursuing shows a measured WIN
    with a clean scaling separation, passes the quantum-native litmus, and is
    Q50-runnable.
  </p>
</div>
"""

# ---------------------------------------------------------------------------
# Metric glossary
# ---------------------------------------------------------------------------
_GLOSSARY = """
<div style="max-width:860px;margin:24px 0 8px;font-family:sans-serif">
  <h2 style="font-size:16px;color:#333;border-bottom:1px solid #ddd;
             padding-bottom:6px;margin-bottom:10px">Metric glossary</h2>
  <table style="border-collapse:collapse;width:100%;font-size:13px">
    <thead>
      <tr style="background:#f5f5f5;text-align:left">
        <th style="padding:7px 12px;border:1px solid #ddd;width:160px">Metric</th>
        <th style="padding:7px 12px;border:1px solid #ddd">Definition</th>
        <th style="padding:7px 12px;border:1px solid #ddd;width:220px">What counts as good</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:7px 12px;border:1px solid #ddd;font-family:monospace">samples_to_eps</td>
        <td style="padding:7px 12px;border:1px solid #ddd">
          Oracle calls (quantum) or MC samples (classical) needed to reach
          accuracy&nbsp;&epsilon;. <em>Lower is better.</em>
        </td>
        <td style="padding:7px 12px;border:1px solid #ddd">
          Quantum &lt;&lt; classical at the same &epsilon;; the gap widens as
          &epsilon; shrinks.
        </td>
      </tr>
      <tr style="background:#fafafa">
        <td style="padding:7px 12px;border:1px solid #ddd;font-family:monospace">approx_ratio</td>
        <td style="padding:7px 12px;border:1px solid #ddd">
          Objective achieved / true optimum, in [0,&nbsp;1]. <em>Higher is better.</em>
          1.0&nbsp;= exact optimum.
        </td>
        <td style="padding:7px 12px;border:1px solid #ddd">
          Ratio near 1.0, staying high as problem size n grows.
        </td>
      </tr>
      <tr>
        <td style="padding:7px 12px;border:1px solid #ddd;font-family:monospace">auc</td>
        <td style="padding:7px 12px;border:1px solid #ddd">
          ROC-AUC (Area Under the ROC Curve), in [0,&nbsp;1]. <em>Higher is
          better.</em> 0.5&nbsp;= random; 1.0&nbsp;= perfect classifier.
        </td>
        <td style="padding:7px 12px;border:1px solid #ddd">
          Quantum AUC &gt; classical AUC by a margin larger than noise.
        </td>
      </tr>
      <tr style="background:#fafafa">
        <td style="padding:7px 12px;border:1px solid #ddd;font-family:monospace">scaling signature</td>
        <td style="padding:7px 12px;border:1px solid #ddd">
          The measured advantage ratio or log-log slope indicator across the
          swept axis. A value &gt;&nbsp;1 means the quantum curve grows slower
          than the classical curve.
        </td>
        <td style="padding:7px 12px;border:1px solid #ddd">
          Monotonically increasing with problem hardness; separation visible in
          the log-log scaling plot.
        </td>
      </tr>
    </tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# Top-level render
# ---------------------------------------------------------------------------
def render(records, plots_dir, out: Path, completed: int, total: int) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        out.write_text(
            "<html><head><meta http-equiv=\"refresh\" content=\"30\"></head>"
            "<body style='font-family:sans-serif'><h1>Triage</h1>"
            f"<p>Waiting for first result... {completed}/{total}</p></body></html>")
        return
    by_method = collections.defaultdict(list)
    for r in records:
        by_method[r.method].append(r)
    best = _best_per_method(records)
    ranked = rank(list(best.values()))
    leader = ranked[0]
    cards = "".join(_card(r, by_method[r.method]) for r in ranked)
    html = f"""<html><head>
      <meta http-equiv="refresh" content="30">
      <title>Quantum-finance triage</title>
      <style>
        body {{ font-family: sans-serif; margin: 24px; max-width: 1100px; }}
        h1 {{ color: #1a2a6c; margin-bottom: 4px; }}
      </style>
      </head>
      <body>
      <h1>Quantum-finance triage</h1>
      <p style="color:#444"><b>{completed}/{total}</b> configs &middot;
         current leader: <b>{leader.method}</b> (candidate {leader.candidate},
         score {score(leader):.3f})</p>
      {_HOW_TO_READ}
      <div style="display:flex;flex-wrap:wrap">{cards}</div>
      {_GLOSSARY}
      </body></html>"""
    out.write_text(html)
