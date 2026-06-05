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
    "qae": "Quantum Amplitude Estimation: O(1/eps) vs Monte-Carlo O(1/eps2). "
           "Option pricing & VaR/CVaR. Deep circuits -> LUMI-sim story.",
    "qaoa": "QAOA portfolio (cardinality-constrained QUBO). Combinatorial = the "
            "quantum-native version. Shallow -> Q50-runnable.",
    "fraud_qml": "Quantum-kernel SVM on card fraud. Visceral live demo; "
                 "shallow feature map -> Q50-hardware-friendly.",
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
           litmus: {'OK' if rec.quantum_native_litmus else 'X'}</div>
      <span style="background:{badge};color:#fff;border-radius:6px;padding:2px 8px;
                   font-size:12px">{badge_txt}</span>
      <span style="font-size:12px;color:#555"> demo {rec.demo_naturalness:.2f} /
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
            f"<p>Waiting for first result... {completed}/{total}</p></body></html>")
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
