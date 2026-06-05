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
