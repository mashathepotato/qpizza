"""The common scoring axis.  Every harness emits an AdvantageRecord; score()
collapses it to one number.

Weight rationale (sum = 1.0)
-----------------------------
  technical   0.45  — advantage_direction (win/tie/loss) is the only *measured*
                       head-to-head signal; it dominates because it is the
                       primary scientific question.
  novelty     0.25  — quantum_native_litmus is a hard structural property
                       (delete-quantum => collapses?); binary but objective.
  feasibility 0.20  — sim_runnable + q50_faithful_runnable are empirical pass/
                       fail results from actual circuit execution.
  business    0.10  — demo_naturalness + op_business_fit are HAND-SET constants
                       per harness and therefore act only as a small tiebreaker.
                       Keeping them non-zero preserves the signal when everything
                       else is tied, but they cannot flip a measured advantage.
"""
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
    sweep_value: float = float("nan")   # the swept-axis value for this config (eps, n_assets, n_features)
    sweep_label: str = ""               # axis name, e.g. "epsilon", "n_assets", "n_features"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AdvantageRecord":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def score(r: AdvantageRecord) -> float:
    """Return a composite score in [0, 1].

    Weights are intentionally unequal: measured signals (technical, novelty,
    feasibility) account for 0.90 of the total; the hand-set demo/business
    constants account for only 0.10 so they cannot override a real measured win.
    """
    technical   = _DIRECTION.get(r.advantage_direction, 0.0)          # 0/0.5/1
    novelty     = 1.0 if r.quantum_native_litmus else 0.0             # 0 or 1
    feasibility = 0.5 * float(r.sim_runnable) + 0.5 * float(r.q50_faithful_runnable)
    business    = 0.5 * r.demo_naturalness + 0.5 * r.op_business_fit  # hand-set

    return (
        0.45 * technical
        + 0.25 * novelty
        + 0.20 * feasibility
        + 0.10 * business
    )


def rank(records: list[AdvantageRecord]) -> list[AdvantageRecord]:
    """Return records sorted by score descending.

    Ties are broken deterministically by (score desc, advantage_magnitude desc,
    method name asc) so ordering never silently depends on input order.

    Implementation uses three stable sorts (Timsort is stable) applied in
    reverse priority order so the final sort's key is the primary criterion.
    """
    # Step 1 (lowest priority): method name ascending
    step1 = sorted(records, key=lambda r: r.method)
    # Step 2: advantage_magnitude descending
    step2 = sorted(step1, key=lambda r: r.advantage_magnitude, reverse=True)
    # Step 3 (highest priority): score descending
    return sorted(step2, key=score, reverse=True)
