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
