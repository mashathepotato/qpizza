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
    # technical pillar is 0.45 — a win vs loss difference of 0.45 always dominates.
    win = _rec(advantage_direction="win")
    loss = _rec(advantage_direction="loss")
    assert score(win) > score(loss)

def test_q50_ready_and_natural_demo_breaks_ties():
    # Two records identical except q50_faithful_runnable (True vs False) and
    # demo_naturalness (0.9 vs 0.2).  Feasibility (0.20) + business (0.10)
    # together lift score(a) above score(b).
    a = _rec(q50_faithful_runnable=True, demo_naturalness=0.9)
    b = _rec(q50_faithful_runnable=False, demo_naturalness=0.2)
    assert score(a) > score(b)

def test_rank_orders_descending_and_is_stable():
    recs = [_rec(config_id="x", advantage_direction="loss"),
            _rec(config_id="y", advantage_direction="win")]
    ranked = rank(recs)
    assert [r.config_id for r in ranked] == ["y", "x"]

def test_rank_breaks_score_ties_deterministically():
    # Two records with EQUAL score (same direction/litmus/sim/q50/demo/op)
    # but different advantage_magnitude — the higher-magnitude one must rank
    # first regardless of input order.
    high_mag = _rec(config_id="high", method="alpha", advantage_magnitude=10.0,
                    advantage_direction="tie", quantum_native_litmus=True,
                    sim_runnable=True, q50_faithful_runnable=True,
                    demo_naturalness=0.5, op_business_fit=0.5)
    low_mag  = _rec(config_id="low",  method="beta",  advantage_magnitude=1.0,
                    advantage_direction="tie", quantum_native_litmus=True,
                    sim_runnable=True, q50_faithful_runnable=True,
                    demo_naturalness=0.5, op_business_fit=0.5)

    # Verify they truly have the same score
    assert score(high_mag) == score(low_mag), (
        f"Scores differ: {score(high_mag):.4f} vs {score(low_mag):.4f} — "
        "test precondition failed"
    )

    # Input order 1: high first
    ranked_a = rank([high_mag, low_mag])
    # Input order 2: low first
    ranked_b = rank([low_mag, high_mag])

    assert ranked_a[0].config_id == "high", "high_mag should rank first (order A)"
    assert ranked_b[0].config_id == "high", "high_mag should rank first (order B)"
