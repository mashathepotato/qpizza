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
    win = _rec(advantage_direction="win")
    loss = _rec(advantage_direction="loss")
    assert score(win) > score(loss)

def test_q50_ready_and_natural_demo_breaks_ties():
    a = _rec(q50_faithful_runnable=True, demo_naturalness=0.9)
    b = _rec(q50_faithful_runnable=False, demo_naturalness=0.2)
    assert score(a) > score(b)

def test_rank_orders_descending_and_is_stable():
    recs = [_rec(config_id="x", advantage_direction="loss"),
            _rec(config_id="y", advantage_direction="win")]
    ranked = rank(recs)
    assert [r.config_id for r in ranked] == ["y", "x"]
