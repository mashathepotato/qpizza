"""
Fast self-checks so you trust the demo on stage. Run:  python test_model.py
(or: pytest test_model.py). No network, no hardware.
"""

import numpy as np

import quantum_model as qm
import classical_model as cm
import data as dataset


def test_qq_equality_is_parameter_free():
    # The QQ-equality must hold for ARBITRARY (alpha, beta) — the headline claim.
    rng = np.random.default_rng(1)
    for _ in range(50):
        alpha, beta = rng.uniform(0, np.pi, size=2)
        r = qm.predict(alpha, beta)
        assert abs(r["qq"]) < 1e-9, f"QQ-equality broken at {alpha},{beta}: {r['qq']}"


def test_order_effect_exists_when_axes_tilted():
    # Tilted axes (beta != 0) => a genuine order effect.
    r = qm.predict(0.62, 0.52)
    assert abs(r["order_effect_B"]) > 1e-3


def test_no_order_effect_when_questions_identical():
    # beta = 0 => same question twice => no order effect.
    r = qm.predict(0.62, 0.0)
    assert abs(r["order_effect_B"]) < 1e-9


def test_classical_cannot_produce_order_effect():
    data = dataset.human_data()
    pred = cm.predict(cm.fit(data))
    assert pred["order_effect_B"] == 0.0


def test_probabilities_normalised():
    r = qm.predict(0.7, 0.5)
    for order in ("AB", "BA"):
        s = sum(r[order].values())
        assert abs(s - 1.0) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("\nAll self-checks passed.")
