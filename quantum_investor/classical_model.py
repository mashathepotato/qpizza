"""
Classical (Bayesian) baseline — the model that FAILS, on purpose.

A classical investor is described by a single fixed joint probability distribution
over the two questions:
    P(A=a, B=b)   for a,b in {yes,no}.

In classical probability the *order* of asking is irrelevant: the joint, and hence
both marginals, are the same whether you ask A-then-B or B-then-A. Therefore a
classical model is STRUCTURALLY INCAPABLE of producing an order effect — its
predicted order effect is identically zero.

When we fit one classical joint to data that contains an order effect, the best it
can do (least squares) is predict the *average* of the two orders, leaving a
residual exactly equal to the order effect it cannot represent. That residual is
the gap the quantum model closes.
"""

import numpy as np


def _canonicalise(data):
    """
    Map both measurement orders onto a common (A-answer, B-answer) joint.

    data['AB'] is keyed (A,B); data['BA'] is keyed (B,A). We rewrite BA into (A,B)
    so the two orders can be compared cell-by-cell.
    """
    AB = data["AB"]
    BA = data["BA"]
    cells = ["yy", "yn", "ny", "nn"]  # (A,B)
    ab = {c: AB[c] for c in cells}
    # BA key (b,a) -> canonical (a,b): swap the two letters
    ba = {a + b: BA[b + a] for a in "yn" for b in "yn"}
    return ab, ba


def fit(data):
    """
    Best-fit classical joint = elementwise average of the two orders.

    (This is the exact least-squares optimum, and it stays a valid probability
    distribution because each order's joint already sums to 1.)
    """
    ab, ba = _canonicalise(data)
    joint = {c: 0.5 * (ab[c] + ba[c]) for c in ab}
    return joint


def predict(joint):
    """
    A classical model predicts the SAME joint for both orders (order can't matter).

    Returns the same structure as quantum_model.predict for apples-to-apples
    comparison.
    """
    AB = dict(joint)                                  # (A,B)
    BA = {b + a: joint[a + b] for a in "yn" for b in "yn"}  # (B,A) — same probabilities
    pB_yes = joint["yy"] + joint["ny"]                # A any, B yes
    return {
        "AB": AB,
        "BA": BA,
        "pB_yes_after_A": pB_yes,
        "pB_yes_first": pB_yes,       # identical by construction
        "order_effect_B": 0.0,        # the classical model can NEVER produce one
        "qq": 0.0,
    }


def sse(prediction, data):
    """Sum of squared errors between a prediction and data, over both orders."""
    err = 0.0
    for order in ("AB", "BA"):
        for cell in ("yy", "yn", "ny", "nn"):
            err += (prediction[order][cell] - data[order][cell]) ** 2
    return err
