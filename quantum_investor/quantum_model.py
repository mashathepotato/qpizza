"""
Quantum model of investor decision-making (the part Newton couldn't predict).

Core idea
---------
Two yes/no questions, A and B, are modelled as PROJECTIVE MEASUREMENTS on a
single-qubit "belief state". Because the two questions correspond to measurement
axes that are *tilted* relative to each other, their projectors do NOT commute.
Non-commuting measurements => the order you ask them in changes the answers.

That is the quantum signature of human judgement:
  * ORDER EFFECT:  p(B=yes | A asked first)  !=  p(B=yes | B asked first)
  * QQ-EQUALITY:   a *parameter-free* identity the model satisfies exactly
                   (Wang, Solloway, Shiffrin & Busemeyer, PNAS 2014) — and which
                   real survey data also satisfy. This is the killer evidence:
                   a metaphor can't make a parameter-free prediction; this does.

We use PennyLane to prepare the belief state as a real quantum circuit, then do
the sequential projective-measurement algebra explicitly so every step is visible.

Finance framing (swap the wording freely):
  A = "Do you trust the market right now?"      (yes / no)
  B = "Will you invest your savings today?"      (yes / no)
"""

import numpy as np
import pennylane as qml

_dev = qml.device("default.qubit", wires=1)


@qml.qnode(_dev)
def _belief_state_circuit(alpha):
    """Prepare the belief state |psi> = cos(alpha)|0> + sin(alpha)|1> on a qubit."""
    qml.RY(2.0 * alpha, wires=0)
    return qml.state()


def belief_state(alpha):
    """Belief state as a length-2 complex vector, via a PennyLane circuit."""
    return np.asarray(_belief_state_circuit(alpha), dtype=complex)


def _projectors(beta):
    """
    Projectors for the two questions.

    Question A is measured in the computational basis:
        |A=yes> = |0>,  |A=no> = |1>
    Question B is measured in a basis rotated by angle `beta` (the "tilt"):
        |B=yes> =  cos(beta)|0> + sin(beta)|1>
        |B=no>  = -sin(beta)|0> + cos(beta)|1>
    beta = 0  => the questions are identical (no order effect).
    beta != 0 => the axes are tilted, projectors don't commute => order matters.
    """
    a_yes = np.array([1.0, 0.0], dtype=complex)
    a_no = np.array([0.0, 1.0], dtype=complex)
    b_yes = np.array([np.cos(beta), np.sin(beta)], dtype=complex)
    b_no = np.array([-np.sin(beta), np.cos(beta)], dtype=complex)

    def P(v):  # projector |v><v|
        return np.outer(v, v.conj())

    return {
        "A": (P(a_yes), P(a_no)),
        "B": (P(b_yes), P(b_no)),
    }


def _sequential_joint(psi, first, second, projectors):
    """
    Joint probabilities of measuring `first` then `second` (with collapse).

    For projective measurements, p(x then y) = || P_y P_x |psi> ||^2, which
    automatically equals p(x) * p(y | post-measurement state).

    Returns a dict keyed by the two answers, e.g. {'yy','yn','ny','nn'}, where the
    first letter is the `first` question's answer and the second is `second`'s.
    """
    (fx_yes, fx_no) = projectors[first]
    (sy_yes, sy_no) = projectors[second]
    out = {}
    for fname, Pf in (("y", fx_yes), ("n", fx_no)):
        for sname, Ps in (("y", sy_yes), ("n", sy_no)):
            v = Ps @ (Pf @ psi)
            out[fname + sname] = float(np.vdot(v, v).real)
    return out


def predict(alpha, beta):
    """
    Full prediction of the quantum model for parameters (alpha, beta).

    Returns a dict with:
      'AB' : joint probs when A is asked first  (keys = A-answer + B-answer)
      'BA' : joint probs when B is asked first  (keys = B-answer + A-answer)
      'pB_yes_after_A' : p(B=yes) when A was asked first
      'pB_yes_first'   : p(B=yes) when B was asked first
      'order_effect_B' : the difference (0 for a classical model, != 0 here)
      'qq'             : the QQ-equality value (≈ 0 exactly, parameter-free)
    """
    psi = belief_state(alpha)
    proj = _projectors(beta)

    AB = _sequential_joint(psi, "A", "B", proj)  # A then B
    BA = _sequential_joint(psi, "B", "A", proj)  # B then A

    # Marginal p(B=yes) in each order
    pB_yes_after_A = AB["yy"] + AB["ny"]   # A any, B yes
    pB_yes_first = BA["yy"] + BA["yn"]     # B yes, A any

    # QQ-equality: prob of giving *different* answers is order-independent.
    #   AB different = (A yes, B no) + (A no, B yes)
    #   BA different = (B yes, A no) + (B no, A yes)
    ab_diff = AB["yn"] + AB["ny"]
    ba_diff = BA["yn"] + BA["ny"]
    qq = ab_diff - ba_diff

    return {
        "AB": AB,
        "BA": BA,
        "pB_yes_after_A": pB_yes_after_A,
        "pB_yes_first": pB_yes_first,
        "order_effect_B": pB_yes_first - pB_yes_after_A,
        "qq": qq,
    }


if __name__ == "__main__":
    # Quick sanity demo: show an order effect and the parameter-free QQ-equality.
    for (a, b) in [(0.6, 0.4), (0.9, 0.7), (0.3, 1.1)]:
        r = predict(a, b)
        print(
            f"alpha={a:.2f} beta={b:.2f} | "
            f"p(B=yes|A first)={r['pB_yes_after_A']:.3f}  "
            f"p(B=yes|B first)={r['pB_yes_first']:.3f}  "
            f"order_effect={r['order_effect_B']:+.3f}  "
            f"QQ={r['qq']:+.2e}"
        )
    print("\nNote: order_effect is nonzero (order matters) yet QQ ≈ 0 always — "
          "the parameter-free quantum prediction.")
