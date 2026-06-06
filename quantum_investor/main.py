"""
Run the whole demo end-to-end.

    $ python main.py

Prints the narrative + numbers and saves the headline figure to figure.png.

The honest, bulletproof claim (validated on REAL human data, Wang et al. PNAS 2014):
  1. Humans show a real question-ORDER EFFECT (asking order changes the answer).
  2. A classical (Bayesian) model is STRUCTURALLY INCAPABLE of any order effect.
  3. The real data satisfies the parameter-free QQ-EQUALITY (q = -0.003) — a
     quantitative identity that QUANTUM probability predicts a priori and that
     classical probability gives no reason to expect. That is the killer evidence:
     not a fitted number, a parameter-free prediction confirmed on real humans.

Honesty note: a single-qubit (non-degenerate) quantum model reproduces the
*structure* (an order effect + the QQ-equality for ANY parameters) but does not
fit the full joint magnitudes of this particular survey better than classical —
Boyer-Kassem et al. (2015) show this experiment needs a higher-dimensional
(degenerate) model to fit exactly. The headline win is the parameter-free
QQ-equality, not a fit-quality contest.
"""

import numpy as np

import data as dataset
import quantum_model as qm
import classical_model as cm
import plot as plotter


def hr(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main():
    data = dataset.human_data()
    qa, qb = data["questions"]["A"], data["questions"]["B"]

    hr("THE QUESTIONS (asked in randomised order)")
    print(f"  A: {qa}")
    print(f"  B: {qb}")
    print(f"  data: {data.get('source', 'n/a')}")

    obs = dataset.observed_order_effect(data)
    obs_qq = dataset.observed_qq(data)
    hr("WHAT THE HUMANS DID (real data)")
    pB_after_A = data["AB"]["yy"] + data["AB"]["ny"]
    pB_first = data["BA"]["yy"] + data["BA"]["yn"]
    print(f"  p(B=yes) when A asked first : {pB_after_A:.3f}")
    print(f"  p(B=yes) when B asked first : {pB_first:.3f}")
    print(f"  => ORDER EFFECT             : {obs:+.3f}  (order changes the answer!)")
    print(f"  => observed QQ-equality q   : {obs_qq:+.3f}  (≈ 0 — the PNAS 2014 signature)")

    # ---- Classical baseline ----
    c_joint = cm.fit(data)
    c_pred = cm.predict(c_joint)
    hr("CLASSICAL (BAYESIAN) MODEL — the one that fails")
    print(f"  predicted order effect : {c_pred['order_effect_B']:+.3f}  "
          f"(ZERO — it structurally cannot produce one)")
    print("  predicted QQ-equality  : undefined as a prediction — classical theory")
    print("    gives NO a-priori reason for q = 0; the quantum framework does.")

    # ---- Quantum model: the parameter-free QQ-equality ----
    hr("QUANTUM MODEL — the parameter-free prediction")
    rng = np.random.default_rng(0)
    qqs = []
    for _ in range(1000):
        alpha, beta = rng.uniform(0, np.pi, size=2)
        qqs.append(qm.predict(alpha, beta)["qq"])
    qqs = np.array(qqs)
    print(f"  QQ-equality over 1000 RANDOM parameter pairs:")
    print(f"    max |q| = {np.abs(qqs).max():.2e}   (≡ 0 for every parameter — parameter-free)")
    print("  The quantum model produces a nonzero order effect AND forces q = 0,")
    print("  with NO tuning. The real human data lands on that same q ≈ 0 line.")

    hr("THE VERDICT")
    print("  • Humans show a real, large order effect (|Δ| = "
          f"{abs(obs):.3f}).")
    print("  • Classical probability cannot represent ANY order effect (Δ ≡ 0).")
    print(f"  • Yet the order-sensitive data obeys the parameter-free QQ-equality "
          f"(q = {obs_qq:+.3f} ≈ 0)")
    print("    — predicted a priori by quantum probability, unexplained by classical.")
    print("\n  Newton: 'I can't predict the madness of people.'")
    print("  Feynman: 'You don't understand quantum mechanics.'")
    print("  Resolution: the madness IS quantum.\n")

    plotter.headline_figure(data, obs, obs_qq, c_pred, path="figure.png")


if __name__ == "__main__":
    main()
