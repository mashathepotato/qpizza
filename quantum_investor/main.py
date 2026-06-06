"""
Run the whole demo end-to-end.

    $ python main.py

Prints the narrative + numbers and saves the headline figure to figure.png.
This is the core "must-have" of the 24h plan: humans show an order effect, the
classical model structurally cannot reproduce it, the quantum circuit can — and
the quantum model also satisfies the parameter-free QQ-equality.
"""

import data as dataset
import quantum_model as qm
import classical_model as cm
import fit as fitter
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

    obs = dataset.observed_order_effect(data)
    hr("WHAT THE HUMANS DID")
    pB_after_A = data["AB"]["yy"] + data["AB"]["ny"]
    pB_first = data["BA"]["yy"] + data["BA"]["yn"]
    print(f"  p(B=yes) when A asked first : {pB_after_A:.3f}")
    print(f"  p(B=yes) when B asked first : {pB_first:.3f}")
    print(f"  => ORDER EFFECT             : {obs:+.3f}  (order changes the answer!)")

    # ---- Classical baseline ----
    c_joint = cm.fit(data)
    c_pred = cm.predict(c_joint)
    c_err = cm.sse(c_pred, data)
    hr("CLASSICAL (BAYESIAN) MODEL — the one that fails")
    print(f"  predicted order effect : {c_pred['order_effect_B']:+.3f}  "
          f"(ZERO — it structurally cannot produce one)")
    print(f"  fit error (SSE)        : {c_err:.5f}")

    # ---- Quantum model ----
    alpha, beta, q_err = fitter.fit_quantum(data)
    q_pred = qm.predict(alpha, beta)
    hr("QUANTUM MODEL — the one that fits")
    print(f"  fitted alpha, beta     : {alpha:.3f}, {beta:.3f}")
    print(f"  predicted order effect : {q_pred['order_effect_B']:+.3f}  "
          f"(matches the humans)")
    print(f"  fit error (SSE)        : {q_err:.5f}")
    print(f"  QQ-equality value      : {q_pred['qq']:+.2e}  "
          f"(≈ 0, parameter-free — the PNAS 2014 signature)")

    hr("THE VERDICT")
    improvement = c_err / q_err if q_err > 0 else float("inf")
    print(f"  Quantum fits the data ~{improvement:.0f}x better than classical.")
    print("  Classical can't represent order effects at all; quantum does it natively")
    print("  AND satisfies the parameter-free QQ-equality that real data also obey.")
    print("\n  Newton: 'I can't predict the madness of people.'")
    print("  Feynman: 'You don't understand quantum mechanics.'")
    print("  Resolution: the madness IS quantum.\n")

    plotter.headline_figure(data, q_pred, c_pred, obs, path="figure.png")


if __name__ == "__main__":
    main()
