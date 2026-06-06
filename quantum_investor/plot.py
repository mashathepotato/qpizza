"""The headline figure: humans show an order effect; classical can't, quantum can."""

import matplotlib
matplotlib.use("Agg")  # save to file, no display needed
import matplotlib.pyplot as plt


def headline_figure(data, q_pred, c_pred, obs_effect, path="figure.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: p(B=yes) in each order ----
    groups = ["A asked first", "B asked first"]
    human = [data["AB"]["yy"] + data["AB"]["ny"], data["BA"]["yy"] + data["BA"]["yn"]]
    classical = [c_pred["pB_yes_after_A"], c_pred["pB_yes_first"]]
    quantum = [q_pred["pB_yes_after_A"], q_pred["pB_yes_first"]]

    x = range(len(groups))
    w = 0.26
    ax1.bar([i - w for i in x], human, w, label="Humans (data)", color="#dfe7ff", edgecolor="#333")
    ax1.bar([i for i in x], classical, w, label="Classical model", color="#ff6b81")
    ax1.bar([i + w for i in x], quantum, w, label="Quantum model", color="#46d39a")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(groups)
    ax1.set_ylabel("p(B = yes)")
    ax1.set_title("Does asking order change the answer?")
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.annotate("classical predicts\nthe SAME bar twice\n(no order effect)",
                 xy=(1, classical[1]), xytext=(0.35, 0.88),
                 fontsize=9, color="#b03048",
                 arrowprops=dict(arrowstyle="->", color="#b03048"))

    # ---- Right: order-effect magnitude ----
    labels = ["Humans", "Classical", "Quantum"]
    effects = [abs(obs_effect), abs(c_pred["order_effect_B"]), abs(q_pred["order_effect_B"])]
    colors = ["#9fb0d4", "#ff6b81", "#46d39a"]
    ax2.bar(labels, effects, color=colors, edgecolor="#333")
    ax2.set_ylabel("|order effect| on p(B = yes)")
    ax2.set_title("The gap classical can't close")
    for i, v in enumerate(effects):
        ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle("The Madness of People Is Quantum — investor question-order effect",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    print(f"[saved] {path}")
    return path
