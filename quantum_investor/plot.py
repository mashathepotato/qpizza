"""
The headline figure (honest, QQ-equality framing), dense technical-report style:
  Left  — humans show an order effect; the classical model predicts the SAME bar
          twice and so cannot represent it.
  Right — the order effect is real and LARGE, yet the parameter-free QQ-equality
          combination of the same data vanishes (q ~ 0): the quantum signature.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the shared repo-wide style (results/style.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from results import style


def headline_figure(data, obs_effect, obs_qq, c_pred, path="figure.png"):
    style.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: p(B=yes) in each order - humans vs classical ----
    groups = ["A asked first", "B asked first"]
    human = [data["AB"]["yy"] + data["AB"]["ny"], data["BA"]["yy"] + data["BA"]["yn"]]
    classical = [c_pred["pB_yes_after_A"], c_pred["pB_yes_first"]]

    x = range(len(groups))
    w = 0.34
    b1 = ax1.bar([i - w / 2 for i in x], human, w, label="Humans (real data)",
                 color=style.PALETTE["quantum"], edgecolor=style.PALETTE["ink"])
    b2 = ax1.bar([i + w / 2 for i in x], classical, w, label="Classical model",
                 color=style.PALETTE["classical"], edgecolor=style.PALETTE["ink"])
    for bars in (b1, b2):
        for rect in bars:
            ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                     f"{rect.get_height():.3f}", ha="center", fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(groups)
    ax1.set_ylabel("p(B = yes)  [probability]")
    ax1.set_title("Asking order changes the human answer")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 1)
    ax1.annotate("classical predicts\nthe SAME value twice\n(no order effect)",
                 xy=(1 + w / 2, classical[1]), xytext=(0.30, 0.90),
                 fontsize=9, color=style.PALETTE["classical"],
                 arrowprops=dict(arrowstyle="->", color=style.PALETTE["classical"]))
    # bracket between the two HUMAN bars: the order effect itself
    top = max(human) + 0.10
    ax1.annotate("", xy=(0 - w / 2, top), xytext=(1 - w / 2, top),
                 arrowprops=dict(arrowstyle="<->", color=style.PALETTE["quantum"], lw=1.4))
    ax1.text(1 - w / 2 + 0.06, top,
             f"$\\Delta$ = {human[1] - human[0]:+.3f}",
             ha="left", va="center", fontsize=9.5, color=style.PALETTE["quantum"],
             fontweight="bold")

    # ---- Right: order effect is large, but the QQ combination ~ 0 ----
    labels = ["|order effect|\n(real, large)", "|QQ-equality q|\n(~ 0, parameter-free)"]
    vals = [abs(obs_effect), abs(obs_qq)]
    colors = [style.PALETTE["muted"], style.PALETTE["accent"]]
    ax2.bar(labels, vals, color=colors, edgecolor=style.PALETTE["ink"])
    ax2.set_ylabel("magnitude [probability units]")
    ax2.set_title("The parameter-free quantum signature")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=11)
    ax2.set_ylim(0, max(vals) * 1.30 + 0.02)
    ax2.annotate("quantum predicts THIS = 0\nfor any parameters;\n"
                 "PNAS: chi^2(1)=0.01, p=0.91",
                 xy=(1, vals[1]), xytext=(0.45, max(vals) * 0.7),
                 fontsize=9, color=style.PALETTE["accent"],
                 arrowprops=dict(arrowstyle="->", color=style.PALETTE["accent"]))
    if abs(obs_qq) > 0:
        ratio = abs(obs_effect) / abs(obs_qq)
        ax2.text(0.5, max(vals) * 1.12,
                 f"~{ratio:.0f}$\\times$ smaller — with zero fitted parameters",
                 ha="center", fontsize=10, color=style.PALETTE["ink"],
                 style="italic")

    fig.suptitle("The Madness of People Is Quantum - real question-order data obeys "
                 "the parameter-free QQ-equality", fontsize=13, fontweight="bold")
    style.caption(fig, "Left: humans differ by order; classical model is order-blind. "
                       "Right: the same data's QQ-equality combination vanishes.")
    style.provenance(fig, "Wang et al., PNAS 111:9431 (2014); Gallup 1997, ~1000 US adults")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")
    return path
