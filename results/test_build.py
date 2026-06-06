import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results import style


def test_palette_has_core_roles():
    for role in ("quantum", "classical", "accent", "muted"):
        assert role in style.PALETTE
        assert style.PALETTE[role].startswith("#")


def test_apply_style_sets_dpi():
    style.apply_style()
    assert matplotlib.rcParams["savefig.dpi"] >= 130


def test_table_image_writes_png(tmp_path):
    out = tmp_path / "t.png"
    style.table_image(
        header=["metric", "value"],
        rows=[["q", "-0.003"], ["p", "0.91"]],
        path=str(out),
        title="demo",
    )
    assert out.exists() and out.stat().st_size > 0


def test_caption_and_provenance_do_not_raise():
    style.apply_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    style.caption(fig, "a caption")
    style.provenance(fig, "src: test")
    plt.close(fig)
