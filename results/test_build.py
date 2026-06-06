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


def test_cognition_figure_regenerates(tmp_path):
    import sys, os
    qi = os.path.join(os.path.dirname(__file__), "..", "quantum_investor")
    sys.path.insert(0, os.path.abspath(qi))
    import importlib
    import data as dataset            # noqa
    import classical_model as cm      # noqa
    import plot as plotter            # noqa
    importlib.reload(plotter)
    d = dataset.human_data()
    obs = dataset.observed_order_effect(d)
    qq = dataset.observed_qq(d)
    c_pred = cm.predict(cm.fit(d))
    out = tmp_path / "fig.png"
    plotter.headline_figure(d, obs, qq, c_pred, path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_manifest_has_three_tracks():
    from results import manifest
    keys = {t["key"] for t in manifest.TRACKS}
    assert keys == {"cognition", "pricer", "triage"}


def test_each_track_has_required_fields():
    from results import manifest
    for t in manifest.TRACKS:
        for field in ("key", "title", "claim", "figure", "table", "prose", "provenance"):
            assert field in t, f"{t['key']} missing {field}"
        assert "header" in t["table"] and "rows" in t["table"]


def test_summary_has_headline_numbers():
    from results import manifest
    assert len(manifest.SUMMARY["headlines"]) == 3
