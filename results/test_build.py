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


def test_embed_roundtrips(tmp_path):
    from results import build_dashboard as b
    out = tmp_path / "x.png"
    from results import style
    style.table_image(["a"], [["1"]], str(out))
    uri = b.embed(out.read_bytes())
    assert uri.startswith("data:image/png;base64,")


def test_collect_missing_returns_none():
    from results import build_dashboard as b
    assert b.collect("/no/such/figure.png") is None


def test_build_emits_html(tmp_path):
    from results import build_dashboard as b
    html_path = tmp_path / "index.html"
    md_path = tmp_path / "RESULTS.md"
    b.main(out_html=str(html_path), out_md=str(md_path),
           figures_dir=str(tmp_path / "figures"))
    assert html_path.exists()
    text = html_path.read_text()
    assert "The Madness of People Is Quantum" in text   # cognition always available
    assert "data:image/png;base64," in text             # at least one embedded figure
    assert md_path.exists() and md_path.stat().st_size > 0


def test_triage_slope_claim_uses_cost_vs_eps_axes():
    """Regression guard: the -1/-2 slopes are COST vs target eps. Stating them as
    'error falls with slope -1 vs -2 in queries' inverts the conclusion (on an
    error-vs-queries plot MC is -0.5, not -2)."""
    from results import manifest
    triage = next(t for t in manifest.TRACKS if t["key"] == "triage")
    claim = triage["claim"].lower()
    assert "error falls" not in claim, "claim describes the wrong axes"
    assert "cost" in claim or "queries" in claim
    assert "eps" in claim  # slopes must be tied to the target-accuracy axis
    # table must state which axis the slope refers to
    header = " ".join(triage["table"]["header"]).lower()
    assert "eps" in header or "cost" in header


def test_results_json_calibration_is_look_ahead_free():
    """If results.json is present, its calibration window must end at/before asof."""
    import json
    path = os.path.join(os.path.dirname(__file__), "results.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip("results.json not built yet")
    with open(path) as fh:
        R = json.load(fh)
    meta = R["meta"]
    if "asof" not in meta:
        import pytest
        pytest.skip("results.json predates the asof protocol")
    if meta.get("calib_end"):
        assert meta["calib_end"] <= meta["asof"], "calibration saw data beyond t0"
    oos = R.get("out_of_sample")
    if oos and "vol_forecast" in oos:
        assert oos["vol_forecast"]["tag"] == "VALIDATED"
        assert oos["realized_payoff"]["tag"] == "ILLUSTRATIVE"
