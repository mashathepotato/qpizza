import os

from results import build_demo

_HERE = os.path.dirname(os.path.abspath(__file__))


def test_template_is_self_contained_and_has_injection_points():
    tpl = open(os.path.join(_HERE, "demo_template.html")).read()
    assert tpl.lstrip().lower().startswith("<!doctype html")
    # injection markers the build script fills
    assert "/*__DATA__*/ null" in tpl and "/*__FIGS__*/ {}" in tpl
    # no external CDN/framework dependency
    assert "http://" not in tpl and "https://" not in tpl


def test_build_bakes_data_and_figures_into_one_file(tmp_path):
    out = tmp_path / "demo.html"
    # read the real json + figures from results/, write the baked file to tmp
    info = build_demo.build(out_dir=_HERE, figs_dir=os.path.join(_HERE, "figures"),
                            out_html=str(out))
    assert out.exists()
    h = out.read_text()
    assert h.lstrip().lower().startswith("<!doctype html")
    # figures are embedded as data URIs (self-contained)
    assert "data:image/png;base64," in h
    assert info["n_figs"] >= 4
    # DATA is no longer the null placeholder — real numbers are inlined
    assert "/*__DATA__*/ null" not in h
    assert '"ground_truth"' in h or '"forecast"' in h
    # still self-contained: no remote URLs
    assert "http://" not in h and "https://" not in h


def test_build_writes_demo_html_in_results():
    info = build_demo.build()
    assert os.path.exists(info["out_html"])
    assert info["out_html"].endswith("results/demo.html")
