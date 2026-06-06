import os

HTML = os.path.join(os.path.dirname(__file__), "demo_animation.html")


def _html():
    with open(HTML) as fh:
        return fh.read()


def test_html_is_self_contained_and_loads_all_four_jsons():
    h = _html()
    assert h.lstrip().lower().startswith("<!doctype html")
    for f in ("results.json", "prices.json", "convergence.json", "hardware.json"):
        assert f in h, f"frontend must fetch {f}"
    # no framework / CDN dependency -> demo survives dead wifi
    assert "http://" not in h and "https://" not in h


def test_html_has_navigation_and_cockpit_toggle():
    h = _html()
    assert "ArrowRight" in h and "ArrowLeft" in h
    assert "'e'" in h.lower() or '"e"' in h.lower()  # E toggles cockpit
    assert "data not exported yet" in h.lower()       # degradation notice
