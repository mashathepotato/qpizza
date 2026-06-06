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
    # only the optional localhost live-server probe may reference a URL; no CDNs
    urls = [l for l in h.splitlines() if "http://" in l or "https://" in l]
    assert all("localhost:5057" in l for l in urls), urls


def test_html_has_navigation_and_cockpit_toggle():
    h = _html()
    assert "ArrowRight" in h and "ArrowLeft" in h
    assert "'e'" in h.lower() or '"e"' in h.lower()  # E toggles cockpit
    assert "data not exported yet" in h.lower()       # degradation notice


def test_acts_1_2_have_madness_and_loading_content():
    h = _html()
    assert "madness of people" in h.lower()
    assert "2^M" in h or "2<sup>M" in h          # 2^M paths messaging
    assert "no loading oracle" in h.lower() or "no distribution-loading" in h.lower()
    assert "renderAct0" in h and "renderAct1" in h


def test_acts_3_4_have_race_and_speedup_with_honesty_labels():
    h = _html()
    assert "renderAct2" in h and "renderAct3" in h
    assert "ground truth" in h.lower()
    # axis must be queries/samples, never time
    assert "queries" in h.lower()
    assert "not wall-clock" in h.lower() or "not time" in h.lower()
    # QSVT honesty: plateau / floor mentioned
    assert "floor" in h.lower() or "plateau" in h.lower()


def test_act5_and_cockpit_and_hardware_badge():
    h = _html()
    assert "renderAct4" in h and "renderCockpit" in h
    assert "q50" in h.lower()
    assert "pending" in h.lower()       # badge default state
    assert "beat the textbook" in h.lower() or "vs sota" in h.lower() or "oracle-qae" in h.lower()
