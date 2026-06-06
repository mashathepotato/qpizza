"""Tests for the live auto-refreshing dashboard.

Kept FAST: allow_network=False (synthetic params, deterministic), run_hw=False
(skip iqm), small M and benchmark grids.
"""
import json

from quantum_pricer import dashboard


def test_collect_state_has_core_fields():
    state = dashboard.collect_state(allow_network=False, run_hw=False, M=3)
    # market params + provenance
    assert "params" in state and {"S0", "sigma", "r"} <= set(state["params"])
    assert "meta" in state and "source" in state["meta"]
    assert "K" in state and "T" in state and "M" in state
    # ground truth
    assert "exact_tree" in state and isinstance(state["exact_tree"], float)
    # at least the classical / fourier / qae route prices
    prices = state["prices"]
    assert "classical_mc" in prices
    assert "fourier" in prices
    assert "qae" in prices
    # resource table
    assert "resource_table" in state and len(state["resource_table"]) >= 1
    # timestamp
    assert "timestamp" in state and isinstance(state["timestamp"], str)


def test_render_html_is_valid_and_references_figures():
    state = dashboard.collect_state(allow_network=False, run_hw=False, M=3)
    html = dashboard.render_html(state, [])
    assert "<html" in html
    assert 'http-equiv="refresh"' in html
    assert "complexity.png" in html
    assert "speedup.png" in html
    # ground-truth price rendered somewhere
    assert f"{state['exact_tree']:.4f}" in html


def test_dashboard_build_writes_files(tmp_path, monkeypatch):
    # redirect output paths into tmp so we don't clobber the committed dashboard
    monkeypatch.setattr(dashboard, "HTML_PATH", str(tmp_path / "dashboard.html"))
    monkeypatch.setattr(dashboard, "STATE_PATH", str(tmp_path / "dashboard_state.json"))
    monkeypatch.setattr(dashboard, "LOG_PATH", str(tmp_path / "dashboard_log.jsonl"))
    monkeypatch.setattr(dashboard, "COMPLEXITY_PNG", str(tmp_path / "complexity.png"))
    monkeypatch.setattr(dashboard, "SPEEDUP_PNG", str(tmp_path / "speedup.png"))

    state = dashboard.build_once(allow_network=False, run_hw=False, M=3)
    html_file = tmp_path / "dashboard.html"
    state_file = tmp_path / "dashboard_state.json"
    assert html_file.exists() and html_file.stat().st_size > 0
    assert "<html" in html_file.read_text()
    assert state_file.exists()
    loaded = json.loads(state_file.read_text())
    assert "prices" in loaded and "timestamp" in loaded
    assert isinstance(state, dict)
