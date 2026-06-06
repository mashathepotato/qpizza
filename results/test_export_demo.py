from results import export_demo_data as ex

# small/fast params; M=4, seeds=2 keeps the QAE runs quick but still descending
FAST = dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)


def test_build_convergence_has_real_descending_series_and_slopes():
    conv = ex.build_convergence(M=4, seeds=2, **FAST)
    assert conv["ground_truth"] > 0
    assert conv["axis_label"].lower().startswith("oracle queries")
    for method in ("classical_mc", "qae"):
        pts = conv["series"][method]
        assert len(pts) >= 2
        assert all(p["x"] > 0 and p["y"] > 0 for p in pts)
        assert pts == sorted(pts, key=lambda p: p["x"])  # sorted by budget
    # classical MC log-log slope ~ -1/2 (error ~ 1/sqrt(N))
    assert -0.75 <= conv["slopes"]["classical_mc"] <= -0.25
    # QAE descends faster than MC (slope more negative) OR is flagged as saturated-theory
    qae_saturated = any("saturat" in n for n in conv["notes"]["qae"])
    assert qae_saturated or conv["slopes"]["qae"] < conv["slopes"]["classical_mc"]


import json
import os


def test_build_prices_carries_path_and_tree():
    series_meta = ({"dates": ["a", "b", "c"], "closes": [4.0, 4.2, 4.1],
                    "S0": 4.1, "sigma": 0.3, "r": 0.03}, {"source": "synthetic"})
    p = ex.build_prices(series_meta, K=4.1, T=1.0, M_paths=3)
    assert p["closes"] == [4.0, 4.2, 4.1] and p["source"] == "synthetic"
    assert p["tree"]["M"] == 3
    # 2^M terminal values + matching path probabilities that sum to 1
    assert len(p["tree"]["terminal_values"]) == 8
    assert len(p["tree"]["path_probs"]) == 8
    assert abs(sum(p["tree"]["path_probs"]) - 1.0) < 1e-9


def test_hardware_placeholder_is_pending():
    hw = ex.hardware_placeholder()
    assert hw["status"] == "pending" and hw["backend"] == "Q50"


def test_main_writes_three_jsons_and_preserves_real_hardware(tmp_path):
    # a real hardware result already present must NOT be overwritten
    hw = tmp_path / "hardware.json"
    hw.write_text(json.dumps({"status": "done", "price": 2.78}))
    ex.main(out_dir=str(tmp_path), allow_network=False, M_conv=4, seeds=2)
    for name in ("prices.json", "convergence.json", "hardware.json"):
        assert (tmp_path / name).exists()
    assert json.loads(hw.read_text())["status"] == "done"  # preserved
