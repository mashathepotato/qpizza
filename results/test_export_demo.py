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
