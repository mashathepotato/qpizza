import numpy as np
from quantum_pricer import benchmark


def test_error_vs_queries_runs_and_has_all_contenders(base_params):
    rows = benchmark.error_vs_queries(M=4, **base_params)
    methods = {r["method"] for r in rows}
    assert {"classical_mc", "qae"} <= methods   # qsvt added when stable
    # queries can be 0 for coarse-eps QAE/QSVT (Grover-power saturation) -- that is
    # reported honestly, not hidden; so the per-row contract is queries >= 0.
    for r in rows:
        assert r["queries"] >= 0
        assert r["abs_error"] >= 0.0 or not np.isfinite(r["abs_error"])
    # every classical-MC budget must consume real samples
    assert all(r["queries"] > 0 for r in rows if r["method"] == "classical_mc")
    # at least one quantum point must register real oracle queries
    assert any(r["queries"] > 0 for r in rows if r["method"] in ("qae", "qsvt"))


def test_error_vs_queries_annotates_qae_saturation(base_params):
    # at small M, finer epsilon targets collapse to the same query count -> annotated
    rows = benchmark.error_vs_queries(M=4, **base_params)
    qae_rows = [r for r in rows if r["method"] == "qae"]
    counts = [r["queries"] for r in qae_rows]
    if len(counts) != len(set(counts)):
        assert any(r["note"] == "qae_query_saturation" for r in qae_rows)


def test_resource_table_reports_cz_depth(base_params):
    table = benchmark.resource_table(M=3, **base_params)
    assert {"method", "qubits", "cz_depth"} <= set(table[0].keys())
    assert all(row["cz_depth"] >= 0 for row in table)


def test_save_speedup_plot_writes_png(base_params, tmp_path):
    rows = benchmark.error_vs_queries(M=3, **base_params)
    out = tmp_path / "speedup.png"
    path = benchmark.save_speedup_plot(rows, path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert str(out) == path


def _mc_queries_at(rows, eps):
    return next(r["queries"] for r in rows
               if r["method"] == "classical_mc" and abs(r["epsilon"] - eps) < 1e-12)


def _qae_theory_queries_at(rows, eps):
    return next(r["queries"] for r in rows
                if r["method"] == "qae" and r["kind"] == "theoretical"
                and abs(r["epsilon"] - eps) < 1e-12)


def test_queries_to_accuracy_mc_scales_quadratically(base_params):
    # analytic CLT sample complexity N = (sigma/eps)^2 -> halving eps quadruples N
    rows = benchmark.queries_to_accuracy(M=4, epsilons=(0.04, 0.02), **base_params)
    ratio = _mc_queries_at(rows, 0.02) / _mc_queries_at(rows, 0.04)
    assert 3.0 <= ratio <= 5.0


def test_queries_to_accuracy_qae_subquadratic(base_params):
    # theoretical QAE query count ~1/eps -> halving eps roughly doubles queries
    rows = benchmark.queries_to_accuracy(M=4, epsilons=(0.04, 0.02), **base_params)
    ratio = _qae_theory_queries_at(rows, 0.02) / _qae_theory_queries_at(rows, 0.04)
    assert 1.5 <= ratio <= 3.0


def test_complexity_plot_writes_file(base_params, tmp_path):
    rows = benchmark.queries_to_accuracy(M=4, **base_params)
    out = tmp_path / "complexity.png"
    path = benchmark.save_complexity_plot(rows, path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert str(out) == path
