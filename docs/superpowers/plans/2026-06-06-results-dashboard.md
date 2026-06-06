# Results Dashboard + Figure Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `results/` folder whose `build_dashboard.py` regenerates/collects every track's figures into one self-contained, dense technical-report HTML dashboard, and improve every figure in the repo through one shared visual style.

**Architecture:** A declarative `results/manifest.py` lists each track (title, claim, figure path, key-number table, prose, provenance). `results/style.py` provides one matplotlib look (palette, rcParams, caption/provenance/table helpers) used by every figure. `results/build_dashboard.py` regenerates the cognition figure, collects the pricer + triage figures from their declared paths, base64-embeds them, and emits `results/index.html` + `results/RESULTS.md`. Missing sources degrade to placeholders, never crash.

**Tech Stack:** Python 3, matplotlib (Agg), base64, pytest. No server, no external assets.

> **Branch-layout note (2026-06-06):** the quantum-pricer track has merged to `main` (`quantum_pricer/`), so its figures are regenerated in place; only **triage** remains a worktree at `.claude/worktrees/triage-lab/triage/`. All source locations live in `manifest.py`, so this is a one-file concern.

> **Repo conventions:** existing plotting uses `matplotlib.use("Agg")`, `fig.savefig(path, dpi=...)`. Existing dashboards (`quantum_pricer/dashboard.py`, triage `dashboard.py`) embed PNGs as `data:image/png;base64,{b64encode(buf).decode()}`. Follow these patterns. Run tests with the repo venv: `quantum_investor/.venv/bin/python` (has numpy+matplotlib) or `python3` if matplotlib is on the system path.

---

## File Structure

| File | Responsibility |
|---|---|
| `results/style.py` (create) | Shared matplotlib style: `PALETTE`, `apply_style()`, `caption()`, `provenance()`, `table_image()`. Pure, importable standalone. |
| `results/manifest.py` (create) | `TRACKS` list + `SUMMARY` dict (the 3 headline numbers + framing). Single source of truth for content + source paths. |
| `results/build_dashboard.py` (create) | Orchestrator: `regenerate_cognition()`, `collect()`, `embed()`, `render_html()`, `write_results_md()`, `main()`. |
| `results/test_build.py` (create) | pytest checks: embed round-trips, collect handles missing, build emits non-empty HTML with a section per available track. |
| `results/__init__.py` (create) | Makes `results` importable (empty). |
| `quantum_investor/plot.py` (modify) | Restyle headline figure through `results.style`; add value labels + χ² annotation. |
| `quantum_pricer/benchmark.py` (modify) | Restyle `save_complexity_plot` / `save_speedup_plot_rms` / `save_depth_crossover_plot` through shared style. |
| `.claude/worktrees/triage-lab/triage/style.py` (create, vendored copy) + triage plot code (modify) | Restyle `qae_scaling` figure through vendored style. |
| `results/figures/` (build output) | Collected/regenerated PNGs. |
| `results/index.html`, `results/RESULTS.md` (build output) | The dashboard + text rollup. |

---

## Task 1: Package skeleton + shared style module

**Files:**
- Create: `results/__init__.py`
- Create: `results/style.py`
- Test: `results/test_build.py`

- [ ] **Step 1: Create the package marker**

Create `results/__init__.py` (empty file):

```python
```

- [ ] **Step 2: Write the failing test for the style module**

Create `results/test_build.py`:

```python
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `quantum_investor/.venv/bin/python -m pytest results/test_build.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'results.style'` (or import error).

- [ ] **Step 4: Implement `results/style.py`**

Create `results/style.py`:

```python
"""One consistent dense technical-report look for every figure in the repo.

Pure helpers, no I/O except explicit save paths. Importable standalone so the
same module can be vendored into worktrees that build their own figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colorblind-safe, high-contrast, reused across all tracks for visual unity.
PALETTE = {
    "quantum":   "#2a6f97",   # quantum / "our" result
    "classical": "#c44536",   # classical baseline
    "accent":    "#2a9d8f",   # the killer number (q, speedup)
    "muted":     "#8d99ae",   # secondary bars / reference lines
    "ink":       "#22223b",   # text / axes
    "grid":      "#d7d9e0",
}


def apply_style():
    """Set global matplotlib rcParams for a dense technical-report look."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": PALETTE["ink"],
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def caption(fig, text):
    """Italic footnote line under a figure (method / interpretation)."""
    fig.text(0.5, -0.02, text, ha="center", va="top",
             fontsize=8.5, style="italic", color=PALETTE["ink"], wrap=True)


def provenance(fig, text):
    """Small bottom-right stamp (dataset / citation / build note)."""
    fig.text(0.995, 0.005, text, ha="right", va="bottom",
             fontsize=7.5, color=PALETTE["muted"])


def table_image(header, rows, path, title=None, figsize=(6, 0.5)):
    """Render a results table to a standalone PNG for dashboard embedding."""
    apply_style()
    n = len(rows) + 1
    fig, ax = plt.subplots(figsize=(figsize[0], 0.45 * n + (0.4 if title else 0)))
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", pad=10)
    tbl = ax.table(cellText=rows, colLabels=header, loc="center",
                   cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if r == 0:
            cell.set_facecolor(PALETTE["quantum"])
            cell.set_text_props(color="white", fontweight="bold")
    fig.savefig(path)
    plt.close(fig)
    return path
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `quantum_investor/.venv/bin/python -m pytest results/test_build.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add results/__init__.py results/style.py results/test_build.py
git commit -m "feat(results): shared figure style module + tests"
```

---

## Task 2: Restyle the cognition headline figure

**Files:**
- Modify: `quantum_investor/plot.py`
- Test: `results/test_build.py` (add one test)

- [ ] **Step 1: Add a failing test that the restyled figure regenerates**

Append to `results/test_build.py`:

```python
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
```

- [ ] **Step 2: Run it to verify current behavior (baseline)**

Run: `quantum_investor/.venv/bin/python -m pytest results/test_build.py::test_cognition_figure_regenerates -v`
Expected: PASS already (the current `headline_figure` works) — this guards against regressions while restyling.

- [ ] **Step 3: Restyle `quantum_investor/plot.py`**

Replace the body of `headline_figure` to import the shared style and apply the palette, value labels, a χ² annotation, and a caption/provenance. Full replacement of `quantum_investor/plot.py`:

```python
"""
The headline figure (honest, QQ-equality framing), dense technical-report style:
  Left  — humans show an order effect; the classical model predicts the SAME bar
          twice and so cannot represent it.
  Right — the order effect is real and LARGE, yet the parameter-free QQ-equality
          combination of the same data vanishes (q ~ 0): the quantum signature.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the shared repo-wide style (results/style.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from results import style


def headline_figure(data, obs_effect, obs_qq, c_pred, path="figure.png"):
    style.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: p(B=yes) in each order - humans vs classical ----
    groups = ["A asked first", "B asked first"]
    human = [data["AB"]["yy"] + data["AB"]["ny"], data["BA"]["yy"] + data["BA"]["yn"]]
    classical = [c_pred["pB_yes_after_A"], c_pred["pB_yes_first"]]

    x = range(len(groups))
    w = 0.34
    b1 = ax1.bar([i - w / 2 for i in x], human, w, label="Humans (real data)",
                 color=style.PALETTE["quantum"], edgecolor=style.PALETTE["ink"])
    b2 = ax1.bar([i + w / 2 for i in x], classical, w, label="Classical model",
                 color=style.PALETTE["classical"], edgecolor=style.PALETTE["ink"])
    for bars in (b1, b2):
        for rect in bars:
            ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                     f"{rect.get_height():.3f}", ha="center", fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(groups)
    ax1.set_ylabel("p(B = yes)  [probability]")
    ax1.set_title("Asking order changes the human answer")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 1)
    ax1.annotate("classical predicts\nthe SAME value twice\n(no order effect)",
                 xy=(1 + w / 2, classical[1]), xytext=(0.30, 0.90),
                 fontsize=9, color=style.PALETTE["classical"],
                 arrowprops=dict(arrowstyle="->", color=style.PALETTE["classical"]))

    # ---- Right: order effect is large, but the QQ combination ~ 0 ----
    labels = ["|order effect|\n(real, large)", "|QQ-equality q|\n(~ 0, parameter-free)"]
    vals = [abs(obs_effect), abs(obs_qq)]
    colors = [style.PALETTE["muted"], style.PALETTE["accent"]]
    ax2.bar(labels, vals, color=colors, edgecolor=style.PALETTE["ink"])
    ax2.set_ylabel("magnitude [probability units]")
    ax2.set_title("The parameter-free quantum signature")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=11)
    ax2.set_ylim(0, max(vals) * 1.30 + 0.02)
    ax2.annotate("quantum predicts THIS = 0\nfor any parameters;\n"
                 "PNAS: chi^2(1)=0.01, p=0.91",
                 xy=(1, vals[1]), xytext=(0.45, max(vals) * 0.7),
                 fontsize=9, color=style.PALETTE["accent"],
                 arrowprops=dict(arrowstyle="->", color=style.PALETTE["accent"]))

    fig.suptitle("The Madness of People Is Quantum - real question-order data obeys "
                 "the parameter-free QQ-equality", fontsize=13, fontweight="bold")
    style.caption(fig, "Left: humans differ by order; classical model is order-blind. "
                       "Right: the same data's QQ-equality combination vanishes.")
    style.provenance(fig, "Wang et al., PNAS 111:9431 (2014); Gallup 1997, ~1000 US adults")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")
    return path
```

- [ ] **Step 4: Run the regression test + regenerate the real figure**

Run: `quantum_investor/.venv/bin/python -m pytest results/test_build.py::test_cognition_figure_regenerates -v`
Expected: PASS.
Then regenerate the committed figure: `cd quantum_investor && ../quantum_investor/.venv/bin/python main.py && cd ..`
Expected: prints the narrative and `[saved] figure.png`.

- [ ] **Step 5: Commit**

```bash
git add quantum_investor/plot.py quantum_investor/figure.png results/test_build.py
git commit -m "feat(cognition): restyle headline figure via shared style"
```

---

## Task 3: Manifest of tracks

**Files:**
- Create: `results/manifest.py`
- Test: `results/test_build.py` (add tests)

- [ ] **Step 1: Write failing tests for the manifest**

Append to `results/test_build.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `quantum_investor/.venv/bin/python -m pytest results/test_build.py -k manifest -v`
Expected: FAIL — `No module named 'results.manifest'`.

- [ ] **Step 3: Implement `results/manifest.py`**

Create `results/manifest.py`:

```python
"""Single source of truth for dashboard content + figure source locations.

Each track declares where its figure is, the key-number table to show, the prose
interpretation, and provenance. build_dashboard.py reads ONLY this for content.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(*parts):
    return os.path.join(ROOT, *parts)


# Triage still lives in a worktree; pricer + cognition are on main.
_TRIAGE = _p(".claude", "worktrees", "triage-lab", "triage")

TRACKS = [
    {
        "key": "cognition",
        "title": "The Madness of People Is Quantum",
        "claim": "Real question-order data obeys the parameter-free QQ-equality (q = -0.003).",
        "figure": _p("quantum_investor", "figure.png"),
        "regenerate": "cognition",          # build_dashboard re-runs this one
        "table": {
            "header": ["quantity", "value", "meaning"],
            "rows": [
                ["order effect |Δ|", "0.095", "asking order changes the answer (real, large)"],
                ["QQ-equality q", "-0.003", "parameter-free quantum prediction = 0"],
                ["PNAS chi^2(1)", "0.01 (p=0.91)", "data is consistent with q = 0"],
                ["classical order effect", "0.000", "structurally impossible to produce"],
            ],
        },
        "prose": ("Humans answer differently depending on question order; a classical "
                  "(Bayesian) joint distribution is order-blind and cannot represent this. "
                  "Yet the same data satisfies the QQ-equality that quantum probability "
                  "predicts with no fitted parameters. Honest caveat: a single-qubit model "
                  "reproduces the structure, not the full joint fit of this survey."),
        "provenance": "Wang et al., PNAS 111:9431 (2014); Gallup 1997 Clinton/Gore, ~1000 adults.",
    },
    {
        "key": "pricer",
        "title": "Quantum Option Pricer — quadratic Monte-Carlo speedup",
        "claim": "QNDM+QAE reaches accuracy eps in O(1/eps) oracle queries vs MC's O(1/eps^2).",
        "figure": _p("quantum_pricer", "complexity.png"),
        "extra_figures": [
            _p("quantum_pricer", "speedup.png"),
            _p("quantum_pricer", "depth_crossover.png"),
        ],
        "regenerate": None,                 # collected as built by quantum_pricer
        "table": {
            "header": ["route", "query complexity", "depth (IQM CZ)", "note"],
            "rows": [
                ["Classical MC", "O(1/eps^2)", "0", "baseline"],
                ["QNDM Fourier", "O(1/eps^2) shots", "~112 CZ (M=4)", "shallow, exact loading"],
                ["QNDM QAE", "O(1/eps)", "~16 CZ", "quadratic speedup"],
                ["QSVT (novel)", "O(1/eps)", "~2240 CZ", "honest straddle construction"],
            ],
        },
        "prose": ("Loads the full price-path tree into a superposition and reads the fair "
                  "price off it via QNDM phase encoding + amplitude estimation, removing the "
                  "expensive distribution-loading oracle that bottlenecks prior quantum pricers. "
                  "Ground truth = exact CRR binomial tree; Nokia (NOKIA.HE) European/Asian call."),
        "provenance": "Stamatopoulos et al., Quantum 4:291 (2020); Montanaro, Proc. R. Soc. A (2015).",
    },
    {
        "key": "triage",
        "title": "Triage Lab — QAE scaling advantage",
        "claim": "QAE error falls with slope ~ -1 vs classical Monte-Carlo ~ -2 in queries.",
        "figure": os.path.join(_TRIAGE, "plots", "qae_scaling.png"),
        "regenerate": None,
        "table": {
            "header": ["method", "scaling slope", "interpretation"],
            "rows": [
                ["QAE (quantum)", "~ -1", "error ~ O(1/queries): quadratic edge at tight eps"],
                ["Monte-Carlo", "~ -2", "error ~ O(1/sqrt(samples))"],
            ],
        },
        "prose": ("Overnight triage of quantum-finance methods (QAE / QAOA / fraud). The QAE "
                  "scaling curve shows the quadratic edge — but only at tight accuracy; at coarse "
                  "eps the fixed per-round cost lets classical win. Honest, not hyped."),
        "provenance": "Triage-lab worktree REPORT.md; same QAE math as the pricer track.",
    },
]

SUMMARY = {
    "framing": ("Newton could not predict 'the madness of people'; Feynman said classical "
                "intuition fails. Both say markets are non-classical — so we compute with it."),
    "headlines": [
        {"label": "Cognition", "value": "q = -0.003", "sub": "parameter-free QQ-equality holds"},
        {"label": "Pricing", "value": "O(1/eps) vs O(1/eps^2)", "sub": "quadratic MC speedup"},
        {"label": "Triage", "value": "slope -1 vs -2", "sub": "QAE scaling edge"},
    ],
}
```

- [ ] **Step 4: Run to verify pass**

Run: `quantum_investor/.venv/bin/python -m pytest results/test_build.py -k "manifest or summary" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add results/manifest.py results/test_build.py
git commit -m "feat(results): declarative track manifest"
```

---

## Task 4: The build_dashboard orchestrator

**Files:**
- Create: `results/build_dashboard.py`
- Test: `results/test_build.py` (add tests)

- [ ] **Step 1: Write failing tests for collect/embed/build**

Append to `results/test_build.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `quantum_investor/.venv/bin/python -m pytest results/test_build.py -k "embed or collect or emits" -v`
Expected: FAIL — `No module named 'results.build_dashboard'`.

- [ ] **Step 3: Implement `results/build_dashboard.py`**

Create `results/build_dashboard.py`:

```python
"""Build the unified results dashboard.

  python -m results.build_dashboard

Regenerates the cognition figure, collects the pricer + triage figures from the
paths declared in manifest.py, base64-embeds everything into one self-contained
results/index.html (no server, no external assets), and writes results/RESULTS.md.
Missing sources degrade to a placeholder section; the build never crashes.
"""
import base64
import html as _html
import os
import shutil
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

from results import manifest

ROOT = manifest.ROOT
_RESULTS = os.path.join(ROOT, "results")


def embed(png_bytes):
    """bytes -> data URI for inline <img>."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def collect(path):
    """Return PNG bytes at path, or None if absent."""
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()
    return None


def regenerate_cognition():
    """Re-run the quantum_investor pipeline to refresh figure.png. Best-effort."""
    import sys
    qi = os.path.join(ROOT, "quantum_investor")
    sys.path.insert(0, qi)
    try:
        import importlib
        import data as dataset
        import classical_model as cm
        import plot as plotter
        importlib.reload(plotter)
        d = dataset.human_data()
        obs = dataset.observed_order_effect(d)
        qq = dataset.observed_qq(d)
        c_pred = cm.predict(cm.fit(d))
        out = os.path.join(qi, "figure.png")
        plotter.headline_figure(d, obs, qq, c_pred, path=out)
        return out
    except Exception as exc:   # noqa: BLE001 - dashboard must not crash
        print(f"[warn] cognition regenerate failed: {exc}")
        return os.path.join(qi, "figure.png")  # fall back to whatever exists


def _section_html(track, figures_dir):
    """One <section> per track; placeholder when the figure is unavailable."""
    title = _html.escape(track["title"])
    claim = _html.escape(track["claim"])
    prose = _html.escape(track["prose"])
    prov = _html.escape(track["provenance"])

    if track.get("regenerate") == "cognition":
        regenerate_cognition()

    figs = [track["figure"]] + track.get("extra_figures", [])
    imgs = []
    for fp in figs:
        data = collect(fp)
        if data is None:
            continue
        # persist a copy into results/figures/ for archival
        os.makedirs(figures_dir, exist_ok=True)
        shutil.copy(fp, os.path.join(figures_dir, os.path.basename(fp)))
        imgs.append(f'<img src="{embed(data)}" alt="{title}"/>')

    if imgs:
        fig_block = '<div class="figs">' + "".join(imgs) + "</div>"
    else:
        fig_block = ('<div class="missing">Figure unavailable — rebuild the '
                     f'<code>{_html.escape(track["key"])}</code> track, then re-run the dashboard.</div>')

    rows = "".join(
        "<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in r) + "</tr>"
        for r in track["table"]["rows"]
    )
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in track["table"]["header"])
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"

    return f"""
    <section>
      <h2>{title}</h2>
      <p class="claim">{claim}</p>
      {fig_block}
      {table}
      <p class="prose">{prose}</p>
      <p class="prov">{prov}</p>
    </section>"""


def _summary_html():
    s = manifest.SUMMARY
    cards = "".join(
        f'<div class="card"><div class="lab">{_html.escape(h["label"])}</div>'
        f'<div class="val">{_html.escape(h["value"])}</div>'
        f'<div class="sub">{_html.escape(h["sub"])}</div></div>'
        for h in s["headlines"]
    )
    return (f'<div class="summary"><p class="framing">{_html.escape(s["framing"])}</p>'
            f'<div class="cards">{cards}</div></div>')


_CSS = """
:root{--ink:#22223b;--q:#2a6f97;--grid:#d7d9e0;}
*{box-sizing:border-box}body{font-family:Georgia,'Times New Roman',serif;color:var(--ink);
margin:0;background:#f7f7fb;line-height:1.5}
header{background:var(--ink);color:#fff;padding:22px 32px}
header h1{margin:0;font-size:24px}header .ts{opacity:.7;font-size:12px;margin-top:6px}
main{max-width:1100px;margin:0 auto;padding:24px 32px}
.summary{background:#fff;border:1px solid var(--grid);border-radius:10px;padding:18px;margin-bottom:24px}
.framing{font-style:italic;margin:0 0 14px}
.cards{display:flex;gap:14px;flex-wrap:wrap}
.card{flex:1;min-width:220px;border:1px solid var(--grid);border-radius:8px;padding:12px}
.card .lab{font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#888}
.card .val{font-size:22px;font-weight:bold;color:var(--q);margin:4px 0}
.card .sub{font-size:13px;color:#555}
section{background:#fff;border:1px solid var(--grid);border-radius:10px;padding:18px 22px;margin-bottom:22px}
section h2{margin:0 0 6px;color:var(--q)}
.claim{font-weight:bold;margin:0 0 12px}
.figs{display:flex;flex-wrap:wrap;gap:12px;justify-content:center}
.figs img{max-width:100%;border:1px solid var(--grid);border-radius:6px}
.missing{padding:24px;text-align:center;color:#a33;background:#fff5f5;border:1px dashed #e0a0a0;border-radius:6px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid var(--grid);padding:6px 10px;text-align:left}
th{background:var(--q);color:#fff}
.prose{font-size:14px;color:#333}.prov{font-size:12px;color:#888;font-style:italic}
footer{text-align:center;color:#999;font-size:12px;padding:18px}
"""


def render_html(figures_dir):
    sections = "".join(_section_html(t, figures_dir) for t in manifest.TRACKS)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>QHack Results Dashboard</title><style>{_CSS}</style></head>
<body><header><h1>The Madness of People Is Quantum — Results Dashboard</h1>
<div class="ts">Built {ts} · re-run <code>python -m results.build_dashboard</code> to refresh</div></header>
<main>{_summary_html()}{sections}</main>
<footer>Self-contained · figures embedded · OP Pohjola × Junction Helsinki</footer></body></html>"""


def write_results_md(path):
    s = manifest.SUMMARY
    lines = ["# Results rollup", "", s["framing"], ""]
    for h in s["headlines"]:
        lines.append(f"- **{h['label']}** — {h['value']} ({h['sub']})")
    lines.append("")
    for t in manifest.TRACKS:
        lines.append(f"## {t['title']}")
        lines.append(t["claim"])
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main(out_html=None, out_md=None, figures_dir=None):
    out_html = out_html or os.path.join(_RESULTS, "index.html")
    out_md = out_md or os.path.join(_RESULTS, "RESULTS.md")
    figures_dir = figures_dir or os.path.join(_RESULTS, "figures")
    html_text = render_html(figures_dir)
    with open(out_html, "w") as fh:
        fh.write(html_text)
    write_results_md(out_md)
    # status line per track
    for t in manifest.TRACKS:
        figs = [t["figure"]] + t.get("extra_figures", [])
        avail = sum(1 for f in figs if os.path.exists(f))
        print(f"  [{ 'ok ' if avail else 'MISS'}] {t['key']:9s} {avail}/{len(figs)} figures")
    print(f"[saved] {out_html}")
    print(f"[saved] {out_md}")
    return out_html


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `quantum_investor/.venv/bin/python -m pytest results/test_build.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Build the real dashboard and eyeball it**

Run: `quantum_investor/.venv/bin/python -m results.build_dashboard`
Expected: per-track status lines + `[saved] .../results/index.html`. Open `results/index.html` in a browser; confirm summary band + three sections render (triage may show a placeholder if its worktree figure path differs — that is acceptable and expected to degrade gracefully).

- [ ] **Step 6: Commit**

```bash
git add results/build_dashboard.py results/test_build.py results/index.html results/RESULTS.md results/figures
git commit -m "feat(results): unified self-contained dashboard build"
```

---

## Task 5: Restyle the pricer figures (on main)

**Files:**
- Modify: `quantum_pricer/benchmark.py` (functions `save_complexity_plot`, `save_speedup_plot_rms`, `save_depth_crossover_plot`)

- [ ] **Step 1: Read the three plotting functions to learn their axes/labels**

Run: `quantum_investor/.venv/bin/python - <<'PY'`
```python
import re, pathlib
src = pathlib.Path("quantum_pricer/benchmark.py").read_text()
for name in ("save_complexity_plot", "save_speedup_plot_rms", "save_depth_crossover_plot"):
    i = src.index("def " + name)
    print("="*70); print(src[i:i+1400])
PY
```
Expected: prints each function body so you can see the current matplotlib calls.

- [ ] **Step 2: Apply the shared style inside each function**

At the top of `quantum_pricer/benchmark.py`, after its existing imports, add:

```python
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from results import style as _style
```

Then, inside each of `save_complexity_plot`, `save_speedup_plot_rms`, and
`save_depth_crossover_plot`, add `_style.apply_style()` as the FIRST line of the
function body (before `fig, ax = ...`), recolor the quantum series with
`_style.PALETTE["quantum"]`/`["accent"]` and the classical series with
`_style.PALETTE["classical"]`, ensure axis labels carry units (e.g.
`"target accuracy eps"`, `"oracle queries"`), and before each `fig.savefig(...)`
add `_style.provenance(fig, "quantum_pricer/benchmark.py · Nokia NOKIA.HE")`.
Keep the existing `path=` arguments and `savefig` calls unchanged.

- [ ] **Step 3: Regenerate the pricer figures**

Run: `quantum_investor/.venv/bin/python -m quantum_pricer.benchmark` if it has a `__main__`, otherwise regenerate via the demo: `quantum_investor/.venv/bin/python -m quantum_pricer.demo`.
Expected: `complexity.png`, `speedup.png`, `depth_crossover.png` rewritten. (If pricer deps are missing in this venv, install: `quantum_investor/.venv/bin/pip install -r quantum_pricer/requirements.txt`.)

- [ ] **Step 4: Verify the figures changed**

Run: `git status --short quantum_pricer/*.png`
Expected: the three PNGs show as modified.

- [ ] **Step 5: Rebuild dashboard + commit**

```bash
quantum_investor/.venv/bin/python -m results.build_dashboard
git add quantum_pricer/benchmark.py quantum_pricer/*.png results/index.html results/figures
git commit -m "feat(pricer): restyle benchmark figures via shared style"
```

---

## Task 6: Restyle the triage figure (vendored style in worktree)

**Files:**
- Create: `.claude/worktrees/triage-lab/triage/style.py` (vendored copy of `results/style.py`)
- Modify: triage plotting code that writes `qae_scaling.png`

- [ ] **Step 1: Locate the triage code that writes `qae_scaling.png`**

Run: `grep -rn "qae_scaling\|savefig" .claude/worktrees/triage-lab/triage/*.py`
Expected: the file + line that saves the scaling figure (likely `dashboard.py` `_scaling_png` or a plots script). Note the function.

- [ ] **Step 2: Vendor the style module into the worktree**

Run: `cp results/style.py .claude/worktrees/triage-lab/triage/style.py`
Expected: file copied. (Vendored because the worktree is a separate branch and cannot import `results.style` from main.)

- [ ] **Step 3: Apply the style in the triage plotting function**

In the triage file that builds the scaling figure, add `import style` (or `from triage import style`, matching the file's existing import style) and call `style.apply_style()` as the first line of that plotting function; recolor the quantum series with `style.PALETTE["quantum"]` and classical with `style.PALETTE["classical"]`; label axes with units (`"oracle queries / MC samples"`, `"estimation error eps"`); add `style.provenance(fig, "triage-lab · qae_scaling")` before its `savefig`.

- [ ] **Step 4: Regenerate the triage figure**

Run the triage step that produces the plot, using that worktree's venv if present:
`cd .claude/worktrees/triage-lab && (.venv/bin/python -m triage.dashboard || python3 -m triage.dashboard); cd -`
Expected: `triage/plots/qae_scaling.png` rewritten. If the path differs, note the actual output path.

- [ ] **Step 5: Point the manifest at the real triage figure path (if needed)**

If Step 1/4 revealed the figure lives at a path different from `plots/qae_scaling.png`, update the `"figure"` path for the `triage` track in `results/manifest.py` to match. Otherwise no change.

- [ ] **Step 6: Rebuild dashboard, verify triage section renders, commit**

```bash
quantum_investor/.venv/bin/python -m results.build_dashboard
# commit the main-branch changes (manifest tweak if any + rebuilt dashboard)
git add results/manifest.py results/index.html results/figures
git commit -m "feat(triage): restyle scaling figure + wire into dashboard"
# commit the vendored style + plot change inside the worktree branch
cd .claude/worktrees/triage-lab && git add triage/style.py triage/*.py triage/plots/qae_scaling.png && git commit -m "feat(triage): vendor shared style, restyle qae_scaling" ; cd -
```

---

## Task 7: README + final verification

**Files:**
- Create: `results/README.md`
- Modify: `README.md` (root, add a pointer)

- [ ] **Step 1: Write `results/README.md`**

Create `results/README.md`:

```markdown
# results/ — presentation dashboard

One command rebuilds a self-contained HTML dashboard unifying every track's
figures + headline numbers (no server, no external assets).

```bash
python -m results.build_dashboard   # writes results/index.html + results/RESULTS.md
open results/index.html             # (macOS) view it
```

- `style.py` — shared figure style (palette, rcParams, caption/provenance/table helpers).
- `manifest.py` — the ONE file to edit to change dashboard content or figure sources.
- `build_dashboard.py` — regenerates the cognition figure, collects pricer + triage
  figures, embeds them, emits `index.html` + `RESULTS.md`. Missing sources degrade
  to a placeholder; the build never crashes.

To add a result: drop its figure somewhere, add a track dict to `manifest.TRACKS`,
re-run the build.
```

- [ ] **Step 2: Add a pointer from the root README**

Add to the root `README.md` (near the top, after the title) the line:

```markdown
> **Results dashboard:** `python -m results.build_dashboard` → open `results/index.html` for all figures + headline numbers in one place.
```

- [ ] **Step 3: Full test run**

Run: `quantum_investor/.venv/bin/python -m pytest results/test_build.py -v`
Expected: all PASS.

- [ ] **Step 4: Full build + visual confirmation**

Run: `quantum_investor/.venv/bin/python -m results.build_dashboard`
Expected: status lines show `ok` for cognition + pricer (triage `ok` if its figure path resolved). Open `results/index.html`; confirm: summary band with 3 headline cards, three sections each with restyled figure(s) + table + prose, build timestamp present.

- [ ] **Step 5: Commit**

```bash
git add results/README.md README.md
git commit -m "docs(results): document the dashboard build + usage"
```

---

## Self-Review notes (addressed)
- **Spec coverage:** style module (T1), all-figure restyle (T2/T5/T6), manifest (T3), build/collect/embed/HTML/RESULTS.md (T4), graceful-missing (T4 `collect`/placeholder + test), tests (T1–T4, T7), docs (T7). ✓
- **Branch reality:** pricer moved to `main` since the spec; manifest paths updated accordingly; triage remains vendored-in-worktree. ✓
- **Type/name consistency:** `embed`, `collect`, `regenerate_cognition`, `render_html`, `write_results_md`, `main(out_html,out_md,figures_dir)`, `style.apply_style/PALETTE/caption/provenance/table_image`, `manifest.TRACKS/SUMMARY` used identically across tasks. ✓
- **No placeholders:** every code step shows complete code; restyle steps name exact functions to edit. ✓
