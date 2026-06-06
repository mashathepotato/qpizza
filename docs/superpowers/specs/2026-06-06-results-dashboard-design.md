# Results Dashboard + Figure Improvement — Design Spec

**Date:** 2026-06-06
**Status:** Approved (pending user spec review)
**Goal:** Improve every figure in the project and create a `results/` folder that
collects results, updates, and presentation material in maximally visual form —
centered on a single, re-buildable, self-contained HTML dashboard.

---

## 1. Problem & intent

The project has three result-producing tracks, currently scattered across the main
branch and two git worktrees, each with its own figures and result notes:

| Track | Location | Figures today | Headline result |
|---|---|---|---|
| **Cognition** ("madness is quantum") | `quantum_investor/` (main) | `figure.png` | order effect real & large; parameter-free QQ-equality `q = -0.003` |
| **Pricer** (technical core) | `.claude/worktrees/quantum-pricer/quantum_pricer/` | `complexity.png`, `speedup.png` | QNDM+QAE query complexity `O(1/ε)` vs MC `O(1/ε²)` |
| **Triage** | `.claude/worktrees/triage-lab/triage/` | `plots/qae_scaling.png` | QAE scaling slope ≈ −1 vs classical ≈ −2 |

There is no single place a judge (or the team) can look to see all results at a
glance, and figure quality/style is inconsistent across tracks.

**Intent:** one command rebuilds a dense, technical-report-style HTML dashboard that
unifies all three tracks, plus genuinely improved figures everywhere.

### Success criteria
- `python results/build_dashboard.py` produces `results/index.html` that opens in a
  browser with **no server and no external assets** (PNGs base64-embedded).
- The dashboard shows all three tracks, each with: headline claim → improved figure →
  dense results table with provenance → short "what this shows / honest caveat".
- Every figure in the repo is regenerated through one shared visual style.
- Re-running the build after any track's results change refreshes the dashboard.
- A missing worktree degrades gracefully (placeholder section, no crash).

### Non-goals (YAGNI)
- No live-reloading server or websocket auto-refresh — rebuild-on-demand only.
- No new scientific results; this is presentation of existing results.
- No merging of worktree branches into main; the build reads worktree paths in place.

---

## 2. Architecture

```
results/                       (new, on main)
  style.py            shared matplotlib style + helpers (palette, rcParams, caption, provenance, table-to-PNG)
  manifest.py         declarative list of the 3 tracks (title, narrative, figure files, key-number tables, citations)
  build_dashboard.py  the entry point: regenerate → collect → emit
  figures/            build output: collected/regenerated PNGs
  index.html          build output: the self-contained dashboard
  RESULTS.md          build output: plain-text rollup of headline numbers
```

Prior art: `triage/dashboard.py` already emits a self-contained HTML with base64
PNGs. `build_dashboard.py` generalizes that pattern (embedding + HTML assembly) to
three tracks driven by `manifest.py`.

### Data flow
```
manifest.py (track declarations)
        │
        ▼
build_dashboard.py
   1. regenerate quantum_investor/figure.png  (import quantum_investor, run its plot with style.py)
   2. for each worktree track: locate its PNGs at the declared path
        ├─ present  → copy into results/figures/, record numbers from manifest
        └─ missing  → mark section "unavailable"
   3. assemble index.html: summary band + one <section> per track (base64 PNG + table + prose)
   4. write RESULTS.md rollup
        │
        ▼
results/index.html   (open in browser; refresh after rebuild)
```

### Why this shape
- **`manifest.py` is the single source of truth** for what appears and what the
  numbers/citations are, so updating the dashboard is editing one declarative file,
  not the renderer.
- **`style.py` is importable from every track** (vendored copy in each worktree) so
  figure improvement is real re-rendering, not a raster post-filter.
- **`build_dashboard.py` only orchestrates** — collect + assemble. It can be
  understood and changed without touching style or content.

---

## 3. Components

### 3.1 `results/style.py`
One consistent **dense technical-report** look applied to all figures.
- `apply_style()` — sets matplotlib rcParams: serif/technical font, ~150 dpi, light
  gridlines, tick direction in, axis labels with units required by convention.
- `PALETTE` — a small named, high-contrast, colorblind-safe palette
  (quantum / classical / accent / muted) reused across all tracks for visual unity.
- `caption(fig, text)` — adds an italic footnote line under a figure (source / method).
- `provenance(fig, text)` — small bottom-right stamp (dataset/citation + build note).
- `table_image(rows, header, path)` — render a results table to PNG for embedding
  where a track has no native figure for some numbers.
- Pure functions, no I/O beyond the explicit save path. Importable standalone.

### 3.2 `results/manifest.py`
A plain Python module exporting `TRACKS`, a list of dicts:
```python
{
  "key": "cognition",
  "title": "The Madness of People Is Quantum",
  "claim": "Human question-order data obeys the parameter-free QQ-equality.",
  "figure": <abs path to PNG>,            # regenerated or collected
  "regenerate": <callable or None>,        # None for worktree tracks
  "table": {"header": [...], "rows": [[...], ...]},   # dense key-number table
  "prose": "what this shows ... honest caveat ...",
  "provenance": "Wang et al., PNAS 111:9431 (2014) ...",
}
```
The three headline numbers for the summary band (`q = -0.003`; `O(1/ε)` vs `O(1/ε²)`;
slope −1 vs −2) are declared here too. Worktree paths are computed relative to the
repo root so the file is portable.

### 3.3 `results/build_dashboard.py`
- `regenerate_cognition()` — imports `quantum_investor` modules, recomputes the
  numbers, and calls the (restyled) `headline_figure` to write the PNG fresh.
- `collect(track)` — resolves a track's figure path; returns `(png_bytes | None)`.
- `embed(png_bytes)` — base64 data-URI (reused from triage pattern).
- `render_html(tracks, summary)` — emits the dense HTML: a sticky summary band,
  Newton↔Feynman framing line, then one section per track. Inline CSS only.
- `write_results_md(tracks)` — plain-text headline rollup.
- `main()` — orchestrates 1–4 above; prints the output path and a per-track
  available/unavailable status line.

### 3.4 Figure improvements (per track, through `style.py`)
- **Cognition** — rewrite `quantum_investor/plot.py`:
  - Panel A: `p(B=yes)` by order, humans vs classical, with data labels and the
    "classical predicts the same value twice" annotation kept.
  - Panel B: `|order effect|` vs `|QQ-equality q|` with a `χ²(1)=0.01, p=0.91`
    annotation (from PNAS) and value labels.
  - Add a compact numeric results table (via `style.table_image` or a third panel).
  - Use `PALETTE` and `caption`/`provenance`.
- **Pricer** — in the `quantum-pricer` worktree, restyle `complexity.png` and
  `speedup.png` (the benchmark plotting code) through a vendored `style.py`:
  consistent palette, log-log axes labeled with units, reference-slope guide lines,
  the 4-route table reused on the dashboard.
- **Triage** — in the `triage-lab` worktree, restyle `qae_scaling.png` (slope −1 vs
  −2, error vs queries) through the vendored `style.py`.

`style.py` is vendored (copied) into each worktree so its plotting code can
`import style`. The canonical copy lives in `results/`.

---

## 4. Error handling & edge cases
- **Missing worktree / figure:** `collect()` returns `None`; the section renders an
  "unavailable — rebuild the <track> worktree" placeholder. Build still succeeds.
- **Regeneration failure (cognition):** caught per-track; the section falls back to
  the last `figure.png` on disk if present, else placeholder; the error is printed.
- **Stale figures:** the build always re-embeds whatever PNG is currently on disk and
  stamps `index.html` with a build timestamp so staleness is visible.
- **No `Date.now()` concerns** — this is Python; timestamp via `datetime` at build.

---

## 5. Testing
- `results/` gets a lightweight check (extend `quantum_investor/test_model.py` or a
  new `results/test_build.py`): build runs end-to-end, `index.html` exists, is
  non-empty, contains one section per available track, and embeds at least the
  cognition PNG.
- Visual check: open `index.html` and confirm all three figures render and tables
  are populated.
- Each restyled figure is eyeballed against its track's known headline numbers.

---

## 6. Implementation order (for the plan)
1. `results/style.py` + vendor mechanism.
2. Restyle `quantum_investor/plot.py`; wire `regenerate_cognition()`.
3. `results/manifest.py` with all three tracks declared.
4. `results/build_dashboard.py`: collect + embed + render + RESULTS.md.
5. Restyle pricer figures in the quantum-pricer worktree.
6. Restyle triage figure in the triage-lab worktree.
7. `results/test_build.py`; final visual pass.
