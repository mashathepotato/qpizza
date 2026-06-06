# results/ — presentation dashboard

One command rebuilds a self-contained HTML dashboard unifying every track's
figures + headline numbers (no server, no external assets).

```bash
python -m results.build_dashboard   # writes results/index.html + results/RESULTS.md
open results/index.html             # (macOS) view it
```

Run it from the repo root with an env that has `matplotlib` + `numpy`
(e.g. `quantum_investor/.venv/bin/python -m results.build_dashboard`).

## Files
- `style.py` — shared figure style (palette, rcParams, `caption`/`provenance`/`table_image`).
  Vendored into worktrees that build their own figures (e.g. triage).
- `manifest.py` — the ONE file to edit to change dashboard content or figure sources.
  Each track declares: title, claim, figure path(s), key-number table, prose, provenance.
- `build_dashboard.py` — regenerates the cognition figure, collects pricer + triage
  figures, base64-embeds them, emits `index.html` + `RESULTS.md`. Missing sources
  degrade to a placeholder; the build never crashes.
- `figures/` — archival copies of the collected/regenerated PNGs (build output).
- `index.html`, `RESULTS.md` — build outputs.

## Adding a result
Drop its figure somewhere, add a track dict to `manifest.TRACKS` (point `figure`
at the path), re-run the build. To add headline cards, append to `manifest.SUMMARY`.

## Tracks today
| track | source | headline |
|---|---|---|
| cognition | `quantum_investor/figure.png` (regenerated) | parameter-free QQ-equality, q = −0.003 |
| pricer | `quantum_pricer/*.png` (collected) | O(1/ε) vs O(1/ε²) query complexity |
| triage | `.claude/worktrees/triage-lab/triage/plots/qae_scaling.png` (collected) | QAE slope −1 vs MC −2 |
