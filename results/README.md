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

## Animated pricing-race demo

1. Export the real data (once, or after new pricer runs):
   `quantum_pricer/.venv/bin/python -m results.export_demo_data`
   → writes `prices.json`, `convergence.json`, `hardware.json`.
2. Open `results/demo_animation.html` in a browser. ←/→/space move through the 5 acts;
   press **E** for the explore cockpit. Works fully offline from `file://`.
3. (Optional) live mode: `quantum_pricer/.venv/bin/python -m results.live_server`,
   then reload — adds "⟳ re-run" buttons. If the server is down the demo is unchanged.
4. Q50 hardware: a teammate overwrites `results/hardware.json` with
   `{"status":"done","backend":"Q50","route":"fourier","price":<p>,"abs_error":<e>,"shots":<n>}`;
   the act-5 badge and cockpit then light up green. Until then it reads "pending".

Honesty: the convergence/race axis is **oracle queries / samples, not wall-clock time**;
the QAE lead appears at tight accuracy; QSVT plateaus at its polynomial-approx floor;
ground truth is the exact CRR tree price.
