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
        # persist a copy into results/figures/ for archival (skip if already there)
        os.makedirs(figures_dir, exist_ok=True)
        dst = os.path.join(figures_dir, os.path.basename(fp))
        if os.path.abspath(fp) != os.path.abspath(dst):
            shutil.copy(fp, dst)
        imgs.append(f'<img src="{embed(data)}" alt="{title}"/>')

    if imgs:
        fig_block = '<div class="figs">' + "".join(imgs) + "</div>"
    else:
        fig_block = ('<div class="missing">Figure unavailable - rebuild the '
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
<body><header><h1>The Madness of People Is Quantum - Results Dashboard</h1>
<div class="ts">Built {ts} - re-run <code>python -m results.build_dashboard</code> to refresh</div></header>
<main>{_summary_html()}{sections}</main>
<footer>Self-contained - figures embedded - OP Pohjola x Junction Helsinki</footer></body></html>"""


def write_results_md(path):
    s = manifest.SUMMARY
    lines = ["# Results rollup", "", s["framing"], ""]
    for h in s["headlines"]:
        lines.append(f"- **{h['label']}** - {h['value']} ({h['sub']})")
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
