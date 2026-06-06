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
