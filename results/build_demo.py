"""Bake the guided-slides demo into ONE self-contained file: results/demo.html.

Reads the exported JSON (results/convergence/forecast/hardware) and base64-embeds the
real result figures, injecting both inline into results/demo_template.html so the output
opens by double-click over file:// with ZERO network fetches. Re-run after you update a
figure or re-export data.

Run:  python -m results.build_demo
"""
import base64
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIGS = os.path.join(_HERE, "figures")

# figures woven into the story + cockpit (name -> file under results/figures/)
FIGURES = {
    "price_forecast":   "price_forecast.png",
    "backtest_rolling": "backtest_rolling.png",
    "model_results":    "model_results.png",
    "error_vs_queries": "error_vs_queries.png",
    "depth_crossover":  "depth_crossover.png",
    "qae_scaling":      "qae_scaling.png",
}
# JSON the frontend reads (filename -> DATA key)
DATASETS = {
    "results.json": "results",
    "convergence.json": "convergence",
    "forecast.json": "forecast",
    "hardware.json": "hardware",
}


def _load_json(out_dir):
    data = {}
    for fname, key in DATASETS.items():
        path = os.path.join(out_dir, fname)
        data[key] = json.load(open(path)) if os.path.exists(path) else None
    return data


def _load_figs(figs_dir):
    figs = {}
    for name, fname in FIGURES.items():
        path = os.path.join(figs_dir, fname)
        if os.path.exists(path):
            b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
            figs[name] = "data:image/png;base64," + b64
    return figs


def build(out_dir=None, figs_dir=None, template=None, out_html=None):
    out_dir = out_dir or _HERE
    figs_dir = figs_dir or _FIGS
    template = template or os.path.join(_HERE, "demo_template.html")
    out_html = out_html or os.path.join(out_dir, "demo.html")

    html = open(template).read()
    data = _load_json(out_dir)
    figs = _load_figs(figs_dir)

    html = html.replace("/*__DATA__*/ null", "/*__DATA__*/ " + json.dumps(data))
    html = html.replace("/*__FIGS__*/ {}", "/*__FIGS__*/ " + json.dumps(figs))

    with open(out_html, "w") as fh:
        fh.write(html)
    return dict(out_html=out_html, n_figs=len(figs),
                data_keys=[k for k, v in data.items() if v is not None])


if __name__ == "__main__":
    info = build()
    print("wrote %s  (%d figures embedded, data: %s)"
          % (info["out_html"], info["n_figs"], ", ".join(info["data_keys"])))
