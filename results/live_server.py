"""Optional, ADDITIVE live server: re-run a single pricing route on demand for judges.

The demo_animation.html frontend works fully WITHOUT this server. When it is running,
the frontend shows '⟳ re-run live' buttons; when it is not, those buttons are hidden and
the prebaked animation is unchanged. Never required for the demo.

Run:  quantum_pricer/.venv/bin/python -m results.live_server   (serves on :5057)
"""
from flask import Flask, jsonify, request

from quantum_pricer import classical, qae, tree

app = Flask(__name__)


@app.after_request
def _cors(resp):                      # allow the file:// frontend to call us
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.get("/health")
def health():
    return jsonify(ok=True)


def _params():
    g = request.args
    return dict(S0=float(g.get("S0", 100)), K=float(g.get("K", 100)),
                r=float(g.get("r", 0.05)), sigma=float(g.get("sigma", 0.2)),
                T=float(g.get("T", 1.0)), M=int(g.get("M", 3)))


@app.get("/api/rerun/<route>")
def rerun(route):
    p = _params()
    gt = tree.exact_tree_price(**p)
    eps = float(request.args.get("eps", 0.05))
    if route == "qae":
        res = qae.price(epsilon_target=eps, **p)
        price, queries = res["price"], res["num_oracle_queries"]
    elif route == "mc":
        n = int(request.args.get("n", 20000))
        price, _ = classical.monte_carlo_price(n_paths=n, **p)
        queries = n
    else:
        return jsonify(error="unknown route '%s' (use qae|mc)" % route), 400
    return jsonify(route=route, price=float(price), queries=int(queries),
                   ground_truth=float(gt), abs_error=abs(float(price) - float(gt)))


if __name__ == "__main__":
    app.run(port=5057)
