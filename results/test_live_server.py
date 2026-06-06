from results import live_server


def test_health_ok():
    client = live_server.app.test_client()
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_rerun_qae_returns_price_near_truth():
    client = live_server.app.test_client()
    # small fast instance; route must return a price + query count near the tree truth
    r = client.get("/api/rerun/qae?S0=100&K=100&r=0.05&sigma=0.2&T=1&M=3&eps=0.05")
    assert r.status_code == 200
    body = r.get_json()
    assert body["route"] == "qae"
    assert body["price"] > 0 and body["queries"] >= 0
    assert abs(body["price"] - body["ground_truth"]) < 0.5


def test_rerun_unknown_route_is_400():
    client = live_server.app.test_client()
    r = client.get("/api/rerun/nope")
    assert r.status_code == 400


def test_rerun_mc_returns_price_near_truth():
    client = live_server.app.test_client()
    r = client.get("/api/rerun/mc?S0=100&K=100&r=0.05&sigma=0.2&T=1&M=3&n=20000")
    assert r.status_code == 200
    body = r.get_json()
    assert body["route"] == "mc"
    assert body["queries"] == 20000 and body["price"] > 0
    assert abs(body["price"] - body["ground_truth"]) < 1.0
