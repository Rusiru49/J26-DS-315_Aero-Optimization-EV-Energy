"""API tests: file mode, api mode (backend -> mock C3 over HTTP), and C3 failure handling."""
from fastapi.testclient import TestClient

from app.c3_client import ApiC3Client
from app.config import Settings
from app.main import create_app


def test_catalog_marks_coming_soon(client):
    cat = client.get("/api/catalog").json()
    status = {c["condition_id"]: c["status"] for c in cat["conditions"]}
    assert status == {"WLTP": "available", "SLDC_Urban": "available", "SLDC_Suburban": "coming_soon",
                      "SLDC_Intercity": "coming_soon", "SLDC_Expressway": "available",
                      "SLDC_Rural": "coming_soon"}
    assert cat["meta"]["is_dummy"] is True and cat["meta"]["predicted_only"] is True
    assert cat["evs"][0]["ev_id"] == "EV01" and len(cat["shape_parameter_definitions"]) == 5


def test_function_1_endpoint(client):
    r = client.get("/api/comparison", params={"ev_id": "EV01", "condition_id": "SLDC_Urban"})
    assert r.status_code == 200
    j = r.json()
    assert j["available"] and j["savings"]["energy_saving_pct"] > 0 and j["caution"]
    assert j["baseline"]["geometry_file"] == "EV01_BASE.glb"
    assert float(r.headers["X-Process-Time-ms"]) < 2000  # NFR1 target: about 2-3 seconds


def test_coming_soon_condition_is_not_an_error(client):
    j = client.get("/api/comparison", params={"ev_id": "EV01", "condition_id": "SLDC_Rural"}).json()
    assert j["available"] is False and j["status"] == "coming_soon"


def test_not_found_and_bad_input(client):
    assert client.get("/api/comparison", params={"ev_id": "X", "condition_id": "WLTP"}).status_code == 404
    assert client.get("/api/comparison", params={"ev_id": "EV01", "condition_id": "Nope"}).status_code == 404
    assert client.get("/api/sensitivity", params={"ev_id": "EV01", "condition_id": "WLTP",
                                                  "tariff_change_pct": 900}).status_code == 422
    assert client.get("/api/comparison").status_code == 422  # missing parameters


def test_function_2_endpoints(client):
    c = client.get("/api/conditions/compare", params={"ev_id": "EV01"}).json()
    assert c["benchmark_condition"] == "WLTP" and len(c["conditions"]) == 2
    one = client.get("/api/conditions/compare",
                     params={"ev_id": "EV01", "conditions": "SLDC_Urban"}).json()
    assert [r["condition_id"] for r in one["conditions"]] == ["SLDC_Urban"]
    assert client.get("/api/conditions/compare",
                      params={"ev_id": "EV01", "conditions": "SLDC_Nope"}).status_code == 404
    s = client.get("/api/sensitivity", params={"ev_id": "EV01", "condition_id": "SLDC_Expressway"}).json()
    assert s["cd_check"]["within_uncertainty"] is False and len(s["tariff"]["sweep"]) == 5
    assert client.get("/api/mix", params={"ev_id": "EV01"}).status_code == 404   # the mix route was removed


def test_data_quality_and_statistics_endpoints(client):
    assert client.get("/api/data-quality").json()["status"] == "pass"
    live = client.get("/api/statistics/live").json()
    status = {t["id"]: t["status"] for t in live["tests"]}
    assert status == {"paired_t": "ready", "anova": "locked", "correlation": "locked", "regression": "locked"}
    sample = client.get("/api/statistics/sample").json()
    assert sample["anova"]["significant"] is True and "synthetic" not in str(sample).lower()


def test_api_mode_gives_same_answers_as_file_mode(client, api_mode_client):
    """Backend -> HTTP -> mock C3 returns exactly what the file client returns."""
    assert api_mode_client.get("/api/health").json()["c3_mode"] == "api"
    for cond in ("WLTP", "SLDC_Urban", "SLDC_Expressway", "SLDC_Rural"):
        p = {"ev_id": "EV01", "condition_id": cond}
        a, b = client.get("/api/comparison", params=p).json(), api_mode_client.get("/api/comparison", params=p).json()
        a.pop("meta"), b.pop("meta")
        assert a == b
    assert api_mode_client.get("/api/data-quality").json()["status"] == "pass"


def test_c3_down_is_reported_clearly():
    dead = ApiC3Client("http://127.0.0.1:1", timeout_s=0.2)
    with TestClient(create_app(Settings(c3_mode="api"), client=dead)) as c:
        h = c.get("/api/health").json()
        assert h["status"] == "c3_unavailable" and h["last_error"]
        r = c.get("/api/catalog")
        assert r.status_code == 502 and r.json()["code"] == "c3_error"