"""HTTP contract of every route, with USGS replaced by a fake transport."""
import pytest

import app as app_module
import config
from ml_model import ModelBundle
from tests.conftest import collection, feature
from usgs_client import UpstreamError


def assert_error(resp, status, code=None):
    assert resp.status_code == status
    body = resp.get_json()
    assert set(body) >= {"error", "code", "status"} and body["status"] == status
    assert isinstance(body["error"], str) and body["error"]
    if code:
        assert body["code"] == code


@pytest.fixture
def no_model(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "bundle", ModelBundle(str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "c")))


# ---------------------------------------------------------------- pages / meta
def test_index_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "TerraWatcher" in html and "core.js" in html and "dashboard.js" in html
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert "Access-Control-Allow-Origin" not in r.headers


def test_health_reports_ready(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok" and body["model_loaded"] and body["plate_segments"] > 0


def test_health_degraded_without_model(client, no_model):
    body = client.get("/api/health").get_json()
    assert body["status"] == "degraded" and body["model_error"]


def test_model_info(client):
    body = client.get("/api/model-info").get_json()
    assert body["status"] == "loaded"
    assert body["features"] == ["latitude", "longitude", "depth_km", "local_density_30d", "plate_distance_km"]
    assert "test" in body["metrics"] and "baselines" in body


def test_plates_geojson(client):
    r = client.get("/api/plates")
    assert r.status_code == 200 and r.get_json()["type"] == "FeatureCollection"


def test_unknown_api_route_is_json_404(client):
    assert_error(client.get("/api/does-not-exist"), 404, "not_found")


def test_wrong_method_is_json_405(client):
    assert_error(client.post("/api/earthquakes"), 405)


# ---------------------------------------------------------------- earthquakes
def test_earthquakes_contract(client, fake_usgs):
    bad = feature("bad", 0, 0)
    bad["geometry"]["coordinates"] = [None, None]
    fake_usgs.query_payload = collection(feature("a", 142.0, 38.0, depth=30.0, mag=5.2),
                                         feature("b", -120.0, 36.0, depth=None, mag=None), bad)
    r = client.get("/api/earthquakes?days=7&min_mag=2.5")
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] == 2 and body["dropped_invalid"] == 1 and body["truncated"] is False
    a, b = body["events"]
    assert (a["latitude"], a["longitude"], a["depth_km"], a["mag"]) == (38.0, 142.0, 30.0, 5.2)
    assert b["depth_km"] is None and b["mag"] is None          # missing values stay missing
    assert all(not (e["latitude"] == 0 and e["longitude"] == 0) for e in body["events"])  # no Null Island
    assert "prediction" not in a                               # observed events are not "scored"
    assert body["query"]["days"] == 7 and body["source"].startswith("USGS")
    assert body["cache"] == {"hit": False, "age_seconds": 0, "stale": False, "warning": None}
    path, params = fake_usgs.calls[0]
    assert path == "query" and params["minmagnitude"] == 2.5 and "T" in params["endtime"]


def test_earthquakes_second_call_is_cached(client, fake_usgs):
    fake_usgs.query_payload = collection(feature("a", 1, 1))
    client.get("/api/earthquakes?days=1")
    body = client.get("/api/earthquakes?days=1").get_json()
    assert body["cache"]["hit"] is True and len(fake_usgs.calls) == 1


def test_earthquakes_empty_result(client, fake_usgs):
    body = client.get("/api/earthquakes").get_json()
    assert body["events"] == [] and body["count"] == 0


def test_earthquakes_truncation_is_reported(client, fake_usgs, monkeypatch):
    monkeypatch.setattr(config, "MAX_EVENTS", 2)
    fake_usgs.query_payload = collection(feature("a", 1, 1), feature("b", 2, 2))
    fake_usgs.count_payload = {"count": 57}
    body = client.get("/api/earthquakes").get_json()
    assert body["truncated"] is True and body["total_available"] == 57


@pytest.mark.parametrize("query,code", [
    ("days=0", "out_of_range"), ("days=31", "out_of_range"), ("days=abc", "invalid_parameter"),
    ("days=1.5", "invalid_parameter"), ("min_mag=11", "out_of_range"), ("min_mag=nan", "invalid_parameter"),
    ("min_mag=abc", "invalid_parameter"),
])
def test_earthquakes_validation(client, fake_usgs, query, code):
    assert_error(client.get(f"/api/earthquakes?{query}"), 400, code)
    assert fake_usgs.calls == []          # invalid requests never reach USGS


@pytest.mark.parametrize("error,status", [
    (UpstreamError("timed out", status=504, code="upstream_timeout"), 504),
    (UpstreamError("HTTP 503", code="upstream_server_error"), 502),
    (UpstreamError("bad json", code="upstream_malformed"), 502),
])
def test_earthquakes_upstream_failure_without_cache(client, fake_usgs, error, status):
    fake_usgs.error = error
    assert_error(client.get("/api/earthquakes"), status, error.code)


def test_earthquakes_stale_cache_is_labelled(client, fake_usgs, monkeypatch):
    fake_usgs.query_payload = collection(feature("a", 1, 1))
    client.get("/api/earthquakes")
    monkeypatch.setattr(config, "FEED_CACHE_TTL", -1)      # force the cached copy to be considered expired
    fake_usgs.error = UpstreamError("down", code="upstream_unreachable")
    body = client.get("/api/earthquakes").get_json()
    assert body["cache"]["stale"] is True and "unavailable" in body["cache"]["warning"]
    assert [e["id"] for e in body["events"]] == ["a"]       # real (old) data, clearly flagged


# ---------------------------------------------------------------- hotspots
def _cluster_payload():
    now = app_module.utc_now().timestamp() * 1000
    feats = [feature(f"j{i}", 142.2 + i * 0.01, 38.3, depth=20 + i, mag=4.6 + i / 10, time_ms=now - 60_000 - i * 3_600_000)
             for i in range(5)]
    feats += [feature(f"c{i}", -71.5, -33.1, depth=40, mag=4.8, time_ms=now - 7_200_000) for i in range(3)]
    feats += [feature("lonely", 10.0, 10.0, time_ms=now - 1000)]
    return collection(*feats)


def test_hotspots_are_observed_clusters(client, fake_usgs):
    fake_usgs.query_payload = _cluster_payload()
    body = client.get("/api/hotspots").get_json()
    spots = body["hotspots"]
    assert [s["event_count"] for s in spots] == [5, 3]          # cell with 1 event is excluded
    top = spots[0]
    assert top["cell"] == {"lat_min": 38.0, "lat_max": 40.0, "lon_min": 142.0, "lon_max": 144.0}
    assert top["max_mag"] == pytest.approx(5.0) and top["median_depth_km"] == 22.0
    assert 0 <= top["model_probability"] <= 1
    assert "not a forecast" in body["definition"]
    path, params = fake_usgs.calls[0]
    assert params["minmagnitude"] == 4.5


def test_hotspots_without_model_have_null_scores(client, fake_usgs, no_model):
    fake_usgs.query_payload = _cluster_payload()
    body = client.get("/api/hotspots").get_json()
    assert body["model_status"] == "unavailable"
    assert all(s["model_probability"] is None for s in body["hotspots"])   # no placeholder 0.5


def test_hotspots_limit_validation(client):
    assert_error(client.get("/api/hotspots?limit=0"), 400, "out_of_range")


# ---------------------------------------------------------------- predict
def test_predict_valid(client, fake_usgs):
    fake_usgs.query_payload = _cluster_payload()
    r = client.get("/api/predict?lat=38.3&lon=142.25&depth=25")
    assert r.status_code == 200
    body = r.get_json()
    assert 0 <= body["probability"] <= 1
    assert body["features"]["local_density_30d"] == 5
    assert body["features"]["plate_distance_km"] > 0
    assert body["band"] in ("above_base_rate", "near_base_rate", "below_base_rate")
    assert "not a probability that an earthquake will occur" in body["interpretation"]


def test_predict_lng_alias_and_default_depth(client, fake_usgs):
    body = client.get("/api/predict?lat=0&lng=0").get_json()
    assert body["features"]["depth_km"] == 10.0 and body["features"]["local_density_30d"] == 0


def test_predict_is_deterministic(client, fake_usgs):
    a = client.get("/api/predict?lat=10&lon=20&depth=33").get_json()
    b = client.get("/api/predict?lat=10&lon=20&depth=33").get_json()
    assert a["probability"] == b["probability"]


@pytest.mark.parametrize("query,code", [
    ("lat=10", "missing_parameter"), ("lon=10", "missing_parameter"), ("lat=95&lon=0", "out_of_range"),
    ("lat=0&lon=181", "out_of_range"), ("lat=abc&lon=1", "invalid_parameter"), ("lat=nan&lon=inf", "invalid_parameter"),
    ("lat=0&lon=0&depth=-5", "out_of_range"), ("lat=0&lon=0&depth=701", "out_of_range"),
])
def test_predict_validation(client, query, code):
    assert_error(client.get(f"/api/predict?{query}"), 400, code)


def test_predict_without_model_is_503(client, no_model):
    assert_error(client.get("/api/predict?lat=0&lon=0"), 503, "model_unavailable")


def test_predict_without_context_does_not_guess(client, fake_usgs):
    fake_usgs.error = UpstreamError("down", code="upstream_unreachable")
    assert_error(client.get("/api/predict?lat=0&lon=0"), 503, "context_unavailable")


def test_unexpected_exception_is_generic_500(client, monkeypatch):
    def boom():
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(app_module, "get_context_catalog", boom)
    r = client.get("/api/hotspots")
    assert_error(r, 500, "internal_error")
    assert "secret" not in r.get_data(as_text=True)
