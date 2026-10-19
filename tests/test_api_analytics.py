"""HTTP contract of the analytical endpoints, with USGS replaced by a fake transport."""
import json
import time

import pytest

import app as app_module
import config
from analytics import aftershock as AS
from analytics import regions
from analytics.forecast import ForecastModel
from tests.conftest import collection, feature
from tests.test_api import assert_error
from usgs_client import UpstreamError

DAY = 86_400_000
NOW = int(time.time() * 1000)
FORECAST_REGION = app_module.forecast_models[4.5].meta["regions"][0]


def region_point(rid):
    b = next(r for r in app_module.forecast_models[4.5].meta["regions"] if r["id"] == rid)["bounds"]
    return b["lat_min"] + 5.0, b["lon_min"] + 5.0


def history_payload(burst=0, weeks=50, per_week=2):
    """Steady M4.6 activity in one eligible region, optional burst in the last 3 days, one event per day elsewhere."""
    lat, lon = region_point(FORECAST_REGION["id"])
    feats = []
    for w in range(1, weeks + 1):
        for j in range(per_week):
            feats.append(feature(f"h{w}-{j}", lon, lat, mag=4.6, time_ms=NOW - w * 7 * DAY + j * DAY))
    for i in range(burst):
        feats.append(feature(f"b{i}", lon + 0.1, lat, mag=5.2 if i else 6.1, time_ms=NOW - 3 * DAY + i * 3_600_000))
    for d in range(1, 400):    # global background so no day looks like a feed gap
        feats.append(feature(f"g{d}", -30.0, -60.0 + (d % 50) * 0.01, mag=4.5, time_ms=NOW - d * DAY + 1000))
    return collection(*feats)


# ------------------------------------------------------------------ sequences
def test_sequences_endpoint_groups_feed_events(client, fake_usgs):
    t = NOW - 2 * DAY
    fake_usgs.query_payload = collection(*[feature(f"s{i}", 142.0 + 0.01 * i, 38.0, time_ms=t + i * 3_600_000) for i in range(6)],
                                         feature("lone", -70.0, -30.0, time_ms=t))
    body = client.get("/api/sequences?days=7&min_mag=2.5").get_json()
    assert body["summary"]["sequence_count"] == 1 and body["summary"]["noise_events"] == 1
    seq = body["sequences"][0]
    assert seq["event_count"] == 6 and len(seq["member_ids"]) == 6
    by_id = {e["id"]: e["sequence"] for e in body["events"]}
    assert by_id["lone"] is None and by_id["s0"] == seq["id"]
    assert body["parameters"]["eps_km"] == 50 and "explanation" in body and body["limitations"]


@pytest.mark.parametrize("query,code", [("eps_km=1", "out_of_range"), ("eps_hours=x", "invalid_parameter"),
                                        ("min_samples=2.5", "invalid_parameter"), ("min_samples=500", "out_of_range")])
def test_sequences_parameter_validation(client, query, code):
    assert_error(client.get(f"/api/sequences?{query}"), 400, code)


def test_sequences_result_is_cached_per_parameters(client, fake_usgs):
    fake_usgs.query_payload = collection(*[feature(f"s{i}", 142.0, 38.0, time_ms=NOW - DAY + i) for i in range(5)])
    client.get("/api/sequences?days=7")
    client.get("/api/sequences?days=7")
    client.get("/api/sequences?days=7&eps_km=20")
    assert len([c for c in fake_usgs.calls if c[0] == "query"]) == 1     # one feed download for all three


# ------------------------------------------------------------------ forecasts
def test_forecast_all_regions(client, fake_usgs):
    fake_usgs.query_payload = history_payload()
    body = client.get("/api/forecast?threshold=4.5").get_json()
    assert body["horizon_days"] == 7 and body["origin"].endswith("T00:00:00Z")
    assert len(body["regions"]) == len(app_module.forecast_models[4.5].meta["regions"])
    row = next(r for r in body["regions"] if r["region"] == FORECAST_REGION["id"])
    assert row["expected"] > 0 and row["interval_80"][0] <= row["interval_80"][1] <= row["interval_95"][1]
    assert row["baselines"]["rolling_mean_4w"] == pytest.approx(2.0)
    assert "not say when" in body["interpretation"]
    assert body["model"]["test_metrics"] and body["data_window"]["events_downloaded"] > 0
    q = fake_usgs.calls[0][1]
    assert q["minmagnitude"] == 4.5 and q["limit"] == config.USGS_QUERY_LIMIT


def test_forecast_region_detail(client, fake_usgs):
    fake_usgs.query_payload = history_payload()
    rid = FORECAST_REGION["id"]
    body = client.get(f"/api/forecast/{rid}?threshold=4.5").get_json()
    assert body["region"]["region"] == rid
    counts = body["observed_weekly"]["counts"]
    assert len(counts) == 52 and sum(counts) >= 2 * 49
    assert [f["name"] for f in body["features"]][0] == "lag1" and all("unit" in f for f in body["features"])
    assert len(body["backtest"]["observed"]) == len(body["backtest"]["week_starts"])


def test_forecast_region_errors(client, fake_usgs):
    fake_usgs.query_payload = history_payload()
    assert_error(client.get("/api/forecast/R+35+140"), 404, "unknown_region")
    ineligible = next(regions.region_id(i) for i in range(regions.N_REGIONS)
                      if i not in app_module.forecast_models[4.5].region_index)
    assert_error(client.get(f"/api/forecast/{ineligible}"), 404, "insufficient_history")
    assert_error(client.get("/api/forecast?threshold=4.7"), 400, "invalid_parameter")


def test_forecast_model_unavailable_does_not_break_live_monitor(client, fake_usgs, monkeypatch, tmp_path):
    monkeypatch.setitem(app_module.forecast_models, 4.5, ForecastModel(str(tmp_path), 4.5))
    assert_error(client.get("/api/forecast"), 503, "model_unavailable")
    assert client.get("/api/earthquakes").status_code == 200
    assert client.get("/api/health").get_json()["status"] == "degraded"


def test_history_failure_is_an_error_not_an_empty_forecast(client, fake_usgs):
    fake_usgs.error = UpstreamError("USGS responded with HTTP 503", code="upstream_server_error")
    r = client.get("/api/forecast")
    assert_error(r, 503, "history_unavailable")
    assert "No result was computed" in r.get_json()["error"]


def test_truncated_history_is_refused(client, fake_usgs, monkeypatch):
    monkeypatch.setattr(config, "USGS_QUERY_LIMIT", 10)
    fake_usgs.query_payload = history_payload()
    assert_error(client.get("/api/anomalies"), 503, "history_unavailable")


# ------------------------------------------------------------------ anomalies
def test_anomalies_flag_a_burst_and_explain_it(client, fake_usgs):
    fake_usgs.query_payload = history_payload(burst=40)
    body = client.get("/api/anomalies?threshold=4.5&window_weeks=1").get_json()
    assert body["window_weeks"] == 1 and body["reference_weeks"] == 52
    assert body["observation_window"]["to"] > body["reference_window"]["to"] >= body["reference_window"]["from"]
    row = next(r for r in body["results"] if r["region"] == FORECAST_REGION["id"])
    assert row["status"] == "elevated" and row["observed"] >= 40 and row["expected"] < 5
    assert row["largest_event"]["mag"] == 6.1 and row["q_value"] <= 0.05
    assert "not a prediction" in body["interpretation"]
    assert body["data_quality"]["possible_gap_days_in_window"] == []


def test_anomaly_region_detail_lists_contributing_events(client, fake_usgs):
    fake_usgs.query_payload = history_payload(burst=10)
    rid = FORECAST_REGION["id"]
    body = client.get(f"/api/anomalies/{rid}?window_weeks=1").get_json()
    assert len(body["weekly"]["counts"]) == 53 and body["weekly"]["window_weeks"] == 1
    ids = {e["id"] for e in body["contributing_events"]}
    assert {f"b{i}" for i in range(10)} <= ids


def test_anomaly_gap_days_are_not_reported_as_normal(client, fake_usgs):
    payload = history_payload()
    payload["features"] = [f for f in payload["features"] if not (NOW - 6 * DAY < f["properties"]["time"] < NOW - 3 * DAY)]
    fake_usgs.query_payload = payload
    body = client.get("/api/anomalies?window_weeks=1").get_json()
    assert body["data_quality"]["possible_gap_days_in_window"]
    assert all(r["status"] in ("data_incomplete", "elevated", "insufficient_reference") for r in body["results"])


@pytest.mark.parametrize("query", ["window_weeks=3", "reference_weeks=10", "fdr=0.2", "threshold=abc"])
def test_anomaly_parameter_validation(client, query):
    assert_error(client.get(f"/api/anomalies?{query}"), 400, "invalid_parameter")


def test_anomaly_region_without_activity(client, fake_usgs):
    fake_usgs.query_payload = history_payload()
    assert_error(client.get("/api/anomalies/R-90-180"), 404, "no_activity")


# ------------------------------------------------------------------ aftershocks
def aftershock_handler(n=40, missing=False):
    main = feature("main1", 125.0, 10.0, mag=7.0, time_ms=NOW - 20 * DAY)

    def handler(path, params):
        if "eventid" in params:
            if missing:
                raise UpstreamError("USGS responded with HTTP 404")
            return main
        assert params["maxradiuskm"] > 0 and params["latitude"] == 10.0
        t0 = NOW - 20 * DAY
        times = [t0 + int(DAY * 0.3 * (1.25 ** i) / 10) for i in range(n)]
        evs = [main] + [feature(f"a{i}", 125.05, 10.02, mag=4.6, time_ms=t) for i, t in enumerate(times)]
        evs.append(feature("pre", 125.0, 10.1, mag=4.7, time_ms=t0 - 5 * DAY))
        return collection(*evs)
    return handler


def test_aftershock_analysis(client, fake_usgs):
    fake_usgs.handler = aftershock_handler()
    body = client.get("/api/aftershocks?event_id=main1&min_mag=4.5&pre_days=30&post_days=60").get_json()
    assert body["reference_event"]["id"] == "main1"
    assert body["statistics"]["counts"]["post"] > 0 and body["pre_event_count"] == 1
    assert all(e["id"] != "main1" for e in body["events"])
    assert body["parameters"]["radius_default"] and body["parameters"]["radius_km"] == AS.default_radius_km(7.0)
    assert body["fit"]["status"] in ("fitted", "unreliable", "insufficient_data") and "reasons" in body["fit"]
    assert body["data_window"]["to"] <= app_module.ms_iso(NOW + 60_000)        # never beyond the present
    assert len(body["selection_rules"]) == 5


def test_aftershock_insufficient_data_is_honest(client, fake_usgs):
    fake_usgs.handler = aftershock_handler(n=5)
    fit = client.get("/api/aftershocks?event_id=main1").get_json()["fit"]
    assert fit["status"] == "insufficient_data" and "parameters" not in fit


def test_aftershock_errors(client, fake_usgs):
    fake_usgs.handler = aftershock_handler(missing=True)
    assert_error(client.get("/api/aftershocks?event_id=nope1"), 404, "event_not_found")
    assert_error(client.get("/api/aftershocks?event_id=bad id!"), 400, "invalid_parameter")
    assert_error(client.get("/api/aftershocks"), 400, "invalid_parameter")
    assert_error(client.get("/api/aftershocks?event_id=main1&min_depth=100&max_depth=10"), 400, "out_of_range")
    assert_error(client.get("/api/aftershocks?event_id=main1&radius_km=5000"), 400, "out_of_range")


def test_aftershock_upstream_failure(client, fake_usgs):
    def handler(path, params):
        if "eventid" in params:
            return feature("main1", 125.0, 10.0, mag=7.0, time_ms=NOW - 20 * DAY)
        raise UpstreamError("USGS request timed out", status=504, code="upstream_timeout")
    fake_usgs.handler = handler
    assert_error(client.get("/api/aftershocks?event_id=main1"), 503, "upstream_unavailable")


# ------------------------------------------------------------------ evaluation and health
def test_model_performance_reports_files_verbatim(client):
    body = client.get("/api/model-performance").get_json()
    with open(f"{config.MODEL_DIR}/forecast_m4.5.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert body["forecast"]["4.5"]["test_metrics"] == meta["test_metrics"]
    assert body["forecast"]["4.5"]["periods"]["test"]["from"] == meta["periods"]["test"]["from"]
    assert body["sequences"]["scenarios"] and body["anomaly"]["null_simulation"]
    assert body["aftershock"]["out_of_sample_30_to_90_days"] and len(body["aftershock"]["sequences"]) <= 25
    assert body["legacy_severity_model"]["status"] == "loaded"


def test_health_lists_modules(client):
    m = client.get("/api/health").get_json()["modules"]
    assert m["forecast"]["4.5"]["ready"] and m["anomaly_long_term_dispersion"]["ready"]
    assert all(m["evaluations"].values())
