"""Parsing, validation and failure handling for the USGS client."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import requests

import usgs_client
from usgs_client import UpstreamError, normalize_collection, normalize_feature
from tests.conftest import collection, feature


def test_normalize_valid_feature_maps_fields():
    e = normalize_feature(feature("us1", 142.5, 38.1, depth=24.3, mag=5.1, time_ms=1_700_000_000_000))
    assert e["id"] == "us1"
    assert (e["latitude"], e["longitude"], e["depth_km"]) == (38.1, 142.5, 24.3)  # GeoJSON order is lon, lat
    assert e["mag"] == 5.1 and e["mag_type"] == "mb"
    assert e["time_iso"] == "2023-11-14T22:13:20.000Z"
    assert e["tsunami"] is False


def test_two_dimensional_coordinates_give_unknown_depth_not_a_default():
    e = normalize_feature(feature("us2", 10.0, 20.0, depth=None))
    assert e["depth_km"] is None


def test_null_magnitude_is_preserved_as_none():
    assert normalize_feature(feature("us3", 0.5, 0.5, mag=None))["mag"] is None


@pytest.mark.parametrize("coords", [[], [10.0], [None, 5.0], [200.0, 10.0], [10.0, -95.0], ["x", "y"], [float("nan"), 1.0]])
def test_invalid_coordinates_are_rejected(coords):
    f = feature("bad", 0, 0)
    f["geometry"]["coordinates"] = coords
    assert normalize_feature(f) is None


def test_missing_time_is_rejected():
    f = feature("t", 1, 1)
    f["properties"]["time"] = None
    assert normalize_feature(f) is None


def test_collection_drops_invalid_and_duplicates():
    bad = feature("bad", 0, 0)
    bad["geometry"] = None
    events, stats = normalize_collection(collection(feature("a", 1, 1), feature("a", 1, 1), bad, feature("b", 2, 2)))
    assert [e["id"] for e in events] == ["a", "b"]
    assert stats == {"dropped_invalid": 1, "duplicates_removed": 1}


@pytest.mark.parametrize("payload", [None, [], {"type": "FeatureCollection"}, {"features": "nope"}])
def test_malformed_collection_raises(payload):
    with pytest.raises(UpstreamError) as exc:
        normalize_collection(payload)
    assert exc.value.code == "upstream_malformed"


def _resp(status, body=None, headers=None, bad_json=False):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    if bad_json:
        r.json.side_effect = ValueError("not json")
    else:
        r.json.return_value = body
    return r


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(usgs_client.time, "sleep", lambda s: None)


def test_query_uses_full_utc_timestamps(monkeypatch):
    get = MagicMock(return_value=_resp(200, collection()))
    monkeypatch.setattr(usgs_client.requests, "get", get)
    start = datetime(2026, 1, 1, 6, 30, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, 6, 30, tzinfo=timezone.utc)
    usgs_client.fetch_events(start, end, min_mag=2.5, limit=100)
    params = get.call_args.kwargs["params"]
    assert params["starttime"] == "2026-01-01T06:30:00"
    assert params["endtime"] == "2026-01-02T06:30:00"   # not truncated to midnight
    assert params["minmagnitude"] == 2.5 and params["limit"] == 100
    assert get.call_args.kwargs["timeout"] > 0


def test_timeout_is_retried_then_reported_as_504(monkeypatch, no_sleep):
    get = MagicMock(side_effect=requests.exceptions.Timeout())
    monkeypatch.setattr(usgs_client.requests, "get", get)
    with pytest.raises(UpstreamError) as exc:
        usgs_client._get("query", {}, retries=1)
    assert exc.value.status == 504 and get.call_count == 2


def test_server_error_then_success(monkeypatch, no_sleep):
    get = MagicMock(side_effect=[_resp(503), _resp(200, collection())])
    monkeypatch.setattr(usgs_client.requests, "get", get)
    assert usgs_client._get("query", {}, retries=1) == collection()


def test_rate_limit_exhausts_retries(monkeypatch, no_sleep):
    get = MagicMock(return_value=_resp(429, headers={"Retry-After": "2"}))
    monkeypatch.setattr(usgs_client.requests, "get", get)
    with pytest.raises(UpstreamError) as exc:
        usgs_client._get("query", {}, retries=2)
    assert exc.value.code == "upstream_rate_limited" and get.call_count == 3


def test_bad_request_is_not_retried(monkeypatch, no_sleep):
    get = MagicMock(return_value=_resp(400))
    monkeypatch.setattr(usgs_client.requests, "get", get)
    with pytest.raises(UpstreamError):
        usgs_client._get("query", {}, retries=3)
    assert get.call_count == 1


def test_invalid_json_is_reported(monkeypatch):
    monkeypatch.setattr(usgs_client.requests, "get", MagicMock(return_value=_resp(200, bad_json=True)))
    with pytest.raises(UpstreamError) as exc:
        usgs_client._get("query", {}, retries=0)
    assert exc.value.code == "upstream_malformed"


def test_connection_error(monkeypatch):
    monkeypatch.setattr(usgs_client.requests, "get", MagicMock(side_effect=requests.exceptions.ConnectionError()))
    with pytest.raises(UpstreamError) as exc:
        usgs_client._get("query", {}, retries=0)
    assert exc.value.code == "upstream_unreachable"
