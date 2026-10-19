"""
Shared fixtures. Every test that touches USGS uses a fake transport, so the
suite runs offline and deterministically. Live-network checks are done
separately (see README "Verification").
"""
import copy

import pytest

import app as app_module
import usgs_client

NOW_MS = 1_790_000_000_000  # fixed reference time used by synthetic records
DAY = 86_400_000


def feature(event_id, lon, lat, depth=10.0, mag=4.6, time_ms=NOW_MS, **props):
    coords = [lon, lat] if depth is None else [lon, lat, depth]
    p = {"mag": mag, "place": f"Test place {event_id}", "time": time_ms, "updated": time_ms,
         "magType": "mb", "type": "earthquake", "status": "reviewed", "tsunami": 0, "net": "us",
         "url": f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}"}
    p.update(props)
    return {"type": "Feature", "id": event_id, "properties": p,
            "geometry": {"type": "Point", "coordinates": coords}}


def collection(*features):
    return {"type": "FeatureCollection", "metadata": {"status": 200}, "features": list(features)}


class FakeUSGS:
    """Replacement for usgs_client._get that records calls and returns canned payloads."""

    def __init__(self):
        self.query_payload = collection()
        self.count_payload = {"count": 0}
        self.error = None
        self.handler = None      # optional callable(path, params) -> payload, for tests that need per-query answers
        self.calls = []

    def __call__(self, path, params, timeout=None, retries=None, session=None):
        self.calls.append((path, dict(params)))
        if self.error is not None:
            raise self.error
        if self.handler is not None:
            return copy.deepcopy(self.handler(path, dict(params)))
        return copy.deepcopy(self.query_payload if path == "query" else self.count_payload)


@pytest.fixture
def fake_usgs(monkeypatch):
    fake = FakeUSGS()
    monkeypatch.setattr(usgs_client, "_get", fake)
    app_module.feed_cache.clear()
    app_module.analysis_cache.clear()
    yield fake
    app_module.feed_cache.clear()
    app_module.analysis_cache.clear()


@pytest.fixture
def client(fake_usgs):
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
