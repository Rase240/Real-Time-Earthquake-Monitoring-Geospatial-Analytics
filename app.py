"""
TerraWatcher — Real-Time Earthquake Analytics & Geospatial Intelligence.

Flask application: serves the dashboard and a small JSON API.

  GET /                      dashboard (templates/index.html)
  GET /api/health            readiness of model, plate data and upstream cache
  GET /api/earthquakes       observed events from USGS (normalised)
  GET /api/hotspots          observed activity clusters (M4.5+, last 30 days)
  GET /api/predict           conditional severity score for a coordinate
  GET /api/plates            plate boundary GeoJSON for the map overlay
  GET /api/model-info        model metadata and evaluation results

The frontend is served from the same origin, so no CORS headers are emitted.
"""
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

import config
import features as F
import usgs_client
from ml_model import ModelBundle
from usgs_client import UpstreamError

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("terrawatcher")

app = Flask(__name__)
app.json.sort_keys = False

HOTSPOT_CELL_DEG = 2.0
HOTSPOT_MIN_EVENTS = 3
SOURCE_NAME = "USGS FDSN event web service"


# =============================================================================
# Errors
# =============================================================================
class ApiError(Exception):
    def __init__(self, message, status=400, code="bad_request"):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code


def error_response(message, status, code):
    return jsonify({"error": message, "code": code, "status": status}), status


@app.errorhandler(ApiError)
def _api_error(exc):
    return error_response(exc.message, exc.status, exc.code)


@app.errorhandler(UpstreamError)
def _upstream_error(exc):
    return error_response(str(exc), exc.status, exc.code)


@app.errorhandler(HTTPException)
def _http_error(exc):
    if request.path.startswith("/api/"):
        return error_response(exc.description or exc.name, exc.code, exc.name.lower().replace(" ", "_"))
    return exc


@app.errorhandler(Exception)
def _unhandled(exc):
    log.exception("Unhandled error on %s", request.path)
    return error_response("Internal server error", 500, "internal_error")


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if resp.mimetype == "text/html":
        resp.headers.setdefault("Content-Security-Policy", (
            "default-src 'self'; "
            "script-src 'self' https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "img-src 'self' data: https://server.arcgisonline.com; "
            "connect-src 'self'; frame-ancestors 'none'"
        ))
    return resp


# =============================================================================
# Request parsing
# =============================================================================
def parse_number(name, default=None, lo=None, hi=None, cast=float, required=False, aliases=()):
    raw = request.args.get(name)
    for alias in aliases:
        raw = raw if raw not in (None, "") else request.args.get(alias)
    if raw in (None, ""):
        if required:
            raise ApiError(f"Missing required query parameter '{name}'", code="missing_parameter")
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise ApiError(f"Parameter '{name}' must be a {'whole number' if cast is int else 'number'}",
                       code="invalid_parameter") from None
    if isinstance(value, float) and not math.isfinite(value):
        raise ApiError(f"Parameter '{name}' must be a finite number", code="invalid_parameter")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ApiError(f"Parameter '{name}' must be between {lo} and {hi}", code="out_of_range")
    return value


# =============================================================================
# Cache (bounded, thread-safe, keeps the last good copy for stale fallback)
# =============================================================================
class TTLCache:
    def __init__(self, max_entries):
        self._data = OrderedDict()
        self._lock = threading.Lock()
        self.max_entries = max_entries

    def get(self, key):
        """Return (value, age_seconds) or (None, None)."""
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None, None
            self._data.move_to_end(key)
            stored_at, value = item
            return value, time.time() - stored_at

    def put(self, key, value):
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


feed_cache = TTLCache(config.CACHE_MAX_ENTRIES)


def cached_fetch(key, ttl, fetch):
    """
    Return (payload, cache_info). Fresh cache hits skip the upstream call.
    If the upstream fails and a copy younger than STALE_MAX_AGE exists, that copy
    is returned and explicitly marked stale; otherwise the UpstreamError propagates.
    """
    cached, age = feed_cache.get(key)
    if cached is not None and age < ttl:
        return cached, {"hit": True, "age_seconds": int(age), "stale": False, "warning": None}
    try:
        payload = fetch()
    except UpstreamError as exc:
        if cached is not None and age < config.STALE_MAX_AGE:
            log.warning("Serving stale cache for %s after upstream failure: %s", key, exc)
            return cached, {"hit": True, "age_seconds": int(age), "stale": True,
                            "warning": f"USGS is currently unavailable ({exc}). Showing data from "
                                       f"{int(age // 60)} min ago."}
        raise
    feed_cache.put(key, payload)
    return payload, {"hit": False, "age_seconds": 0, "stale": False, "warning": None}


def utc_now():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# Static resources loaded once at startup
# =============================================================================
plate_segments = F.load_plate_segments(config.TECTONIC_FILE)
bundle = ModelBundle(config.MODEL_FILE, config.SCALER_FILE, config.MODEL_META_FILE)


def get_context_catalog():
    """M>=4.5 events from the last 30 days: input for density features and hotspots."""
    def fetch():
        end = utc_now()
        events, _ = usgs_client.fetch_events(end - timedelta(days=F.DENSITY_WINDOW_DAYS), end,
                                             min_mag=F.CATALOG_MIN_MAG, limit=20000)
        return {"events": events, "fetched_at": iso(end), "end_ms": int(end.timestamp() * 1000)}
    return cached_fetch(("context",), config.CONTEXT_CACHE_TTL, fetch)


def score_points(lats, lons, depths, ref_ms, context_events):
    """Build model features for arbitrary points at reference time ref_ms and score them."""
    cat_t = [e["time"] for e in context_events]
    cat_lat = [e["latitude"] for e in context_events]
    cat_lon = [e["longitude"] for e in context_events]
    n = len(lats)
    dens = F.causal_density(cat_t, cat_lat, cat_lon, [ref_ms] * n, lats, lons)
    pdist = F.plate_distance_km(lats, lons, plate_segments)
    X = F.build_feature_matrix(lats, lons, depths, dens, pdist)
    return bundle.predict_proba(X), dens, pdist


# =============================================================================
# Routes
# =============================================================================
@app.route("/")
@app.route("/dashboard-v2")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    _, age = feed_cache.get(("context",))
    ok = bundle.ready and len(plate_segments) > 0
    return jsonify({
        "status": "ok" if ok else "degraded",
        "model_loaded": bundle.ready,
        "model_error": bundle.load_error,
        "plate_segments": int(len(plate_segments)),
        "context_cache_age_seconds": None if age is None else int(age),
        "time": iso(utc_now()),
    })


@app.route("/api/earthquakes")
def earthquakes():
    days = parse_number("days", default=1, lo=1, hi=config.MAX_DAYS, cast=int)
    min_mag = round(parse_number("min_mag", default=2.5, lo=-1.0, hi=10.0), 1)

    def fetch():
        end = utc_now()
        start = end - timedelta(days=days)
        events, stats = usgs_client.fetch_events(start, end, min_mag=min_mag, limit=config.MAX_EVENTS)
        truncated = len(events) >= config.MAX_EVENTS
        total = usgs_client.count_events(start, end, min_mag=min_mag) if truncated else len(events)
        return {"events": events, "count": len(events), "truncated": truncated, "total_available": total,
                "query": {"days": days, "min_mag": min_mag, "start": iso(start), "end": iso(end),
                          "limit": config.MAX_EVENTS},
                "fetched_at": iso(end), **stats}

    payload, cache = cached_fetch(("feed", days, min_mag), config.FEED_CACHE_TTL, fetch)
    return jsonify({**payload, "source": SOURCE_NAME, "cache": cache})


@app.route("/api/hotspots")
def hotspots():
    limit = parse_number("limit", default=12, lo=1, hi=50, cast=int)
    ctx, cache = get_context_catalog()
    events = ctx["events"]

    cells = {}
    for e in events:
        key = (math.floor(e["latitude"] / HOTSPOT_CELL_DEG), math.floor(e["longitude"] / HOTSPOT_CELL_DEG))
        cells.setdefault(key, []).append(e)
    ranked = sorted((c for c in cells.items() if len(c[1]) >= HOTSPOT_MIN_EVENTS),
                    key=lambda kv: (len(kv[1]), max((x["mag"] or 0) for x in kv[1])), reverse=True)[:limit]

    spots = []
    for (ci, cj), evs in ranked:
        depths = [x["depth_km"] for x in evs if x["depth_km"] is not None]
        mags = [x["mag"] for x in evs if x["mag"] is not None]
        spots.append({
            "latitude": round(float(np.mean([x["latitude"] for x in evs])), 3),
            "longitude": round(float(np.mean([x["longitude"] for x in evs])), 3),
            "cell": {"lat_min": ci * HOTSPOT_CELL_DEG, "lat_max": (ci + 1) * HOTSPOT_CELL_DEG,
                     "lon_min": cj * HOTSPOT_CELL_DEG, "lon_max": (cj + 1) * HOTSPOT_CELL_DEG},
            "event_count": len(evs),
            "max_mag": max(mags) if mags else None,
            "latest_time": max(x["time"] for x in evs),
            "median_depth_km": round(float(np.median(depths)), 1) if depths else None,
            "model_probability": None,
        })

    if bundle.ready and spots:
        ok = [s for s in spots if s["median_depth_km"] is not None]
        if ok:
            p, _, _ = score_points([s["latitude"] for s in ok], [s["longitude"] for s in ok],
                                   [s["median_depth_km"] for s in ok], ctx["end_ms"], events)
            for s, prob in zip(ok, p, strict=True):
                s["model_probability"] = round(float(prob), 4)

    return jsonify({
        "hotspots": spots,
        "definition": (f"{HOTSPOT_CELL_DEG:g} degree grid cells with the most M{F.CATALOG_MIN_MAG}+ events "
                       f"recorded in the last {F.DENSITY_WINDOW_DAYS} days (observed activity, not a forecast)."),
        "window_days": F.DENSITY_WINDOW_DAYS, "min_mag": F.CATALOG_MIN_MAG, "cell_deg": HOTSPOT_CELL_DEG,
        "catalog_events": len(events), "model_status": "loaded" if bundle.ready else "unavailable",
        "source": SOURCE_NAME, "fetched_at": ctx["fetched_at"], "cache": cache,
    })


@app.route("/api/predict")
def predict():
    lat = parse_number("lat", lo=-90.0, hi=90.0, required=True, aliases=("latitude",))
    lon = parse_number("lon", lo=-180.0, hi=180.0, required=True, aliases=("lng", "longitude"))
    depth = parse_number("depth", default=10.0, lo=0.0, hi=700.0)

    if not bundle.ready:
        raise ApiError(bundle.load_error or "Model not loaded", status=503, code="model_unavailable")
    try:
        ctx, cache = get_context_catalog()
    except UpstreamError as exc:
        raise ApiError(f"Recent-activity feature unavailable because USGS could not be reached ({exc}). "
                       "No score was computed.", status=503, code="context_unavailable") from exc

    p, dens, pdist = score_points([lat], [lon], [depth], ctx["end_ms"], ctx["events"])
    prob = float(p[0])
    ratio, band = bundle.relative_band(prob)
    return jsonify({
        "probability": round(prob, 4),
        "base_rate": bundle.base_rate,
        "relative_to_base_rate": ratio,
        "band": band,
        "features": {
            "latitude": lat, "longitude": lon, "depth_km": depth,
            "local_density_30d": int(dens[0]), "plate_distance_km": round(float(pdist[0]), 1),
        },
        "interpretation": (f"If an M{F.CATALOG_MIN_MAG}+ earthquake were recorded at this point and depth, "
                           f"the model's estimated chance that it would be M{F.SIGNIFICANT_MAG}+. "
                           "This is not a probability that an earthquake will occur."),
        "model": {"trained_at": bundle.meta.get("trained_at"),
                  "test_roc_auc": bundle.meta.get("metrics", {}).get("test", {}).get("roc_auc")},
        "context": {"fetched_at": ctx["fetched_at"], "stale": cache["stale"]},
    })


@app.route("/api/plates")
def plates():
    """Plate boundary polylines (static reference data, Mueller et al. compilation) for the map overlay."""
    if not os.path.exists(config.TECTONIC_FILE):
        raise ApiError("Plate boundary data is not installed", status=404, code="plates_unavailable")
    resp = send_file(config.TECTONIC_FILE, mimetype="application/json", max_age=86400)
    return resp


@app.route("/api/model-info")
def model_info():
    return jsonify(bundle.describe())


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
