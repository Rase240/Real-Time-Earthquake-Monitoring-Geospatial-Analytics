"""
TerraWatcher — Real-Time Earthquake Analytics & Geospatial Intelligence.

Flask application: serves the dashboard and a JSON API.

  GET /                          dashboard (templates/index.html)
  GET /api/health                readiness of models, plate data and upstream caches
  GET /api/earthquakes           observed events from USGS (normalised)
  GET /api/hotspots              observed activity clusters (M4.5+, last 30 days)
  GET /api/sequences             Module A: space-time DBSCAN sequences of the event feed
  GET /api/forecast              Module B: next-7-day count forecast for every eligible region
  GET /api/forecast/<region>     Module B: one region with history, backtest and features
  GET /api/anomalies             Module C: elevated observed activity per region
  GET /api/anomalies/<region>    Module C: one region with weekly counts and contributing events
  GET /api/aftershocks           Module D: event-centred sequence analysis and Omori-Utsu fit
  GET /api/model-performance     evaluation results of every module (read from models/*.json)
  GET /api/predict               legacy conditional severity score for a coordinate
  GET /api/plates                plate boundary GeoJSON for the map overlay
  GET /api/model-info            legacy severity model metadata and evaluation results

The frontend is served from the same origin, so no CORS headers are emitted.
"""
import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import HTTPException

import config
import features as F
import usgs_client
from analytics import aftershock as AS
from analytics import anomaly as AN
from analytics import regional_features as RF
from analytics import regions as RG
from analytics import sequences as SQ
from analytics.catalog import events_to_frame
from analytics.forecast import ForecastModel, interval, prob_at_least
from analytics.forecast_training import THRESHOLDS as FORECAST_THRESHOLDS
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


@app.context_processor
def _asset_urls():
    """asset('x.js') -> /static/x.js?v=<mtime>, so browsers never run a stale script after an update."""
    def asset(name):
        try:
            version = int(os.path.getmtime(os.path.join(app.static_folder, name)))
        except OSError:
            version = 0
        return url_for("static", filename=name, v=version)
    return {"asset": asset}


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
analysis_cache = TTLCache(config.ANALYSIS_CACHE_MAX_ENTRIES)   # derived results, keyed by data version + parameters
_fetch_locks, _fetch_locks_guard = {}, threading.Lock()


def _lock_for(key):
    """One lock per cache key, so concurrent cold requests trigger a single upstream download."""
    with _fetch_locks_guard:
        if len(_fetch_locks) > 256:
            _fetch_locks.clear()
        return _fetch_locks.setdefault(key, threading.Lock())


def cached_fetch(key, ttl, fetch, stale_max=None):
    """
    Return (payload, cache_info). Fresh cache hits skip the upstream call.
    If the upstream fails and a copy younger than stale_max (default STALE_MAX_AGE)
    exists, that copy is returned and explicitly marked stale; otherwise the
    UpstreamError propagates.
    """
    stale_max = config.STALE_MAX_AGE if stale_max is None else stale_max
    cached, age = feed_cache.get(key)
    if cached is not None and age < ttl:
        return cached, {"hit": True, "age_seconds": int(age), "stale": False, "warning": None}
    with _lock_for(key):
        cached, age = feed_cache.get(key)          # another request may have refreshed it meanwhile
        if cached is not None and age < ttl:
            return cached, {"hit": True, "age_seconds": int(age), "stale": False, "warning": None}
        try:
            payload = fetch()
        except UpstreamError as exc:
            if cached is not None and age < stale_max:
                log.warning("Serving stale cache for %s after upstream failure: %s", key, exc)
                return cached, {"hit": True, "age_seconds": int(age), "stale": True,
                                "warning": f"USGS is currently unavailable ({exc}). Showing data from "
                                           f"{int(age // 60)} min ago."}
            raise
        feed_cache.put(key, payload)
    return payload, {"hit": False, "age_seconds": 0, "stale": False, "warning": None}


def memoized(key, compute):
    """Analysis results are deterministic given their key (which includes the data version)."""
    value, _ = analysis_cache.get(key)
    if value is None:
        value = compute()
        analysis_cache.put(key, value)
    return value


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
    _, hist_age = feed_cache.get(("history",))
    modules = {
        "forecast": {f"{t:.1f}": {"ready": m.ready, "error": m.load_error} for t, m in forecast_models.items()},
        "anomaly_long_term_dispersion": {"ready": bool(anomaly_alpha), "error": anomaly_ref_error},
        "evaluations": {name: value is not None for name, value in evaluations.items()},
    }
    ok = (bundle.ready and len(plate_segments) > 0 and all(m.ready for m in forecast_models.values())
          and bool(anomaly_alpha))
    return jsonify({
        "status": "ok" if ok else "degraded",
        "model_loaded": bundle.ready,
        "model_error": bundle.load_error,
        "plate_segments": int(len(plate_segments)),
        "modules": modules,
        "context_cache_age_seconds": None if age is None else int(age),
        "history_cache_age_seconds": None if hist_age is None else int(hist_age),
        "time": iso(utc_now()),
    })


def feed_params():
    days = parse_number("days", default=1, lo=1, hi=config.MAX_DAYS, cast=int)
    min_mag = round(parse_number("min_mag", default=2.5, lo=-1.0, hi=10.0), 1)
    return days, min_mag


def get_feed(days, min_mag):
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

    return cached_fetch(("feed", days, min_mag), config.FEED_CACHE_TTL, fetch)


@app.route("/api/earthquakes")
def earthquakes():
    payload, cache = get_feed(*feed_params())
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


# =============================================================================
# Analytical modules
# =============================================================================
WEEK_MS = RG.WEEK_MS
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
EVENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,40}$")
HISTORY_MIN_MAG = 4.5


def _load_json(name):
    path = os.path.join(config.MODEL_DIR, name)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("Evaluation file %s unavailable: %s", name, exc)
        return None


forecast_models = {t: ForecastModel(config.MODEL_DIR, t) for t in FORECAST_THRESHOLDS}
anomaly_alpha, anomaly_ref_period, anomaly_ref_error = AN.load_reference(
    os.path.join(config.MODEL_DIR, "anomaly_reference.json"))
evaluations = {name: _load_json(f"{name}_eval.json") for name in ("sequence", "anomaly", "aftershock")}
forecast_backtest = _load_json("forecast_backtest.json") or {}


def parse_choice(name, choices, default, cast=float):
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        value = None
    if value not in choices:
        allowed = ", ".join(f"{c:g}" if isinstance(c, float) else str(c) for c in choices)
        raise ApiError(f"Parameter '{name}' must be one of: {allowed}", code="invalid_parameter")
    return value


def parse_region(region_id):
    idx = RG.parse_region_id(region_id)
    if idx is None:
        raise ApiError("Unknown region id; expected e.g. 'R+30+140' (10-degree cell lower-left corner)",
                       status=404, code="unknown_region")
    return idx


def get_history():
    """Global M4.5+ events of the last HISTORY_DAYS days, validated, shared by forecasts and anomalies."""
    def fetch():
        end = utc_now()
        start = end - timedelta(days=config.HISTORY_DAYS)
        events, stats = usgs_client.fetch_events(start, end, min_mag=HISTORY_MIN_MAG, limit=config.USGS_QUERY_LIMIT,
                                                 timeout=max(config.USGS_TIMEOUT, 60))
        if len(events) >= config.USGS_QUERY_LIMIT:
            raise UpstreamError("USGS returned its maximum number of events; the history would be incomplete",
                                code="upstream_truncated")
        frame, quality = events_to_frame(events)
        return {"frame": frame, "events": {e["id"]: e for e in events}, "start_ms": int(start.timestamp() * 1000),
                "end_ms": int(end.timestamp() * 1000), "fetched_at": iso(end), "quality": {**stats, **quality}}
    try:
        return cached_fetch(("history",), config.HISTORY_CACHE_TTL, fetch, stale_max=config.HISTORY_STALE_MAX_AGE)
    except UpstreamError as exc:
        raise ApiError(f"Historical USGS data could not be loaded ({exc}). No result was computed.",
                       status=503, code="history_unavailable") from exc


def ms_iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _data_window(hist, cache, used_from, used_to):
    return {"source": SOURCE_NAME, "min_mag_downloaded": HISTORY_MIN_MAG, "fetched_at": hist["fetched_at"],
            "used_from": ms_iso(used_from), "used_to": ms_iso(used_to), "events_downloaded": int(len(hist["frame"])),
            "latest_event": ms_iso(int(hist["frame"]["time"].max())) if len(hist["frame"]) else None,
            "stale": cache["stale"], "warning": cache["warning"], **hist["quality"]}


# ------------------------------------------------------------------ sequences
@app.route("/api/sequences")
def sequences_route():
    days, min_mag = feed_params()
    eps_km = parse_number("eps_km", default=SQ.DEFAULTS["eps_km"], lo=SQ.LIMITS["eps_km"][0], hi=SQ.LIMITS["eps_km"][1])
    eps_hours = parse_number("eps_hours", default=SQ.DEFAULTS["eps_hours"], lo=SQ.LIMITS["eps_hours"][0],
                             hi=SQ.LIMITS["eps_hours"][1])
    min_samples = parse_number("min_samples", default=SQ.DEFAULTS["min_samples"], lo=SQ.LIMITS["min_samples"][0],
                               hi=SQ.LIMITS["min_samples"][1], cast=int)
    feed, cache = get_feed(days, min_mag)
    key = ("sequences", days, min_mag, feed["fetched_at"], round(eps_km, 2), round(eps_hours, 2), min_samples)
    result = memoized(key, lambda: SQ.detect(feed["events"], eps_km, eps_hours, min_samples))
    events = [{"id": e["id"], "time": e["time"], "latitude": e["latitude"], "longitude": e["longitude"],
               "depth_km": e["depth_km"], "mag": e["mag"], "place": e["place"], "url": e["url"],
               "sequence": result["assignments"].get(e["id"])} for e in feed["events"]]
    return jsonify({**{k: v for k, v in result.items() if k != "assignments"}, "events": events,
                    "query": feed["query"], "truncated": feed["truncated"], "total_available": feed["total_available"],
                    "source": SOURCE_NAME, "fetched_at": feed["fetched_at"], "cache": cache})


# ------------------------------------------------------------------ forecasts
def _forecast_model(threshold):
    model = forecast_models[threshold]
    if not model.ready:
        raise ApiError(model.load_error, status=503, code="model_unavailable")
    return model


def compute_forecast(threshold):
    model = _forecast_model(threshold)
    hist, cache = get_history()
    origin = hist["end_ms"] // DAY_MS * DAY_MS          # start of the current UTC day

    def compute():
        f = hist["frame"]
        C, L, epoch = RF.live_matrices(f["time"], f["mag"], f["latitude"], f["longitude"], threshold, origin)
        if epoch < hist["start_ms"]:
            raise ApiError("Not enough history downloaded for the 52-week features", status=503, code="insufficient_data")
        reg = model.region_index
        X = RF.features_at(C, L, [RF.LOOKBACK_WEEKS], reg)
        mu = model.predict(X)
        a = model.alpha
        lo80, hi80 = interval(mu, a, 0.8)
        lo95, hi95 = interval(mu, a, 0.95)
        p1 = prob_at_least(1, mu, a)
        rows = []
        for i, info in enumerate(model.meta["regions"]):
            rows.append({"region": info["id"], "label": info["label"], "bounds": info["bounds"],
                         "expected": round(float(mu[i]), 3),
                         "interval_80": [int(lo80[i]), int(hi80[i])], "interval_95": [int(lo95[i]), int(hi95[i])],
                         "prob_at_least_one": round(float(p1[i]), 4),
                         "last_week_observed": int(X[i, RF.FEATURE_NAMES.index("lag1")]),
                         "baselines": {"historical_mean_2001_2015": info["train_weekly_mean"],
                                       "rolling_mean_4w": round(float(X[i, RF.FEATURE_NAMES.index("mean_4w")]), 3),
                                       "rolling_mean_52w": round(float(X[i, RF.FEATURE_NAMES.index("mean_52w")]), 3)}})
        series = {info["id"]: {"weekly": C[r].astype(int).tolist(), "features": dict(zip(RF.FEATURE_NAMES,
                                                                                        map(float, X[i]), strict=True))}
                  for i, (r, info) in enumerate(zip(reg, model.meta["regions"], strict=True))}
        return rows, series, epoch

    rows, series, epoch = memoized(("forecast", threshold, origin, hist["fetched_at"]), compute)
    meta = model.meta
    sel = meta["selected_model"]
    summary = {
        "threshold": threshold, "horizon_days": 7,
        "origin": ms_iso(origin), "valid_from": ms_iso(origin), "valid_to": ms_iso(origin + 7 * DAY_MS),
        "target": meta["target"],
        "model": {"name": sel, "trained_at": meta["trained_at"], "alpha": meta["alpha"],
                  "interval_method": "Negative-binomial predictive quantiles (dispersion fitted on validation data)",
                  "test_period": meta["periods"]["test"]["from"] + " to " + meta["periods"]["test"]["to"],
                  "test_metrics": {k: meta["test_metrics"][k] for k in (sel, "historical_mean", "rolling_mean_4w",
                                                                         "rolling_mean_52w")},
                  "training_window": meta["periods"]["served_model_fit"]},
        "data_window": _data_window(hist, cache, epoch, origin),
        "interpretation": ("Expected number of qualifying earthquakes in each 10-degree cell over the next 7 days, "
                           "from recent counts in and around the cell. It is an aggregate statistical forecast: it "
                           "does not say when, where in the cell, or how large any earthquake will be."),
    }
    return summary, rows, series


@app.route("/api/forecast")
def forecast_route():
    threshold = parse_choice("threshold", FORECAST_THRESHOLDS, FORECAST_THRESHOLDS[0])
    summary, rows, _ = compute_forecast(threshold)
    return jsonify({**summary, "regions": rows})


@app.route("/api/forecast/<region_id>")
def forecast_region_route(region_id):
    threshold = parse_choice("threshold", FORECAST_THRESHOLDS, FORECAST_THRESHOLDS[0])
    idx = parse_region(region_id)
    summary, rows, series = compute_forecast(threshold)
    rid = RG.region_id(idx)
    row = next((r for r in rows if r["region"] == rid), None)
    if row is None:
        raise ApiError(f"Region {rid} has too little M{threshold:g}+ history for a forecast "
                       f"({forecast_models[threshold].meta['eligibility']}).", status=404, code="insufficient_history")
    meta = forecast_models[threshold].meta
    origin = int(datetime.strptime(summary["origin"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)
    weeks = [ms_iso(origin - (RF.LOOKBACK_WEEKS - i) * WEEK_MS)[:10] for i in range(RF.LOOKBACK_WEEKS)]
    back = (forecast_backtest.get(f"{threshold:.1f}") or {})
    region_eval = next((r for r in meta["test_by_region"] if r["id"] == rid), None)
    return jsonify({**summary, "region": row,
                    "observed_weekly": {"week_starts": weeks, "counts": series[rid]["weekly"]},
                    "features": [{**f, "value": series[rid]["features"][f["name"]]} for f in RF.FEATURES],
                    "backtest": {"week_starts": back.get("weeks", []), **(back.get("regions", {}).get(rid) or {}),
                                 "note": "Walk-forward test-period forecasts (out of sample) from the evaluation run."},
                    "region_test_metrics": region_eval})


# ------------------------------------------------------------------ anomalies
def anomaly_params():
    return (parse_choice("threshold", FORECAST_THRESHOLDS, FORECAST_THRESHOLDS[0]),
            parse_choice("window_weeks", AN.WINDOW_WEEKS, AN.DEFAULTS["window_weeks"], cast=int),
            parse_choice("reference_weeks", AN.REFERENCE_WEEKS, AN.DEFAULTS["reference_weeks"], cast=int),
            parse_choice("fdr", AN.FDR_LEVELS, AN.DEFAULTS["fdr"]))


def compute_anomalies(threshold, window, ref, fdr):
    hist, cache = get_history()
    end = hist["end_ms"] // HOUR_MS * HOUR_MS
    epoch = end - (ref + window) * WEEK_MS
    obs_start = end - window * WEEK_MS

    def compute():
        f = hist["frame"]
        t = f["time"].to_numpy()
        C = RG.count_matrix(t, f["mag"], f["latitude"], f["longitude"], threshold, epoch, ref + window)
        gaps_win, gaps_ref = AN.gap_days(t, obs_start, end), AN.gap_days(t, epoch, obs_start)
        results, tested = AN.detect(C, window, ref, fdr, AN.DEFAULTS["min_reference_events"],
                                    gaps_in_window=len(gaps_win), gaps_in_reference=len(gaps_ref),
                                    long_alpha=anomaly_alpha.get(threshold))
        in_win = (t >= obs_start) & (t < end) & (f["mag"].to_numpy(float) >= threshold - RG.MAG_EPS)
        reg_of = RG.region_index(f["latitude"].to_numpy(), f["longitude"].to_numpy())
        for row in results:
            row["label"], row["bounds"] = RG.label(row["index"]), RG.bounds(row["index"])
            sel = f[in_win & (reg_of == row["index"])]
            if len(sel) and sel["mag"].notna().any():
                big = sel.loc[sel["mag"].idxmax()]
                row["largest_event"] = {"id": big["id"], "mag": float(big["mag"]), "time": int(big["time"]),
                                        "place": hist["events"].get(big["id"], {}).get("place")}
        weekly = {r["region"]: C[r["index"]].astype(int).tolist() for r in results}
        return {"results": results, "tested": tested, "weekly": weekly,
                "gaps_window": gaps_win, "gaps_reference": gaps_ref}

    out = memoized(("anomaly", threshold, window, ref, fdr, hist["fetched_at"]), compute)
    summary = {
        "threshold": threshold, "window_weeks": window, "reference_weeks": ref, "fdr": fdr,
        "observation_window": {"from": ms_iso(obs_start), "to": ms_iso(end)},
        "reference_window": {"from": ms_iso(epoch), "to": ms_iso(obs_start)},
        "regions_tested": out["tested"],
        "regions_elevated": sum(r["status"] == "elevated" for r in out["results"]),
        "data_quality": {"possible_gap_days_in_window": [ms_iso(d)[:10] for d in out["gaps_window"]],
                         "possible_gap_days_in_reference": [ms_iso(d)[:10] for d in out["gaps_reference"]],
                         "long_term_dispersion": ("2001-2015 per-region estimate" if anomaly_alpha.get(threshold) is not None
                                                  else anomaly_ref_error)},
        "data_window": _data_window(hist, cache, epoch, end),
        "method": ("For each 10-degree cell, the observed count in the observation window is compared with the "
                   "same cell's weekly counts in the reference window, modelled as negative binomial (dispersion: "
                   "the larger of the reference-window and 2001-2015 maximum-likelihood estimates). The one-sided "
                   "p-value is conditional on the total of both windows (beta-binomial), so it does not depend on "
                   "an estimated mean. p-values are adjusted across all tested cells with Benjamini-Hochberg; a "
                   f"cell is 'elevated' when q <= FDR and at least {AN.MIN_OBSERVED} events were observed."),
        "interpretation": ("Elevated observed activity means more events than this cell's recent history makes "
                           "likely, usually because of an ongoing sequence. It is not a prediction of a future "
                           "large earthquake."),
    }
    return summary, out


@app.route("/api/anomalies")
def anomalies_route():
    summary, out = compute_anomalies(*anomaly_params())
    return jsonify({**summary, "results": out["results"]})


@app.route("/api/anomalies/<region_id>")
def anomaly_region_route(region_id):
    threshold, window, ref, fdr = anomaly_params()
    idx = parse_region(region_id)
    summary, out = compute_anomalies(threshold, window, ref, fdr)
    rid = RG.region_id(idx)
    row = next((r for r in out["results"] if r["region"] == rid), None)
    if row is None:
        raise ApiError(f"Region {rid} had no qualifying events in the reference or observation window.",
                       status=404, code="no_activity")
    hist, _ = get_history()
    f = hist["frame"]
    obs_from = int(datetime.strptime(summary["observation_window"]["from"], "%Y-%m-%dT%H:%M:%SZ")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    obs_to = obs_from + window * WEEK_MS
    mask = ((f["time"] >= obs_from) & (f["time"] < obs_to) & (f["mag"] >= threshold - RG.MAG_EPS)
            & (RG.region_index(f["latitude"].to_numpy(), f["longitude"].to_numpy()) == idx))
    events = [hist["events"][i] for i in f.loc[mask, "id"]][::-1][:500]
    week_ends = [ms_iso(obs_to - (ref + window - 1 - i) * WEEK_MS) for i in range(ref + window)]
    return jsonify({**summary, "region": row, "weekly": {"week_ends": week_ends, "counts": out["weekly"][rid],
                                                         "reference_weeks": ref, "window_weeks": window},
                    "contributing_events": events})


# ------------------------------------------------------------------ aftershocks
@app.route("/api/aftershocks")
def aftershocks_route():
    event_id = (request.args.get("event_id") or "").strip()
    if not EVENT_ID_RE.match(event_id):
        raise ApiError("Parameter 'event_id' must be a USGS event id (letters, digits, '-' or '_')",
                       code="invalid_parameter")
    lim = AS.LIMITS
    pre_days = parse_number("pre_days", default=AS.DEFAULTS["pre_days"], lo=lim["pre_days"][0], hi=lim["pre_days"][1], cast=int)
    post_days = parse_number("post_days", default=AS.DEFAULTS["post_days"], lo=lim["post_days"][0],
                             hi=lim["post_days"][1], cast=int)
    min_mag = round(parse_number("min_mag", default=AS.DEFAULTS["min_mag"], lo=0.0, hi=8.0), 1)
    radius = parse_number("radius_km", default=None, lo=lim["radius_km"][0], hi=lim["radius_km"][1])
    min_depth = parse_number("min_depth", default=None, lo=0.0, hi=800.0)
    max_depth = parse_number("max_depth", default=None, lo=0.0, hi=800.0)
    if min_depth is not None and max_depth is not None and min_depth > max_depth:
        raise ApiError("'min_depth' must not exceed 'max_depth'", code="out_of_range")
    fit_start = parse_number("fit_start_hours", default=AS.DEFAULTS["fit_start_hours"], lo=lim["fit_start_hours"][0],
                             hi=lim["fit_start_hours"][1])

    def fetch_ref():
        ev = usgs_client.fetch_event(event_id)
        if ev is None:
            raise ApiError(f"USGS has no event with id '{event_id}'", status=404, code="event_not_found")
        return ev
    ref, _ = cached_fetch(("event", event_id), config.AFTERSHOCK_CACHE_TTL, fetch_ref)
    radius = radius if radius is not None else AS.default_radius_km(ref["mag"])
    now_ms = int(utc_now().timestamp() * 1000)
    if ref["time"] > now_ms:
        raise ApiError("Reference event time is in the future", status=502, code="upstream_malformed")
    start_ms = ref["time"] - pre_days * DAY_MS
    end_ms = min(ref["time"] + post_days * DAY_MS, now_ms)

    def fetch_circle():
        start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        end = datetime.fromtimestamp((end_ms + 1000) / 1000, tz=timezone.utc)
        events, stats = usgs_client.fetch_events(start, end, min_mag=min_mag, limit=config.USGS_QUERY_LIMIT,
                                                 circle=(ref["latitude"], ref["longitude"], radius),
                                                 timeout=max(config.USGS_TIMEOUT, 60))
        return {"events": events, "fetched_at": iso(utc_now()), "end_ms": end_ms, "stats": stats,
                "truncated": len(events) >= config.USGS_QUERY_LIMIT}
    try:
        circle, cache = cached_fetch(("circle", event_id, round(radius, 1), pre_days, post_days, min_mag),
                                     config.AFTERSHOCK_CACHE_TTL, fetch_circle)
    except UpstreamError as exc:
        raise ApiError(f"Events around the reference event could not be loaded ({exc}). No analysis was computed.",
                       status=503, code="upstream_unavailable") from exc
    key = ("aftershock", event_id, radius, pre_days, post_days, min_mag, min_depth, max_depth, fit_start,
           circle["fetched_at"])
    result = memoized(key, lambda: AS.analyse(circle["events"], ref, radius, pre_days, post_days, min_mag, fit_start,
                                              min_depth, max_depth, circle["end_ms"], circle["truncated"]))
    return jsonify({
        "reference_event": ref,
        "parameters": {"radius_km": radius, "radius_default": request.args.get("radius_km") in (None, ""),
                       "pre_days": pre_days, "post_days": post_days, "min_mag": min_mag, "min_depth": min_depth,
                       "max_depth": max_depth, "fit_start_hours": fit_start},
        "selection_rules": [
            f"Epicentre within {radius:g} km (great circle) of the reference epicentre.",
            f"Magnitude >= {min_mag:g}; events without magnitude are excluded.",
            "Depth limits apply when set; events without reported depth are then excluded.",
            f"Pre-event window {pre_days} days before, post-event window up to {post_days} days after the reference time.",
            "The reference event itself is excluded. Nearby events are not assumed to be caused by it.",
        ],
        **result,
        "data_window": {"from": ms_iso(start_ms), "to": ms_iso(end_ms), "truncated": circle["truncated"],
                        "fetched_at": circle["fetched_at"], "stale": cache["stale"], "warning": cache["warning"],
                        "source": SOURCE_NAME,
                        "note": ("Zero-count bins are periods with no catalogue event meeting the criteria; outages "
                                 "of regional networks cannot be detected from this query.")},
        "model_note": ("Modified Omori-Utsu rate mu + K/(t+c)^p fitted by maximum likelihood after the first "
                       f"{fit_start:g} h (catalogues miss many early events). mu is fixed to the pre-event rate. "
                       "The curve is shown only inside the fitted interval and only when every check passes."),
    })


# ------------------------------------------------------------------ evaluation
@app.route("/api/model-performance")
def model_performance():
    forecasts = {}
    for t, m in forecast_models.items():
        if not m.ready:
            forecasts[f"{t:.1f}"] = {"status": "unavailable", "error": m.load_error}
            continue
        meta = m.meta
        forecasts[f"{t:.1f}"] = {"status": "loaded", **{k: meta[k] for k in (
            "trained_at", "target", "selected_model", "selection_rule", "alpha", "dispersion_method", "periods",
            "samples", "eligibility", "validation_metrics", "test_metrics", "test_by_year", "test_reliability",
            "deviance_skill_vs", "features", "limitations", "glm_coefficients", "data")},
            "regions": len(meta["regions"]),
            "test_by_region": [{"id": r["id"], "label": r["label"], "mae": r["test"]["mae"],
                                "poisson_deviance": r["test"]["poisson_deviance"], "coverage_80": r["test"]["coverage_80"],
                                "baseline_deviance": r["baseline_historical_mean"]["poisson_deviance"],
                                "mean_observed": r["test"]["mean_observed"]} for r in meta["test_by_region"]]}
    after = evaluations["aftershock"]
    if after:
        after = {**after, "sequences": sorted(after["sequences"], key=lambda r: -r["mag"])[:25]}
    return jsonify({"forecast": forecasts, "sequences": evaluations["sequence"], "anomaly": evaluations["anomaly"],
                    "aftershock": after, "legacy_severity_model": bundle.describe(),
                    "note": "Every number here is read from files produced by `python ml_pipeline.py`; none is typed in."})


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
