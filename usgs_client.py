"""
Client for the USGS FDSN event web service (https://earthquake.usgs.gov/fdsnws/event/1/).

Responsibilities:
  * build query URLs with full UTC timestamps,
  * apply timeouts and a small retry policy for 429 / 5xx / timeouts,
  * validate and normalise GeoJSON features into flat event dicts,
  * drop records with unusable coordinates instead of inventing values,
  * de-duplicate by event id.

Nothing here fabricates data: if USGS cannot be reached, UpstreamError is raised
and the caller decides how to report it.
"""
import logging
import math
import time
from datetime import datetime, timezone

import requests

import config

log = logging.getLogger("terrawatcher.usgs")


class UpstreamError(Exception):
    """USGS could not be reached or returned something unusable."""

    def __init__(self, message, status=502, code="upstream_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def _iso(dt):
    """FDSN accepts ISO-8601 timestamps; always send UTC with second precision."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _get(path, params, timeout=None, retries=None, session=None):
    url = f"{config.USGS_BASE_URL}/{path}"
    timeout = config.USGS_TIMEOUT if timeout is None else timeout
    retries = config.USGS_RETRIES if retries is None else retries
    http = session or requests
    last_error = None

    for attempt in range(retries + 1):
        try:
            resp = http.get(url, params=params, timeout=timeout)
        except requests.exceptions.Timeout:
            last_error = UpstreamError("USGS request timed out", status=504, code="upstream_timeout")
        except requests.exceptions.RequestException as exc:
            last_error = UpstreamError(f"Could not reach USGS ({type(exc).__name__})", code="upstream_unreachable")
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    raise UpstreamError("USGS returned a response that is not valid JSON",
                                        code="upstream_malformed") from None
            if resp.status_code == 204:
                # FDSN uses 204 for "no matching events" when nodata=204 is requested.
                return {"type": "FeatureCollection", "features": []}
            if resp.status_code == 400:
                # Our query was rejected; retrying will not help.
                raise UpstreamError("USGS rejected the query parameters", status=502, code="upstream_bad_request")
            if resp.status_code == 429 or resp.status_code >= 500:
                code = "upstream_rate_limited" if resp.status_code == 429 else "upstream_server_error"
                last_error = UpstreamError(f"USGS responded with HTTP {resp.status_code}", code=code)
                retry_after = resp.headers.get("Retry-After", "")
                delay = min(float(retry_after), 5.0) if retry_after.isdigit() else 1.0 * (attempt + 1)
                if attempt < retries:
                    log.warning("USGS HTTP %s, retrying in %.1fs", resp.status_code, delay)
                    time.sleep(delay)
                continue
            raise UpstreamError(f"USGS responded with HTTP {resp.status_code}")

        if attempt < retries:
            log.warning("%s; retrying", last_error)
            time.sleep(1.0 * (attempt + 1))

    raise last_error


def _finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def normalize_feature(feature):
    """
    Convert one USGS GeoJSON feature into TerraWatcher's flat event dict.

    Returns None when the record has no usable coordinates or timestamp.
    Optional fields (magnitude, depth, place, ...) are passed through as None
    when USGS omits them; they are never replaced with made-up defaults.
    """
    if not isinstance(feature, dict):
        return None
    props = feature.get("properties") or {}
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") if isinstance(geom, dict) else None
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None

    lon, lat = _finite(coords[0]), _finite(coords[1])
    if lon is None or lat is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    depth = _finite(coords[2]) if len(coords) > 2 else None

    t = _finite(props.get("time"))
    if t is None:
        return None
    time_ms = int(t)
    time_iso = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds")

    event_id = feature.get("id") or props.get("ids", "").strip(",").split(",")[0] or None
    if not event_id:
        event_id = f"{lat:.4f},{lon:.4f},{time_ms}"

    return {
        "id": str(event_id),
        "time": time_ms,
        "time_iso": time_iso.replace("+00:00", "Z"),
        "updated": int(_finite(props.get("updated"))) if _finite(props.get("updated")) is not None else None,
        "latitude": lat,
        "longitude": lon,
        "depth_km": depth,
        "mag": _finite(props.get("mag")),
        "mag_type": props.get("magType"),
        "place": props.get("place"),
        "event_type": props.get("type"),
        "status": props.get("status"),
        "tsunami": bool(props.get("tsunami")) if props.get("tsunami") is not None else None,
        "felt": props.get("felt"),
        "alert": props.get("alert"),
        "sig": props.get("sig"),
        "net": props.get("net"),
        "url": props.get("url"),
    }


def normalize_collection(payload):
    """Validate a FeatureCollection and normalise every feature. Returns (events, stats)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
        raise UpstreamError("USGS response did not contain a GeoJSON feature list", code="upstream_malformed")

    events, seen = [], set()
    dropped = duplicates = 0
    for feature in payload["features"]:
        event = normalize_feature(feature)
        if event is None:
            dropped += 1
            continue
        if event["id"] in seen:
            duplicates += 1
            continue
        seen.add(event["id"])
        events.append(event)

    if dropped:
        log.info("Dropped %d USGS records without usable coordinates/time", dropped)
    return events, {"dropped_invalid": dropped, "duplicates_removed": duplicates}


def fetch_events(start, end, min_mag=None, limit=None, timeout=None, retries=None, session=None):
    """Fetch and normalise events in [start, end). Newest first."""
    params = {
        "format": "geojson",
        "starttime": _iso(start),
        "endtime": _iso(end),
        "orderby": "time",
        "eventtype": "earthquake",
    }
    if min_mag is not None:
        params["minmagnitude"] = min_mag
    if limit is not None:
        params["limit"] = int(limit)
    payload = _get("query", params, timeout=timeout, retries=retries, session=session)
    return normalize_collection(payload)


def count_events(start, end, min_mag=None, timeout=None):
    """Number of matching events (used to report truncation). Returns None if unavailable."""
    params = {"format": "geojson", "starttime": _iso(start), "endtime": _iso(end), "eventtype": "earthquake"}
    if min_mag is not None:
        params["minmagnitude"] = min_mag
    try:
        payload = _get("count", params, timeout=timeout, retries=0)
        return int(payload.get("count"))
    except (UpstreamError, TypeError, ValueError, AttributeError):
        return None
