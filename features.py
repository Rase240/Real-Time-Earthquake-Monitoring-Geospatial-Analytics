"""
Feature engineering shared by training (train_model.py) and inference (ml_model.py).

Keeping every feature definition in one module is what guarantees that the
model sees identically computed inputs at training time and at serving time.

Features (in this exact order):
  latitude           degrees, WGS84, [-90, 90]
  longitude          degrees, WGS84, [-180, 180)
  depth_km           hypocentre depth in km (USGS convention, positive down)
  local_density_30d  number of catalogue events (M >= CATALOG_MIN_MAG) inside a
                     +/-1 degree lat/lon box during the 30 days strictly BEFORE
                     the reference time (the event itself is never counted, and
                     nothing after the reference time is used)
  plate_distance_km  approximate distance to the nearest mapped plate boundary
                     segment, using a local equirectangular projection
"""
import json
import logging
import os

import numpy as np

log = logging.getLogger("terrawatcher.features")

FEATURE_NAMES = ["latitude", "longitude", "depth_km", "local_density_30d", "plate_distance_km"]

# The model is trained on, and its density feature counts, events at or above
# this magnitude. M4.5 is roughly where the global USGS catalogue is complete;
# below that, coverage depends heavily on regional network density.
CATALOG_MIN_MAG = 4.5
# Target: a catalogue event is "significant" when its magnitude is >= this value.
SIGNIFICANT_MAG = 5.0

DENSITY_WINDOW_DAYS = 30
DENSITY_RADIUS_DEG = 1.0
KM_PER_DEG = 111.195
DAY_MS = 86_400_000


def wrap_lon(lon):
    """Wrap longitude difference/values into [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


# -----------------------------------------------------------------------------
# Plate boundaries
# -----------------------------------------------------------------------------
def load_plate_segments(path):
    """
    Read plate-boundary polylines from GeoJSON and return an (N, 4) float array of
    segments [lon1, lat1, lon2, lat2]. Returns an empty array if the file is
    missing or unreadable (callers must treat that as "feature unavailable").
    """
    if not os.path.exists(path):
        log.warning("Plate boundary file not found: %s", path)
        return np.empty((0, 4))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        log.error("Could not read plate boundaries: %s", exc)
        return np.empty((0, 4))

    segments = []
    for feature in data.get("features", []):
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []
        lines = coords if geom.get("type") == "MultiLineString" else [coords] if geom.get("type") == "LineString" else []
        for line in lines:
            for (lon1, lat1, *_), (lon2, lat2, *_) in zip(line[:-1], line[1:], strict=True):
                segments.append((lon1, lat1, lon2, lat2))
    seg = np.asarray(segments, dtype=float).reshape(-1, 4)
    log.info("Loaded %d plate boundary segments from %s", len(seg), os.path.basename(path))
    return seg


def plate_distance_km(lats, lons, segments, chunk=512):
    """
    Distance (km) from each point to the nearest boundary segment.

    For each point we project segment endpoints into a local equirectangular
    frame centred on the point (x = dlon * cos(lat), y = dlat), with longitude
    differences wrapped so segments across the antimeridian are handled. The
    approximation is good for the short distances that matter here (tens to a
    few hundred km) and degrades for very long distances or near the poles.
    """
    lats = np.atleast_1d(np.asarray(lats, dtype=float))
    lons = np.atleast_1d(np.asarray(lons, dtype=float))
    if len(segments) == 0:
        raise ValueError("No plate boundary segments loaded")

    lon1, lat1, lon2, lat2 = (segments[:, i][None, :] for i in range(4))
    out = np.empty(len(lats))
    for start in range(0, len(lats), chunk):
        plat = lats[start:start + chunk, None]
        plon = lons[start:start + chunk, None]
        k = np.cos(np.radians(plat))
        ax, ay = wrap_lon(lon1 - plon) * k, lat1 - plat   # segment start relative to point
        bx, by = wrap_lon(lon2 - plon) * k, lat2 - plat   # segment end relative to point
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(l2 > 0, -(ax * dx + ay * dy) / l2, 0.0)
        t = np.clip(t, 0.0, 1.0)
        cx, cy = ax + t * dx, ay + t * dy                  # closest point on segment
        out[start:start + chunk] = np.sqrt(cx * cx + cy * cy).min(axis=1) * KM_PER_DEG
    return out


# -----------------------------------------------------------------------------
# Causal local density
# -----------------------------------------------------------------------------
def causal_density(cat_times, cat_lats, cat_lons, q_times, q_lats, q_lons,
                   window_days=DENSITY_WINDOW_DAYS, radius_deg=DENSITY_RADIUS_DEG):
    """
    For each query (time, lat, lon) count catalogue events with
        q_time - window <= event_time < q_time
    inside the +/- radius_deg lat/lon box (longitude wrapped).

    Only strictly earlier events are counted, so the feature never includes the
    event being described or anything that happened after it (no look-ahead).
    """
    cat_times = np.asarray(cat_times, dtype=np.int64)
    order = np.argsort(cat_times, kind="stable")
    ct, clat, clon = cat_times[order], np.asarray(cat_lats, float)[order], np.asarray(cat_lons, float)[order]

    q_times = np.atleast_1d(np.asarray(q_times, dtype=np.int64))
    q_lats = np.atleast_1d(np.asarray(q_lats, dtype=float))
    q_lons = np.atleast_1d(np.asarray(q_lons, dtype=float))

    lo = np.searchsorted(ct, q_times - window_days * DAY_MS, side="left")
    hi = np.searchsorted(ct, q_times, side="left")   # strictly before q_time
    out = np.zeros(len(q_times), dtype=np.int64)
    for i in range(len(q_times)):
        if hi[i] <= lo[i]:
            continue
        dlat = np.abs(clat[lo[i]:hi[i]] - q_lats[i])
        dlon = np.abs(wrap_lon(clon[lo[i]:hi[i]] - q_lons[i]))
        out[i] = int(np.count_nonzero((dlat <= radius_deg) & (dlon <= radius_deg)))
    return out


def build_feature_matrix(lats, lons, depths, densities, plate_dists):
    """Stack features in FEATURE_NAMES order. All inputs must already be validated."""
    X = np.column_stack([
        np.asarray(lats, float),
        np.asarray(lons, float),
        np.asarray(depths, float),
        np.asarray(densities, float),
        np.asarray(plate_dists, float),
    ])
    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains NaN or infinite values")
    return X
