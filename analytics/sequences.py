"""
Module A: space-time sequence detection with DBSCAN.

Distance
--------
Two events i, j are neighbours when

    (chord_ij / chord(eps_km))^2 + (|t_i - t_j| / eps_hours)^2 <= 1,

where chord_ij is the straight-line distance between their Earth-centred unit
vectors. The chord is a monotonic function of the great-circle distance, so the
spatial part is exactly "great-circle distance <= eps_km" when the time
difference is zero, and the neighbourhood is an ellipse in (distance, time)
otherwise. Using 3-D unit vectors means the antimeridian and the poles need no
special handling, and DBSCAN can run on plain Euclidean coordinates
(x, y, z scaled by 1/chord(eps_km), t scaled by 1/eps_hours) with eps = 1.

DBSCAN then forms sequences: a core event has at least `min_samples` events
(itself included) in its neighbourhood; sequences are connected core events
plus the border events within reach of a core event; everything else is noise.

Why DBSCAN: the number of sequences is unknown, most catalogue events are not
part of any dense group, and groups have irregular shapes along faults. DBSCAN
needs no cluster count, labels isolated events as noise, and is deterministic.
Its main weakness, a single density threshold for regions with very different
background rates, is reported via `limitations`. HDBSCAN would adapt to density
but is less predictable for users tuning parameters and was not needed.

Depth is not part of the distance: with the global catalogue's depth
uncertainties, adding it would mostly split shallow sequences arbitrarily.
Magnitude is not part of the distance either, so clusters are defined only by
where and when events happened; magnitudes are summarised afterwards.

A detected sequence is a statistical grouping. It does not establish that the
events are physically related (aftershocks, swarms or coincidence).
"""
import hashlib

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from analytics import geo
from analytics.catalog import events_to_frame

HOUR_MS = 3_600_000
DEFAULTS = {"eps_km": 50.0, "eps_hours": 72.0, "min_samples": 4}
LIMITS = {"eps_km": (5.0, 500.0), "eps_hours": (1.0, 720.0), "min_samples": (2, 50)}
MAX_EVENTS = 5000


def scaled_coordinates(lats, lons, times_ms, eps_km, eps_hours):
    """4-D coordinates in which the neighbourhood above is the Euclidean unit ball."""
    xyz = geo.unit_vectors(lats, lons) / geo.chord_for_arc_km(eps_km)
    t = (np.asarray(times_ms, np.int64) - int(np.min(times_ms))) / HOUR_MS / eps_hours
    return np.column_stack([xyz, t])


def cluster_labels(lats, lons, times_ms, eps_km, eps_hours, min_samples):
    """DBSCAN labels (-1 = noise) and the boolean core-event mask."""
    if len(lats) == 0:
        return np.empty(0, int), np.empty(0, bool)
    X = scaled_coordinates(lats, lons, times_ms, eps_km, eps_hours)
    db = DBSCAN(eps=1.0, min_samples=int(min_samples), metric="euclidean").fit(X)
    core = np.zeros(len(X), bool)
    core[db.core_sample_indices_] = True
    return db.labels_, core


def _q(values, q):
    return round(float(np.quantile(values, q)), 2) if len(values) else None


def _distribution(values):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "min": round(float(v.min()), 2), "q25": _q(v, 0.25), "median": _q(v, 0.5),
            "q75": _q(v, 0.75), "max": round(float(v.max()), 2)}


def _histogram(values, edges):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    counts, _ = np.histogram(v, bins=edges)
    return [{"from": float(a), "to": float(b), "count": int(c)} for a, b, c in zip(edges[:-1], edges[1:], counts, strict=True)]


MAG_EDGES = [-2, 2, 3, 4, 5, 6, 7, 10]
DEPTH_EDGES = [-10, 10, 35, 70, 150, 300, 800]


def detect(events, eps_km=DEFAULTS["eps_km"], eps_hours=DEFAULTS["eps_hours"], min_samples=DEFAULTS["min_samples"]):
    """
    Detect sequences among normalised events. Returns a JSON-serialisable dict with
    one summary per sequence, per-event assignments and data-quality counts.
    """
    df, stats = events_to_frame(events)
    if len(df) > MAX_EVENTS:
        raise ValueError(f"at most {MAX_EVENTS} events can be clustered per request")
    labels, core = cluster_labels(df["latitude"].to_numpy(), df["longitude"].to_numpy(), df["time"].to_numpy(),
                                  eps_km, eps_hours, min_samples)
    df = df.assign(label=labels, core=core)

    # Border events within reach of core events from more than one sequence (DBSCAN assigns them to one).
    ambiguous = np.zeros(len(df), bool)
    if len(df) and (labels >= 0).any():
        X = scaled_coordinates(df["latitude"], df["longitude"], df["time"], eps_km, eps_hours)
        nn = NearestNeighbors(radius=1.0).fit(X[core])
        core_labels = labels[core]
        border = np.flatnonzero((labels >= 0) & ~core)
        if len(border) and core.any():
            for i, neigh in zip(border, nn.radius_neighbors(X[border], return_distance=False), strict=True):
                ambiguous[i] = len(set(core_labels[neigh].tolist())) > 1

    sequences, assignment = [], {}
    for lab in sorted(set(labels.tolist()) - {-1}):
        m = df[df["label"] == lab].sort_values(["time", "id"])
        first = m.iloc[0]
        seq_id = "S-" + hashlib.sha1(str(first["id"]).encode()).hexdigest()[:10]
        lat, lon = m["latitude"].to_numpy(), m["longitude"].to_numpy()
        centroid = geo.spherical_centroid(lat, lon)
        dist = geo.haversine_km(centroid[0], centroid[1], lat, lon)
        west, east, width = geo.lon_span(lon)
        mags = m["mag"].to_numpy(float)
        has_mag = np.isfinite(mags)
        largest = m.iloc[int(np.nanargmax(np.where(has_mag, mags, -np.inf)))] if has_mag.any() else None
        # Linking statistics: for each member, its nearest other member in space and in time.
        if len(m) > 1:
            pts = np.radians(np.column_stack([lat, lon]))
            nearest, _ = NearestNeighbors(n_neighbors=2, metric="haversine").fit(pts).kneighbors(pts)
            nearest_km = nearest[:, 1] * geo.EARTH_RADIUS_KM
            gaps = np.diff(m["time"].to_numpy()) / HOUR_MS
        else:
            nearest_km, gaps = np.zeros(1), np.array([])
        start, end = int(m["time"].iloc[0]), int(m["time"].iloc[-1])
        seq = {
            "id": seq_id, "event_count": int(len(m)), "core_events": int(m["core"].sum()),
            "ambiguous_border_events": int(ambiguous[m.index].sum()),
            "start": start, "end": end, "duration_hours": round((end - start) / HOUR_MS, 2),
            "max_mag": None if largest is None else float(largest["mag"]),
            "largest_event_id": None if largest is None else str(largest["id"]),
            "events_without_magnitude": int((~has_mag).sum()),
            "magnitude": {**_distribution(mags), "histogram": _histogram(mags, MAG_EDGES)},
            "depth_km": {**_distribution(m["depth_km"].to_numpy(float)),
                         "histogram": _histogram(m["depth_km"].to_numpy(float), DEPTH_EDGES)},
            "centroid": {"latitude": round(centroid[0], 4), "longitude": round(centroid[1], 4)},
            "extent": {"lat_min": round(float(lat.min()), 4), "lat_max": round(float(lat.max()), 4),
                       "lon_west": round(west, 4), "lon_east": round(east, 4),
                       "lon_width_deg": round(width, 4), "crosses_antimeridian": bool(west > east),
                       "max_distance_from_centroid_km": round(float(dist.max()), 1)},
            "linking": {"median_nearest_member_km": round(float(np.median(nearest_km)), 1) if len(m) > 1 else None,
                        "median_gap_hours": round(float(np.median(gaps)), 2) if len(gaps) else None,
                        "max_gap_hours": round(float(gaps.max()), 2) if len(gaps) else None},
            "member_ids": m["id"].tolist(),
        }
        sequences.append(seq)
        for eid in seq["member_ids"]:
            assignment[eid] = seq_id
    sequences.sort(key=lambda s: (-s["event_count"], s["start"]))

    sizes = [s["event_count"] for s in sequences]
    return {
        "sequences": sequences,
        "assignments": assignment,
        "summary": {"events_considered": int(len(df)), "sequence_count": len(sequences),
                    "clustered_events": int(sum(sizes)), "noise_events": int(len(df) - sum(sizes)),
                    "largest_sequence": max(sizes) if sizes else 0,
                    "median_sequence_size": float(np.median(sizes)) if sizes else None,
                    "ambiguous_border_events": int(ambiguous.sum()), **stats},
        "parameters": {"eps_km": eps_km, "eps_hours": eps_hours, "min_samples": int(min_samples),
                       "algorithm": "DBSCAN", "metric": "ellipsoidal space-time distance on Earth-centred chords"},
        "explanation": (f"Two events are neighbours when their great-circle separation and time difference fall "
                        f"inside an ellipse of {eps_km:g} km by {eps_hours:g} h. An event with at least "
                        f"{int(min_samples)} neighbours (itself included) is a core event; core events that are "
                        "neighbours are joined into one sequence, and non-core events within reach of a core event "
                        "join it as border events. Other events are left unassigned (noise)."),
        "limitations": [
            "A detected sequence is a statistical space-time grouping; it does not show that its events are "
            "physically or causally related.",
            "One density threshold is used everywhere, so densely monitored regions (with more small events) "
            "form sequences more easily than sparsely monitored ones.",
            "Border events that are within reach of two sequences are assigned to only one "
            "(counted as ambiguous_border_events).",
            "Sequences can be cut at the start or end of the data window and by the magnitude threshold.",
            "Depth and magnitude are not used in the distance.",
        ],
    }
