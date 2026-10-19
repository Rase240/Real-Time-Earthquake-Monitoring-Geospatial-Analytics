"""
Great-circle geometry on a spherical Earth.

All distances use a sphere of radius EARTH_RADIUS_KM (the IUGG mean radius, the
same radius implied by features.KM_PER_DEG). Errors against the WGS84 ellipsoid
are below 0.5%, which is far smaller than typical epicentre uncertainty.

Longitudes may be given in any range; every function is periodic in longitude,
so points on either side of the antimeridian are handled without special cases.
"""
import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised, broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def unit_vectors(lats, lons):
    """(N, 3) Earth-centred unit vectors. Euclidean distance between them is the chord."""
    phi = np.radians(np.asarray(lats, float))
    lmb = np.radians(np.asarray(lons, float))
    return np.column_stack([np.cos(phi) * np.cos(lmb), np.cos(phi) * np.sin(lmb), np.sin(phi)])


def chord_for_arc_km(arc_km):
    """Chord length (in Earth radii) subtending a great-circle arc of arc_km."""
    return 2.0 * np.sin(np.asarray(arc_km, float) / (2.0 * EARTH_RADIUS_KM))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north in [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def spherical_centroid(lats, lons):
    """Mean position of points on the sphere (correct across the antimeridian). Returns (lat, lon)."""
    v = unit_vectors(lats, lons).mean(axis=0)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    v = v / norm
    return float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))), float(np.degrees(np.arctan2(v[1], v[0])))


def lon_span(lons):
    """
    Smallest longitude interval that contains all points, as (west, east, width_deg).
    For a group straddling the antimeridian, west > east (e.g. 178 -> -178, width 4).
    """
    lons = np.sort(((np.asarray(lons, float) + 180.0) % 360.0) - 180.0)
    if len(lons) == 0:
        return None
    if len(lons) == 1:
        return float(lons[0]), float(lons[0]), 0.0
    gaps = np.diff(np.concatenate([lons, [lons[0] + 360.0]]))
    i = int(np.argmax(gaps))                      # the largest empty gap is outside the group
    west, east = lons[(i + 1) % len(lons)], lons[i]
    return float(west), float(east), float(360.0 - gaps[i])
