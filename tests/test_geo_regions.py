"""Shared geometry, the region grid, weekly count matrices and the catalogue table rules."""
import math

import numpy as np
import pytest

from analytics import geo, regions
from analytics.catalog import events_to_frame

WEEK = regions.WEEK_MS


# ------------------------------------------------------------------ geometry
def test_haversine_known_distances():
    assert geo.haversine_km(0, 0, 0, 1) == pytest.approx(111.195, rel=1e-4)
    assert geo.haversine_km(0, 0, 90, 0) == pytest.approx(math.pi / 2 * geo.EARTH_RADIUS_KM, rel=1e-9)
    assert geo.haversine_km(10, 20, 10, 20) == 0


def test_haversine_across_antimeridian_is_short():
    d = geo.haversine_km(-17.0, 179.9, -17.0, -179.9)
    assert d == pytest.approx(0.2 * 111.195 * math.cos(math.radians(17)), rel=1e-3)


def test_haversine_near_pole_ignores_longitude_spread():
    assert geo.haversine_km(89.99, 0, 89.99, 180) == pytest.approx(0.02 * 111.195, rel=1e-3)


def test_chord_matches_unit_vector_distance():
    a, b = geo.unit_vectors([10, 10.4], [20, 20.3])
    arc = geo.haversine_km(10, 20, 10.4, 20.3)
    assert np.linalg.norm(a - b) == pytest.approx(float(geo.chord_for_arc_km(arc)), rel=1e-9)


def test_centroid_and_span_across_antimeridian():
    lat, lon = geo.spherical_centroid([0, 0], [179, -179])
    assert lat == pytest.approx(0, abs=1e-9) and abs(lon) == pytest.approx(180, abs=1e-9)
    west, east, width = geo.lon_span([178, 179.5, -179, -178.5])
    assert (west, east) == (178, -178.5) and width == pytest.approx(3.5)


def test_bearing_cardinal_directions():
    assert geo.bearing_deg(0, 0, 1, 0) == pytest.approx(0)
    assert geo.bearing_deg(0, 0, 0, 1) == pytest.approx(90)


# ------------------------------------------------------------------ region grid
def test_region_index_edges_and_wrapping():
    assert regions.region_index([90.0], [0.0])[0] == regions.region_index([89.9], [0.0])[0]     # top row includes 90
    assert regions.region_index([0.0], [180.0])[0] == regions.region_index([0.0], [-180.0])[0]  # 180 wraps to -180
    assert regions.region_index([-90.0], [-180.0])[0] == 0


def test_region_id_round_trip_and_rejection():
    for idx in (0, 100, 323, regions.N_REGIONS - 1):
        assert regions.parse_region_id(regions.region_id(idx)) == idx
    for bad in ("R+35+140", "R+30+145", "X+30+140", "R+90+000", "R+30+180", "", None, "R+3a+140", "R30+1400"):
        assert regions.parse_region_id(bad) is None


def test_neighbors_wrap_in_longitude_not_latitude():
    first_col = regions.region_index([5.0], [-175.0])[0]
    assert regions.region_index([5.0], [175.0])[0] in regions.neighbors(first_col)
    assert len(regions.neighbors(regions.region_index([85.0], [0.0])[0])) == 5
    assert len(regions.neighbors(regions.region_index([5.0], [0.0])[0])) == 8


def test_count_matrix_boundaries_and_threshold():
    epoch = 1_000 * WEEK
    times = [epoch, epoch + WEEK - 1, epoch + WEEK, epoch - 1, epoch + 3 * WEEK, epoch + 10]
    mags = [4.5, 4.49, 5.0, 6.0, 6.0, float("nan")]
    C = regions.count_matrix(times, mags, [1.0] * 6, [1.0] * 6, 4.5, epoch, 3)
    r = regions.region_index([1.0], [1.0])[0]
    assert C[r].tolist() == [1, 1, 0]       # 4.49 and NaN excluded, boundary event in the later week, outside dropped
    assert C.sum() == 2


# ------------------------------------------------------------------ catalogue table
def test_events_to_frame_drops_invalid_and_keeps_latest_duplicate():
    rows = [{"id": "a", "time": 5, "latitude": 1, "longitude": 1, "mag": 4.0, "updated": 1},
            {"id": "a", "time": 5, "latitude": 1, "longitude": 1, "mag": 4.2, "updated": 2},
            {"id": "b", "time": 3, "latitude": 95, "longitude": 1, "mag": 5.0},
            {"id": "c", "time": None, "latitude": 1, "longitude": 1, "mag": 5.0},
            {"id": "d", "time": 1, "latitude": 2, "longitude": -181, "mag": 5.0},
            {"id": "e", "time": 2, "latitude": 2, "longitude": 2, "mag": None}]
    df, stats = events_to_frame(rows)
    assert df["id"].tolist() == ["e", "a"]               # sorted by time
    assert df.loc[df["id"] == "a", "mag"].item() == 4.2   # most recently updated copy wins
    assert np.isnan(df.loc[df["id"] == "e", "mag"].item())  # missing magnitude kept as NaN, never 0
    assert stats == {"dropped_invalid": 3, "duplicates_removed": 1}


def test_events_to_frame_empty():
    df, stats = events_to_frame([])
    assert len(df) == 0 and stats == {"dropped_invalid": 0, "duplicates_removed": 0}
