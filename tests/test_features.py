"""Feature engineering: geometry, causal density and feature-matrix contract."""
import math

import numpy as np
import pytest

import config
import features as F

DAY = F.DAY_MS


def test_feature_order_is_fixed():
    assert F.FEATURE_NAMES == ["latitude", "longitude", "depth_km", "local_density_30d", "plate_distance_km"]


def test_wrap_lon():
    assert np.allclose(F.wrap_lon([180.0, -180.0, 190.0, -190.0, 0.0]), [-180.0, -180.0, -170.0, 170.0, 0.0])


def test_plate_distance_simple_segment_at_equator():
    seg = np.array([[0.0, 0.0, 2.0, 0.0]])
    d = F.plate_distance_km([1.0], [1.0], seg)[0]           # 1 degree north of the segment
    assert math.isclose(d, F.KM_PER_DEG, rel_tol=1e-6)


def test_plate_distance_across_antimeridian():
    seg = np.array([[-179.9, 51.9, -179.9, 52.1]])
    d = F.plate_distance_km([52.0], [179.9], seg)[0]         # 0.2 deg of longitude away, not 359.8
    expected = 0.2 * math.cos(math.radians(52.0)) * F.KM_PER_DEG
    assert math.isclose(d, expected, rel_tol=1e-3)


def test_plate_distance_uses_nearest_of_many_segments():
    seg = np.array([[0, 10, 1, 10], [0, 0.5, 1, 0.5], [50, 50, 51, 51]], dtype=float)
    assert math.isclose(F.plate_distance_km([0.0], [0.5], seg)[0], 0.5 * F.KM_PER_DEG, rel_tol=1e-6)


def test_plate_distance_requires_segments():
    with pytest.raises(ValueError):
        F.plate_distance_km([0.0], [0.0], np.empty((0, 4)))


def test_bundled_plate_file_loads():
    seg = F.load_plate_segments(config.TECTONIC_FILE)
    assert seg.shape[1] == 4 and len(seg) > 5000


def test_missing_plate_file_returns_empty(tmp_path):
    assert len(F.load_plate_segments(str(tmp_path / "nope.geojson"))) == 0


def test_causal_density_counts_only_the_past_window():
    t0 = 100 * DAY
    times = [t0 - 31 * DAY, t0 - 10 * DAY, t0 - 1, t0, t0 + DAY]   # too old, in, in, self, future
    lats = [0.0] * 5
    lons = [0.0] * 5
    assert F.causal_density(times, lats, lons, [t0], [0.0], [0.0])[0] == 2


def test_causal_density_box_and_antimeridian():
    t0 = 100 * DAY
    times = [t0 - DAY] * 4
    lats = [0.5, 1.5, 0.0, 0.0]
    lons = [179.6, 179.6, -179.8, 170.0]
    # Query at lon 179.9: lon -179.8 is 0.3 deg away across the antimeridian.
    assert F.causal_density(times, lats, lons, [t0], [0.0], [179.9])[0] == 2


def test_causal_density_is_independent_of_catalog_order():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 60 * DAY, 300)
    la, lo = rng.uniform(-2, 2, 300), rng.uniform(-2, 2, 300)
    a = F.causal_density(t, la, lo, t, la, lo)
    perm = rng.permutation(300)
    b = F.causal_density(t[perm], la[perm], lo[perm], t, la, lo)
    assert np.array_equal(a, b)


def test_build_feature_matrix_rejects_nan():
    with pytest.raises(ValueError):
        F.build_feature_matrix([0.0], [0.0], [float("nan")], [1], [10.0])


def test_build_feature_matrix_order():
    X = F.build_feature_matrix([1.0], [2.0], [3.0], [4], [5.0])
    assert X.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0]]
