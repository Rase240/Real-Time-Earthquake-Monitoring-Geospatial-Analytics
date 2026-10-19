"""Module A: space-time sequence detection on synthetic events with known membership."""
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from analytics import evaluation as EV
from analytics import sequences as SQ
from analytics.catalog import events_to_frame

HOUR = 3_600_000
T0 = 1_700_000_000_000


def ev(i, lat, lon, hours, mag=4.6, depth=10.0):
    return {"id": f"e{i}", "time": T0 + int(hours * HOUR), "latitude": lat, "longitude": lon, "mag": mag,
            "depth_km": depth}


def ari(name, seed=0, **params):
    events, truth = EV.scenario(name, seed)
    out = SQ.detect(events, **{**SQ.DEFAULTS, **params})
    df, _ = events_to_frame(events)
    ids = sorted({s["id"] for s in out["sequences"]})
    pred = [ids.index(out["assignments"][i]) if i in out["assignments"] else -1 for i in df["id"]]
    return adjusted_rand_score([truth[i] for i in df["id"]], pred), out


@pytest.mark.parametrize("name", ["dense_clusters_with_noise", "simultaneous_separated", "same_place_separated_in_time",
                                  "antimeridian", "high_latitude", "duplicates_and_invalid"])
def test_known_scenarios_are_recovered(name):
    score, _ = ari(name)
    assert score > 0.95


def test_naive_degree_distance_fails_where_the_proposed_metric_does_not():
    events, truth = EV.scenario("high_latitude", 0)
    df, _ = events_to_frame(events)
    naive = EV._naive_degrees(df, 50.0, 72.0, 4)
    assert adjusted_rand_score([truth[i] for i in df["id"]], naive) < 0.9 < ari("high_latitude")[0]


def test_antimeridian_cluster_is_one_sequence_with_correct_extent():
    events = [ev(i, -17 + 0.01 * i, 179.95 if i % 2 else -179.95, i) for i in range(10)]
    out = SQ.detect(events)
    assert out["summary"]["sequence_count"] == 1
    ext = out["sequences"][0]["extent"]
    assert ext["crosses_antimeridian"] and ext["lon_width_deg"] == pytest.approx(0.1, abs=1e-6)
    assert abs(out["sequences"][0]["centroid"]["longitude"]) > 179.9


def test_time_separation_splits_same_location():
    events = [ev(i, 35, 140, i) for i in range(5)] + [ev(10 + i, 35, 140, 500 + i) for i in range(5)]  # 496 h gap
    assert SQ.detect(events, eps_hours=72)["summary"]["sequence_count"] == 2
    assert SQ.detect(events, eps_hours=720)["summary"]["sequence_count"] == 1


def test_noise_events_are_unassigned_and_counted():
    events = [ev(i, 35, 140, i) for i in range(5)] + [ev(99, -40, 20, 3)]
    out = SQ.detect(events)
    assert out["summary"]["noise_events"] == 1 and "e99" not in out["assignments"]


def test_summary_statistics_and_stable_ids():
    events = [ev(i, 35 + 0.05 * i, 140, i * 2, mag=4.5 + 0.1 * i, depth=10 + i) for i in range(6)]
    out = SQ.detect(events)
    s = out["sequences"][0]
    assert s["event_count"] == 6 and s["max_mag"] == pytest.approx(5.0) and s["largest_event_id"] == "e5"
    assert s["duration_hours"] == pytest.approx(10.0)
    assert s["magnitude"]["median"] == pytest.approx(4.75) and s["depth_km"]["max"] == 15.0
    assert sum(b["count"] for b in s["magnitude"]["histogram"]) == 6
    again = SQ.detect(list(reversed(events)))
    assert again["sequences"][0]["id"] == s["id"]            # id does not depend on input order


def test_missing_magnitude_does_not_break_summary():
    events = [ev(i, 35, 140, i, mag=None) for i in range(5)]
    s = SQ.detect(events)["sequences"][0]
    assert s["max_mag"] is None and s["events_without_magnitude"] == 5


def test_empty_and_oversized_inputs():
    out = SQ.detect([])
    assert out["sequences"] == [] and out["summary"]["noise_events"] == 0
    with pytest.raises(ValueError):
        SQ.detect([ev(i, 0, 0, i) for i in range(SQ.MAX_EVENTS + 1)])


def test_ambiguous_border_event_is_reported():
    # Two tight groups along the equator; the bridge event is 44 km from one member of each group,
    # so it is a border event (3 neighbours < min_samples) reachable from two sequences.
    a = [ev(i, 0, lon, 0.1 * i) for i, lon in enumerate([-0.70, -0.60, -0.50, -0.45, -0.40])]
    b = [ev(10 + i, 0, lon, 0.1 * i) for i, lon in enumerate([0.40, 0.45, 0.50, 0.60, 0.70])]
    bridge = [ev(50, 0, 0.0, 0.2)]
    out = SQ.detect(a + b + bridge, eps_km=50, eps_hours=72, min_samples=5)
    assert out["summary"]["sequence_count"] == 2
    assert out["summary"]["ambiguous_border_events"] == 1
    assert np.isfinite(out["sequences"][0]["linking"]["median_nearest_member_km"])


def test_overlapping_sequences_merge_at_the_defaults():
    """Known, documented failure: two overlapping synthetic sequences come out as one at the default parameters."""
    score, out = ari("overlapping_sequences", seed=0)
    assert len(out["sequences"]) == 1 and score == pytest.approx(0.0, abs=1e-9)
    assert any("merged into one" in x for x in out["limitations"])
    assert "substructure" not in out["sequences"][0]                  # the rejected candidate check is not served


def test_substructure_candidate_flags_a_synthetic_merge_but_not_a_small_sequence():
    events, _ = EV.scenario("overlapping_sequences", 0)
    df, _ = events_to_frame(events)
    X = SQ.scaled_coordinates(df["latitude"], df["longitude"], df["time"], 50, 72)
    assert len(EV.substructure_groups(X)) >= 2
    assert EV.substructure_groups(X[: EV.SUBSTRUCTURE_MIN_EVENTS - 1]) == []


def test_merge_and_split_counts():
    truth = [0, 0, 0, 0, 1, 1, 1, 1, -1]
    assert EV.merges_splits(truth, [5, 5, 5, 5, 5, 5, 5, 5, -1])[:2] == (1, 0)      # one false merge
    assert EV.merges_splits(truth, [1, 1, 2, 2, 3, 3, 3, 3, -1])[:2] == (0, 1)      # sequence 0 split in two
    assert EV.merges_splits(truth, [1, 1, 1, -1, 2, 2, 2, 2, 2])[2] == pytest.approx(1 / 8)


@pytest.mark.parametrize("name", ["overlap_30km_30h", "overlap_150km_30h"])
def test_overlap_scenario_geometry(name):
    events, truth = EV.scenario(name, 0)
    assert set(truth.values()) == {0, 1} and len(events) == 50
    assert sum(t == 0 for t in truth.values()) == 25
    with pytest.raises(ValueError):
        EV.scenario("not_a_scenario", 0)
