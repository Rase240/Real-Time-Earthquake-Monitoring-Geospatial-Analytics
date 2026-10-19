"""Module D: deterministic event selection, binned statistics and Omori-Utsu fitting."""
import math

import numpy as np
import pytest

from analytics import aftershock as AS
from analytics import geo

DAY = AS.DAY_MS
T0 = 1_700_000_000_000
REF = {"id": "main", "time": T0, "latitude": 10.0, "longitude": 125.0, "mag": 7.0, "depth_km": 20.0}


def ev(i, days, dist_km=10.0, mag=4.6, depth=15.0, bearing=45.0):
    p = math.radians(bearing)
    lat = REF["latitude"] + dist_km / 111.195 * math.cos(p)
    lon = REF["longitude"] + dist_km / (111.195 * math.cos(math.radians(REF["latitude"]))) * math.sin(p)
    return {"id": f"e{i}", "time": T0 + int(days * DAY), "latitude": lat, "longitude": lon, "mag": mag, "depth_km": depth}


def omori_times(K, c, p, t0, t1, seed=0):
    """Simulate an inhomogeneous Poisson process with rate K/(t+c)^p on [t0, t1] by inverting its integral."""
    rng = np.random.default_rng(seed)
    total = AS._integral(K, c, p, t0, t1)
    u = np.sort(rng.uniform(0, total, rng.poisson(total)))
    # Lambda(t) = K((t+c)^(1-p) - (t0+c)^(1-p))/(1-p)  ->  t = (u(1-p)/K + (t0+c)^(1-p))^(1/(1-p)) - c
    return (u * (1 - p) / K + (t0 + c) ** (1 - p)) ** (1 / (1 - p)) - c


def test_default_radius_scaling():
    assert AS.default_radius_km(7.0) == pytest.approx(round(10 ** (-2.44 + 0.59 * 7)))
    assert AS.default_radius_km(4.0) == 30 and AS.default_radius_km(9.5) == 1000
    assert AS.default_radius_km(None) == 100.0


def test_selection_rules():
    events = [REF, ev(1, 1), ev(2, 1, dist_km=200), ev(3, 1, mag=4.0), ev(4, -5), ev(5, -40), ev(6, 1, depth=None),
              ev(7, 1, depth=300), ev(8, 70), {**ev(9, 1), "mag": None}]
    pre, post, days, quality = AS.select(events, REF, radius_km=100, pre_days=30, post_days=60, min_mag=4.5)
    assert sorted(post["id"]) == ["e1", "e6", "e7"]             # reference, far, small, late, no-mag excluded
    assert pre["id"].tolist() == ["e4"] and days == 60
    _, post_d, _, _ = AS.select(events, REF, 100, 30, 60, 4.5, max_depth=100)
    assert post_d["id"].tolist() == ["e1"]                       # no-depth event excluded once a limit is set
    _, _, days_cut, _ = AS.select(events, REF, 100, 30, 60, 4.5, data_end_ms=T0 + 10 * DAY)
    assert days_cut == 10


def test_distance_uses_great_circle():
    near_dateline = {**REF, "longitude": 179.95}
    e = {"id": "x", "time": T0 + DAY, "latitude": 10.0, "longitude": -179.95, "mag": 5.0, "depth_km": 10}
    _, post, _, _ = AS.select([e], near_dateline, 50, 0, 10, 4.5)
    assert len(post) == 1 and post["distance_km"].iloc[0] == pytest.approx(
        geo.haversine_km(10, 179.95, 10, -179.95), rel=1e-9)


def test_describe_counts_and_pre_comparison():
    events = [ev(i, 0.5 + i * 0.1) for i in range(30)] + [ev(100 + i, -20 + i) for i in range(3)]
    pre, post, days, _ = AS.select(events, REF, 100, 30, 10, 4.5)
    s = AS.describe(pre, post, days, 30, 100, 4.5)
    assert s["counts"] == {"pre": 3, "post": 30}
    assert sum(b["count"] for b in s["daily"]) == 30 and s["cumulative"][-1]["n"] == 30
    assert s["pre_comparison"]["rate_ratio"] == pytest.approx((30 / 10) / (3 / 30))
    assert s["pre_comparison"]["p_value_post_rate_higher"] < 0.001


def test_fit_recovers_simulated_parameters():
    t = omori_times(K=40.0, c=0.05, p=1.1, t0=0.0, t1=60.0, seed=3)
    fit = AS.fit_omori(t, 0.0, 60.0, mu=0.0)
    assert fit["status"] == "fitted", fit["reasons"]
    assert fit["parameters"]["p"] == pytest.approx(1.1, abs=0.15)
    lo, hi = fit["confidence_95"]["p"]
    assert lo < fit["parameters"]["p"] < hi
    assert fit["diagnostics"]["delta_aic_vs_constant"] > 2 and fit["diagnostics"]["ks_p_value"] >= 0.05


def test_fit_starting_after_c_still_estimates_p():
    """When the fit starts long after c, c is unresolvable; p must still be estimated and c handled explicitly."""
    t = omori_times(K=30.0, c=0.001, p=1.05, t0=0.0, t1=60.0, seed=5)
    fit = AS.fit_omori(t, 0.25, 60.0, mu=0.0)
    assert fit["parameters"]["p"] == pytest.approx(1.05, abs=0.2)
    if fit["c_fixed"]:
        assert fit["notes"] and fit["parameters"]["c_days"] == AS.C_BOUNDS[0]
        assert fit["confidence_95"] is None or fit["confidence_95"]["c_days"] is None


def test_too_few_events_is_insufficient_data():
    fit = AS.fit_omori(np.linspace(0.5, 5, 10), 0.1, 10, 0.0)
    assert fit["status"] == "insufficient_data" and not fit["reliable"] and "parameters" not in fit


def test_constant_rate_data_is_not_called_a_decay():
    t = np.sort(np.random.default_rng(0).uniform(0.25, 60, 120))
    fit = AS.fit_omori(t, 0.25, 60.0, 0.0)
    assert not fit["reliable"]
    assert any("constant-rate" in r or "bound" in r or "KS" in r for r in fit["reasons"])


def test_curve_stays_inside_fit_window():
    t = omori_times(K=40.0, c=0.05, p=1.1, t0=0.0, t1=30.0, seed=3)
    fit = AS.fit_omori(t, 0.25, 30.0, 0.0)
    pts = AS.curve(fit, [0.0, 0.1, 1.0, 10.0, 30.0, 100.0])
    assert pts[0]["from"] == 0.25 and pts[-1]["to"] == 30.0 and all(p["rate_per_day"] > 0 for p in pts)


def test_analyse_end_to_end_and_truncation_flag():
    t = omori_times(K=40.0, c=0.05, p=1.1, t0=0.01, t1=30.0, seed=7)
    events = [REF] + [ev(i, x) for i, x in enumerate(t)]
    out = AS.analyse(events, REF, 100, 0, 30, 4.5, fit_start_hours=6)
    assert out["statistics"]["counts"]["post"] == len(t) and len(out["events"]) == len(t)
    assert all(e["id"] != "main" for e in out["events"])
    trunc = AS.analyse(events, REF, 100, 0, 30, 4.5, truncated=True)
    assert trunc["fit"]["status"] == "incomplete_data"
