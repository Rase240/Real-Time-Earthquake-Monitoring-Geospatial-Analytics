"""
Module B: forecast-origin features, leakage rules, metrics, predictive intervals,
walk-forward training and artifact loading.

Synthetic count matrices here are unit-test fixtures for code paths; they are
never used to train the served models.
"""
import json
import shutil

import numpy as np
import pytest

import config
from analytics import forecast as FC
from analytics import forecast_training as T
from analytics import regional_features as RF
from analytics import regions

WEEK = regions.WEEK_MS
R0 = int(regions.region_index([35.0], [140.0])[0])


def matrix(weeks=60, seed=0, rate=2.0):
    rng = np.random.default_rng(seed)
    C = np.zeros((regions.N_REGIONS, weeks), dtype=np.int64)
    C[R0] = rng.poisson(rate, weeks)
    for n in regions.neighbors(R0)[:2]:
        C[n] = rng.poisson(1.0, weeks)
    L = np.zeros_like(C)
    L[R0, 54] = 2          # inside the 4-week window before origin 56
    L[R0, 57] = 7          # after origin 56: must never be counted
    return C, L


# ------------------------------------------------------------------ features
def test_feature_values_against_hand_computation():
    C, L = matrix()
    k = 56
    x = dict(zip(RF.FEATURE_NAMES, RF.features_at(C, L, [k], [R0])[0], strict=True))
    w = C[R0]
    assert x["lag1"] == w[k - 1] and x["lag2"] == w[k - 2]
    assert x["mean_4w"] == pytest.approx(w[k - 4:k].mean())
    assert x["mean_13w"] == pytest.approx(w[k - 13:k].mean())
    assert x["mean_52w"] == pytest.approx(w[k - 52:k].mean())
    assert x["std_13w"] == pytest.approx(w[k - 13:k].std())
    nb = sum(C[n, k - 4:k].sum() for n in regions.neighbors(R0)) / 4
    assert x["neighbor_mean_4w"] == pytest.approx(nb)
    assert x["large_4w"] == 2


def test_features_never_read_the_target_week_or_later():
    C, L = matrix()
    k = 55
    before = RF.features_at(C, L, [k], [R0])
    C2, L2 = C.copy(), L.copy()
    C2[:, k:] += 50          # change the target week and everything after it
    L2[:, k:] += 5
    assert np.array_equal(before, RF.features_at(C2, L2, [k], [R0]))
    assert RF.targets_at(C2, [k], [R0])[0] == C[R0, k] + 50


def test_weeks_since_last_and_cap():
    C = np.zeros((regions.N_REGIONS, 60), dtype=np.int64)
    L = np.zeros_like(C)
    C[R0, 50] = 1
    x = RF.features_at(C, L, [53, 52], [R0])
    assert x[:, RF.FEATURE_NAMES.index("weeks_since_last")].tolist() == [2, 1]
    empty = RF.features_at(np.zeros_like(C), L, [55], [R0])
    assert empty[0, RF.FEATURE_NAMES.index("weeks_since_last")] == RF.LOOKBACK_WEEKS


def test_origin_requires_full_lookback():
    C, L = matrix()
    with pytest.raises(ValueError):
        RF.features_at(C, L, [RF.LOOKBACK_WEEKS - 1], [R0])


def test_training_and_serving_features_are_identical():
    """The live 52-week matrix built from raw events must give the same features as the training matrix."""
    rng = np.random.default_rng(4)
    epoch = 2_000 * WEEK
    n = 400
    times = epoch + rng.integers(0, 70 * WEEK, n)
    lats, lons = 35 + rng.uniform(-8, 8, n), 140 + rng.uniform(-8, 8, n)
    mags = np.round(rng.uniform(4.4, 6.5, n), 1)
    C = regions.count_matrix(times, mags, lats, lons, 4.5, epoch, 70)
    L = regions.count_matrix(times, mags, lats, lons, RF.LARGE_MAG, epoch, 70)
    k = 64
    reg = np.unique(regions.region_index(lats, lons))
    train_side = RF.features_at(C, L, [k], reg)
    Cl, Ll, _ = RF.live_matrices(times, mags, lats, lons, 4.5, epoch + k * WEEK)
    assert np.array_equal(train_side, RF.features_at(Cl, Ll, [RF.LOOKBACK_WEEKS], reg))


# ------------------------------------------------------------------ metrics and distributions
def test_poisson_deviance_defined_for_zero_counts():
    assert FC.poisson_deviance([0, 0], [1.0, 2.0]) == pytest.approx((2 * 1 + 2 * 2) / 2)
    assert FC.poisson_deviance([3], [3.0]) == pytest.approx(0)


def test_count_metrics_has_no_percentage_errors_and_handles_zeros():
    m = FC.count_metrics([0, 0, 2], [0.5, 0.1, 2.5], alpha=0.5)
    assert not any("pct" in k or "mape" in k for k in m)
    assert m["mae"] == pytest.approx((0.5 + 0.1 + 0.5) / 3, abs=1e-5)
    assert m["mae_zero_weeks"] == pytest.approx(0.3, abs=1e-5)
    assert 0 <= m["coverage_80"] <= 1 and np.isfinite(m["nb_log_score"])


def test_dispersion_mle_recovers_simulated_alpha():
    rng = np.random.default_rng(1)
    mu, alpha = np.full(20000, 3.0), 0.8
    r = 1 / alpha
    y = rng.negative_binomial(r, r / (r + mu))
    assert FC.fit_dispersion(y, mu) == pytest.approx(alpha, rel=0.1)
    assert FC.fit_dispersion(rng.poisson(mu), mu) < 0.05


def test_intervals_poisson_limit_and_nesting():
    lo, hi = FC.interval(np.array([2.0]), 0.0, 0.8)
    assert (lo[0], hi[0]) == (0.0, 4.0)
    lo95, hi95 = FC.interval(np.array([2.0]), 0.5, 0.95)
    lo80, hi80 = FC.interval(np.array([2.0]), 0.5, 0.8)
    assert lo95[0] <= lo80[0] and hi95[0] >= hi80[0]
    assert FC.prob_at_least(1, np.array([1.0]), 0.0)[0] == pytest.approx(1 - np.exp(-1))


def test_historical_mean_uses_only_training_rows():
    hm = FC.HistoricalMean().fit(None, np.array([1.0, 3.0, 10.0]), np.array([5, 5, 6]))
    assert hm.predict(None, np.array([5, 6, 7])).tolist() == [2.0, 10.0, pytest.approx(14 / 3)]


# ------------------------------------------------------------------ walk-forward training
@pytest.fixture(scope="module")
def trained():
    """Run the real selection + walk-forward code on a small synthetic matrix (3 active regions)."""
    epoch = T.ms(T.EPOCH)
    end = T.ms(T.VAL_END) + 60 * WEEK
    n_weeks = (end - epoch) // WEEK
    rng = np.random.default_rng(0)
    C = np.zeros((regions.N_REGIONS, n_weeks), dtype=np.int64)
    active = [R0] + regions.neighbors(R0)[:2]
    for i, r in enumerate(active):
        C[r] = rng.poisson(1.0 + i, n_weeks)
    L = np.zeros_like(C)
    return T.train_threshold(C, L, n_weeks, 4.5), active, n_weeks


def test_walk_forward_periods_are_ordered_and_disjoint(trained):
    (_, meta, _), _, _ = trained
    p = meta["periods"]
    assert p["train"]["to"] <= p["validation"]["from"] < p["validation"]["to"] <= p["test"]["from"]
    for fold in p["test"]["folds"]:
        assert fold["train_to"] <= fold["test_from"]       # every fold trains only on targets before it
    assert sum(f["test_origins"] for f in p["test"]["folds"]) == p["test"]["origins"]


def test_training_selects_regions_and_reports_baselines(trained):
    (est, meta, backtest), active, _ = trained
    assert sorted(r["index"] for r in meta["regions"]) == sorted(active)
    assert meta["selected_model"] in meta["validation_metrics"]
    for name in ("historical_mean", "rolling_mean_4w", "rolling_mean_52w"):
        assert name in meta["test_metrics"]
    assert meta["alpha"] >= 0 and hasattr(est, "predict")
    assert len(backtest["weeks"]) == len(next(iter(backtest["regions"].values()))["observed"])


def test_saved_artifact_round_trip(trained, tmp_path):
    (est, meta, _), _, _ = trained
    FC.save_artifact(str(tmp_path), 4.5, est, meta)
    m = FC.ForecastModel(str(tmp_path), 4.5)
    assert m.ready, m.load_error
    X = np.zeros((2, len(RF.FEATURE_NAMES)))
    assert np.all(m.predict(X) > 0)
    with pytest.raises(ValueError):
        m.predict(np.zeros((1, 3)))


# ------------------------------------------------------------------ committed artifacts
@pytest.mark.parametrize("thr", [4.5, 5.0])
def test_committed_forecast_artifacts_load(thr):
    m = FC.ForecastModel(config.MODEL_DIR, thr)
    assert m.ready, m.load_error
    meta = m.meta
    assert meta["feature_names"] == RF.FEATURE_NAMES and meta["threshold"] == thr
    sel = meta["selected_model"]
    assert {"mae", "rmse", "poisson_deviance", "coverage_80"} <= set(meta["test_metrics"][sel])
    assert meta["periods"]["validation"]["to"] <= meta["periods"]["test"]["from"]


def test_incompatible_or_missing_artifacts_are_refused(tmp_path):
    assert not FC.ForecastModel(str(tmp_path), 4.5).ready
    for name in ("forecast_m4.5.joblib", "forecast_m4.5.json"):
        shutil.copy(f"{config.MODEL_DIR}/{name}", tmp_path / name)
    meta_path = tmp_path / "forecast_m4.5.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["feature_names"] = meta["feature_names"][::-1]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    m = FC.ForecastModel(str(tmp_path), 4.5)
    assert not m.ready and "schema mismatch" in m.load_error
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((1, len(RF.FEATURE_NAMES))))
