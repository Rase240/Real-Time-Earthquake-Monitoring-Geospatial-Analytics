"""Module C: reference distribution, multiple comparisons, gaps and flag logic."""
import json

import numpy as np
import pytest
from scipy import stats

from analytics import anomaly as AN
from analytics import regions

R0 = int(regions.region_index([35.0], [140.0])[0])
DAY = AN.DAY_MS


def matrix(ref_counts, obs_counts, region=R0):
    C = np.zeros((regions.N_REGIONS, len(ref_counts) + len(obs_counts)), dtype=np.int64)
    C[region] = list(ref_counts) + list(obs_counts)
    return C


def test_benjamini_hochberg_known_values():
    # sorted p 0.01, 0.03, 0.04, 0.5 -> p*n/rank 0.04, 0.06, 0.0533, 0.5 -> running minimum from the top
    q = AN.benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert np.allclose(AN.benjamini_hochberg([0.2]), [0.2])
    assert len(AN.benjamini_hochberg([])) == 0


def test_reference_moments_poisson_when_not_overdispersed():
    mean, var, alpha = AN.reference_moments([2, 2, 2, 2], 4)
    assert mean == 8 and alpha == 0
    assert var == pytest.approx(8 + 16 * 2 / 4)          # k*w + k^2*w/n with w = mu
    mean2, var2, _ = AN.reference_moments([2, 2, 2, 2], 4, long_alpha=1.0)
    assert mean2 == 8 and var2 > var                     # long-term dispersion widens the reference
    means, _, alphas = AN.reference_moments([[2, 2, 2, 2], [0, 0, 0, 8]], 4, long_alpha=[0.5, 0.0])
    assert np.allclose(means, [8, 8]) and alphas[0] == 0.5 and alphas[1] == AN.moment_dispersion([0, 0, 0, 8])


def test_moment_dispersion():
    assert AN.moment_dispersion([1, 1, 1, 1]) == 0
    x = np.array([0, 0, 0, 8])
    m, s2 = x.mean(), x.var(ddof=1)
    assert AN.moment_dispersion(x) == pytest.approx((s2 - m) / m ** 2)
    assert AN.moment_dispersion([0, 0, 0]) == 0


def test_mle_dispersion_recovers_alpha_and_resists_one_huge_week():
    rng = np.random.default_rng(3)
    mu, alpha = 2.0, 1.5
    r = 1 / alpha
    x = rng.negative_binomial(r, r / (r + mu), 4000)
    assert AN.mle_dispersion(x) == pytest.approx(alpha, rel=0.1)
    assert AN.mle_dispersion(rng.poisson(2.0, 4000)) < 0.02
    assert AN.mle_dispersion([1, 1, 1, 1]) == 0 and AN.mle_dispersion([0, 0, 0]) == 0
    # One megathrust-like week dominates the variance (moment estimate) far more than the likelihood.
    quiet = rng.poisson(1.0, 780)
    quiet[400] = 800
    assert AN.moment_dispersion(quiet) > 50 * AN.mle_dispersion(quiet) > 0


def test_conditional_p_value_matches_binomial_and_nb_convolution():
    # alpha = 0: classical conditional test, Y | N ~ Binomial(N, k / (n + k)).
    p = AN.conditional_p_value(9, 20, 4, 52, 0.0)
    assert p == pytest.approx(stats.binom.sf(8, 29, 4 / 56))
    # alpha > 0: P(Y = y | X + Y = N) from the two NB pmfs, whatever the mean.
    k, n, alpha, x, y = 4, 52, 0.7, 30, 9
    r = 1 / alpha
    for mu in (0.3, 3.0):
        q = r / (r + mu)
        tot = x + y
        joint = stats.nbinom.pmf(np.arange(tot + 1), k * r, q) * stats.nbinom.pmf(tot - np.arange(tot + 1), n * r, q)
        assert AN.conditional_p_value(y, x, k, n, alpha) == pytest.approx(joint[y:].sum() / joint.sum(), rel=1e-6)
    assert AN.conditional_p_value(0, 30, 4, 52, 0.7) == 1.0
    assert np.all(np.diff(AN.conditional_p_value(9, 30, 4, 52, [0.0, 0.5, 2.0, 10.0])) > 0)   # heavier tail, larger p


def test_conditional_test_holds_its_level_whatever_the_mean():
    """Unlike plugging in the estimated mean, conditioning keeps P(p <= level) <= level for NB data."""
    rng = np.random.default_rng(4)
    k, n, alpha, runs = 4, 52, 3.0, 20000
    r = 1 / alpha
    for mu in (0.2, 1.0, 5.0):
        W = rng.negative_binomial(r, r / (r + mu), (runs, n + k))
        X, Y = W[:, :n].sum(axis=1), W[:, n:].sum(axis=1)
        p = AN.conditional_p_value(Y, X, k, n, alpha)
        for level in (0.01, 0.05):
            assert (p <= level).mean() <= level + 3 * np.sqrt(level / runs)


def test_large_excess_is_elevated_with_explanation_fields():
    C = matrix([1] * 52, [1, 1, 1, 30])
    res, tested = AN.detect(C, 4, 52, 0.05)
    row = res[0]
    assert tested == 1 and row["status"] == "elevated" and row["observed"] == 33
    assert row["expected"] == pytest.approx(4.0) and row["q_value"] <= 0.05
    for key in ("reference_total", "reference_weekly_mean", "p_value", "ratio", "distribution"):
        assert key in row


def test_typical_activity_is_not_elevated():
    rng = np.random.default_rng(0)
    C = matrix(rng.poisson(2, 52), [2, 3, 1, 2])
    assert AN.detect(C, 4, 52, 0.05)[0][0]["status"] == "not_elevated"


def test_minimum_observed_count_required():
    C = matrix([0] * 47 + [1] * 5, [0, 0, 2, 0])          # tiny expected, only 2 observed
    assert AN.detect(C, 4, 52, 0.05)[0][0]["status"] != "elevated"


def test_sparse_reference_is_reported_not_called_normal():
    C = matrix([0] * 50 + [1, 1], [3, 0, 0, 0])
    row = AN.detect(C, 4, 52, 0.05, min_reference_events=5)[0][0]
    assert row["status"] == "insufficient_reference" and row["p_value"] is None


def test_empty_region_is_omitted():
    assert AN.detect(matrix([0] * 52, [0] * 4), 4, 52, 0.05)[0] == []


def test_gaps_turn_normal_results_into_data_incomplete():
    rng = np.random.default_rng(1)
    C = matrix(rng.poisson(2, 52), [2, 2, 2, 2])
    row = AN.detect(C, 4, 52, 0.05, gaps_in_window=2)[0][0]
    assert row["status"] == "data_incomplete"


def test_multiple_comparisons_reduce_flags():
    """A p-value that would pass alone is not enough among many tested regions."""
    rng = np.random.default_rng(2)
    C = np.zeros((regions.N_REGIONS, 56), dtype=np.int64)
    idx = np.arange(200)
    C[idx] = rng.poisson(2, (200, 56))
    C[idx[0], 52:] = [4, 4, 4, 5]                          # mildly high: p ~ 0.01 on its own
    alone, _ = AN.detect(C, 4, 52, 0.05, region_idx=[idx[0]])
    many, _ = AN.detect(C, 4, 52, 0.05, region_idx=idx)
    row = next(r for r in many if r["index"] == idx[0])
    assert row["q_value"] >= alone[0]["q_value"]


def test_width_mismatch_rejected():
    with pytest.raises(ValueError):
        AN.detect(np.zeros((regions.N_REGIONS, 10)), 4, 52, 0.05)


def test_gap_days_finds_days_without_events():
    start = 1000 * DAY
    times = [start + DAY // 2, start + 2 * DAY + 5]
    assert AN.gap_days(times, start, start + 3 * DAY) == [start + DAY]
    assert AN.gap_days(times, start + 1, start + DAY) == []       # no whole day inside the interval


def test_load_reference_validation(tmp_path):
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"period": {"from": "a"}, "thresholds": {"4.5": [0.5] * regions.N_REGIONS}}))
    table, period, err = AN.load_reference(str(good))
    assert err is None and table[4.5].shape == (regions.N_REGIONS,)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"thresholds": {"4.5": [0.5] * 3}}))
    assert AN.load_reference(str(bad))[0] == {} and AN.load_reference(str(bad))[2]
    assert AN.load_reference(str(tmp_path / "missing.json"))[0] == {}


def test_rolling_rule_baseline():
    C = matrix([1] * 52, [2, 2, 2, 2])
    assert AN.rolling_rule(C, 4, 52)[R0]                    # 8 >= 2 * 4
    assert not AN.rolling_rule(matrix([1] * 52, [1, 1, 1, 1]), 4, 52)[R0]


def test_long_term_dispersion_makes_the_test_more_cautious():
    C = matrix([1] * 52, [1, 1, 1, 9])                      # 12 observed vs ~4 expected
    light = AN.detect(C, 4, 52, 0.05)[0][0]
    heavy = AN.detect(C, 4, 52, 0.05, long_alpha=np.full(regions.N_REGIONS, 5.0))[0][0]
    assert light["status"] == "elevated" and light["distribution"] == "poisson"
    assert heavy["status"] == "not_elevated" and heavy["distribution"] == "negative_binomial"
    assert heavy["p_value"] > light["p_value"] and heavy["long_term_dispersion"] == 5.0


def test_detect_row_fields_are_mutually_consistent():
    """expected, dispersion, p and q in each row are exactly what the documented formulas give for that row."""
    rng = np.random.default_rng(8)
    C = np.zeros((regions.N_REGIONS, 56), dtype=np.int64)
    idx = np.arange(60)
    C[idx] = rng.negative_binomial(1.0, 0.4, (60, 56))
    C[idx[:3], 52:] += 15
    la = rng.uniform(0, 2, regions.N_REGIONS)
    res, n_tested = AN.detect(C, 4, 52, 0.05, long_alpha=la)
    tested = [r for r in res if r["p_value"] is not None]
    assert len(tested) == n_tested
    for r in tested:
        ref = C[r["index"], :52]
        alpha = max(AN.moment_dispersion(ref), la[r["index"]])
        assert r["weekly_dispersion"] == pytest.approx(alpha, abs=1e-4)
        assert r["long_term_dispersion"] == pytest.approx(la[r["index"]], abs=1e-4)
        assert r["expected"] == pytest.approx(4 * ref.mean(), abs=1e-3)
        assert r["reference_total"] == ref.sum() and r["observed"] == C[r["index"], 52:].sum()
        assert r["p_value"] == pytest.approx(float(AN.conditional_p_value(r["observed"], ref.sum(), 4, 52, alpha)), rel=1e-3)
    p = np.array([float(AN.conditional_p_value(r["observed"], r["reference_total"], 4, 52,
                                               max(AN.moment_dispersion(C[r["index"], :52]), la[r["index"]])))
                  for r in tested])
    q = AN.benjamini_hochberg(p)
    for r, qq in zip(tested, q, strict=True):
        assert r["q_value"] == pytest.approx(qq, rel=1e-3)
        assert (r["status"] == "elevated") == (qq <= 0.05 and r["observed"] >= AN.MIN_OBSERVED)


def test_smallest_attainable_p_grows_with_dispersion():
    """P(Y = N | N) increases with alpha, so a cell that cannot reach significance at the floor cannot at all."""
    for N in (2, 5, 20, 60):
        pm = AN.conditional_p_value(np.full(5, N), np.zeros(5), 4, 52, [0.0, 0.3, 1.0, 5.0, 50.0])
        assert np.all(pm > 0) and np.all(np.diff(pm) > 0) and pm[-1] <= 4 / 56 + 1e-9


def test_extreme_upper_tail_p_value_is_accurate_not_zero():
    """1 - cdf cancels to 0 in the far tail; the p-value must still equal the pmf summed over the tail."""
    k, n, alpha, y, x = 4, 52, 0.3, 40, 10
    p = float(AN.conditional_p_value(y, x, k, n, alpha))
    exact = stats.betabinom.pmf(np.arange(y, x + y + 1), x + y, k / alpha, n / alpha).sum()
    assert 0 < p == pytest.approx(exact, rel=1e-9)


def test_wilson_interval_known_values():
    from analytics.evaluation import wilson_interval
    for k, n in ((56, 1000), (3, 40), (980, 1000)):
        ref = stats.binomtest(k, n).proportion_ci(method="wilson")
        assert wilson_interval(k, n) == [pytest.approx(ref.low, abs=1e-4), pytest.approx(ref.high, abs=1e-4)]
    assert wilson_interval(0, 50)[0] == 0.0 and wilson_interval(0, 0) == [None, None]


def test_null_simulation_is_reproducible():
    from analytics import evaluation as EV
    mu, var = np.full(regions.N_REGIONS, 1.5), np.full(regions.N_REGIONS, 4.0)
    a = EV._simulate(np.random.default_rng(7), mu, var, np.arange(20), 56)
    b = EV._simulate(np.random.default_rng(7), mu, var, np.arange(20), 56)
    assert np.array_equal(a, b) and a[20:].sum() == 0


def test_injection_strata_add_up_to_the_totals():
    from analytics import evaluation as EV
    rng = np.random.default_rng(9)
    C = np.zeros((regions.N_REGIONS, 64), dtype=np.int64)
    C[:80] = rng.poisson(rng.uniform(0.1, 4.0, (80, 1)), (80, 64))
    floors = {"mle": np.zeros(regions.N_REGIONS)}
    tallies, strata = EV._injection(C, [60, 64], 4, 52, 0.05, floors, {"x": ("mle", True, True)})
    for f in EV.INJECTION_FACTORS:
        assert strata["x"][f][:, 1].sum() == tallies["x"][f][3] and strata["x"][f][:, 0].sum() == tallies["x"][f][0]
    det = [tallies["x"][f][0] / tallies["x"][f][3] for f in EV.INJECTION_FACTORS]
    assert det == sorted(det)                                   # larger injected excess, more detections


def test_committed_anomaly_artifacts_describe_the_active_method():
    import config
    from analytics import evaluation as EV
    with open(f"{config.MODEL_DIR}/anomaly_eval.json", encoding="utf-8") as fh:
        ev = json.load(fh)
    table, period, err = AN.load_reference(f"{config.MODEL_DIR}/anomaly_reference.json")
    assert err is None and set(table) == {4.5, 5.0} and period["to"] == "2016-01-04"
    assert "beta-binomial" in ev["configuration"]["test"] and "maximum likelihood" in ev["configuration"]["long_term_dispersion"]
    assert ev["method_comparison"]["adopted"] == EV.ADOPTED
    assert ev["configuration"]["window_weeks"] == AN.DEFAULTS["window_weeks"]
    lo, hi = ev["null_simulation"]["share_of_runs_with_any_false_alert_wilson95"]
    assert lo <= ev["null_simulation"]["share_of_runs_with_any_false_alert"] <= hi


def test_evaluation_variant_matches_detect():
    """The comparison helper in the offline evaluation reproduces AN.detect for the adopted variant."""
    from analytics import evaluation as EV
    rng = np.random.default_rng(5)
    C = np.zeros((regions.N_REGIONS, 56), dtype=np.int64)
    idx = np.arange(150)
    C[idx] = rng.negative_binomial(0.5, 0.5 / 1.5, (150, 56))
    C[idx[:5], 52:] += 12
    la = rng.uniform(0, 3, regions.N_REGIONS)
    res, _ = AN.detect(C, 4, 52, 0.05, long_alpha=la)
    cells, R, obs = EV._tested(C, 4, 52)
    floor, cond, recent = EV.VARIANTS[EV.ADOPTED]
    el = EV._elevated(EV._variant_p(R, obs, 4, la[cells], cond, recent), obs, 0.05)
    assert {r["index"] for r in res if r["status"] == "elevated"} == set(cells[el].tolist())
    assert el.any()
