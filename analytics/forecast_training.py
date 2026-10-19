"""
Training and evaluation of the seven-day regional count forecast (Module B).

Dataset
-------
* Weekly count matrices from the downloaded catalogue, weeks starting on
  Mondays 00:00 UTC from EPOCH (2000-01-03).
* One sample per (origin week k, eligible region r): features from columns < k
  (analytics.regional_features), target = column k.
* The first 52 weeks are warm-up (features need 52 weeks of history).

Temporal design (no shuffling anywhere)
---------------------------------------
* train       : samples whose target week ends on or before TRAIN_END (2016-01-04)
* validation  : targets in [TRAIN_END, VAL_END)  (2016-2019) -> model/hyperparameter
                selection and NB dispersion; never used for the test score.
* test        : targets from VAL_END (2020-01-06) to the end of the catalogue,
                evaluated walk-forward: for each calendar year Y, every model is refit
                on all samples whose target ends before the first origin of Y and
                predicts the weeks of Y. Hyperparameters and alpha stay as chosen on
                validation, so the test period never influences any choice.
* Region eligibility is decided from the training period only.
* Preprocessing (log1p + StandardScaler for the GLM) lives inside the estimator
  pipeline, so it is refit on each training window only.

The served model is the selected configuration refit on every available sample;
its quality is estimated by the walk-forward test above.
"""
import logging
from datetime import datetime, timezone

import numpy as np

from analytics import forecast as FC
from analytics import regions
from analytics.regional_features import FEATURE_NAMES, FEATURES, LARGE_MAG, features_at, targets_at

log = logging.getLogger("terrawatcher.forecast_training")

WEEK_MS = regions.WEEK_MS
EPOCH = datetime(2000, 1, 3, tzinfo=timezone.utc)
TRAIN_END = datetime(2016, 1, 4, tzinfo=timezone.utc)
VAL_END = datetime(2020, 1, 6, tzinfo=timezone.utc)
THRESHOLDS = (4.5, 5.0)
MIN_TRAIN_WEEKLY_MEAN = 0.1     # eligibility: >= ~5 qualifying events per year in training
SEED = 42
GLM_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
GBM_GRID = ({"max_leaf_nodes": 15, "learning_rate": 0.05, "max_iter": 300},
            {"max_leaf_nodes": 31, "learning_rate": 0.05, "max_iter": 300})
SELECTION_TOLERANCE = 0.01      # prefer the GLM unless trees are >1% better in validation deviance
BACKTEST_WEEKS = 104


def ms(dt):
    return int(dt.timestamp() * 1000)


def iso_day(ms_):
    return datetime.fromtimestamp(ms_ / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def col_of(dt_ms, epoch_ms):
    off = dt_ms - epoch_ms
    if off % WEEK_MS:
        raise ValueError("boundary is not aligned to the weekly grid")
    return off // WEEK_MS


def build_matrices(cat, end_ms, epoch_ms=None):
    epoch_ms = ms(EPOCH) if epoch_ms is None else epoch_ms
    n_weeks = (end_ms - epoch_ms) // WEEK_MS
    t, m, la, lo = (cat[c].to_numpy() for c in ("time", "mag", "latitude", "longitude"))
    mats = {thr: regions.count_matrix(t, m, la, lo, thr, epoch_ms, n_weeks) for thr in THRESHOLDS}
    large = regions.count_matrix(t, m, la, lo, LARGE_MAG, epoch_ms, n_weeks)
    return mats, large, n_weeks


def make_dataset(C, L, ks, reg):
    X = features_at(C, L, ks, reg)
    y = targets_at(C, ks, reg)
    origin = np.repeat(ks, len(reg))
    region = np.tile(reg, len(ks))
    return X, y, origin, region


def _candidates():
    out = {"historical_mean": lambda: FC.HistoricalMean(), "rolling_mean_4w": lambda: FC.RollingMean("mean_4w"),
           "rolling_mean_52w": lambda: FC.RollingMean("mean_52w")}
    for l2 in GLM_GRID:
        out[f"poisson_glm(l2={l2:g})"] = (lambda l2=l2: FC.make_glm(l2))
    for g in GBM_GRID:
        out[f"gbm_poisson(leaves={g['max_leaf_nodes']})"] = (lambda g=g: FC.make_gbm(seed=SEED, **g))
    return out


def _fit_predict(factory, Xtr, ytr, rtr, Xte, rte):
    model = factory()
    if isinstance(model, (FC.HistoricalMean, FC.RollingMean)):
        model.fit(Xtr, ytr, rtr)
        return model, np.maximum(model.predict(Xte, rte), FC.MIN_MU)
    model.fit(Xtr, ytr)
    return model, np.maximum(model.predict(Xte), FC.MIN_MU)


def family(name):
    return "glm" if name.startswith("poisson_glm") else "gbm" if name.startswith("gbm") else "baseline"


def dispersion_evidence(y, mu, origin, split_col):
    """
    Out-of-fold NB log score (lower is better) of a constant alpha vs the mean-dependent curve, on validation
    predictions only: fit on the origins before `split_col`, score those after, and vice versa.
    """
    first = origin < split_col
    out = {}
    for kind, fit in (("constant_alpha", FC.fit_dispersion), ("mean_dependent_curve", FC.fit_dispersion_curve)):
        total = 0.0
        for fm, em in ((first, ~first), (~first, first)):
            total += -FC.nb_logpmf(y[em], mu[em], fit(y[fm], mu[fm])).sum()
        out[kind] = round(float(total / len(y)), 5)
    out["note"] = "Validation period split in two halves by origin; each half's dispersion is scored on the other half."
    return out


def week_bootstrap_skill(y, mu_sel, mu_other, origin, n_boot=1000, seed=SEED):
    """
    95% interval for deviance skill 1 - D(selected) / D(other) by resampling whole forecast weeks (all regions of
    an origin together, since regions in the same week share global activity).
    """
    d_sel, d_oth = FC.poisson_unit_deviance(y, mu_sel), FC.poisson_unit_deviance(y, mu_other)
    weeks, inv = np.unique(origin, return_inverse=True)
    s_sel = np.bincount(inv, weights=d_sel, minlength=len(weeks))
    s_oth = np.bincount(inv, weights=d_oth, minlength=len(weeks))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(weeks), (n_boot, len(weeks)))
    skill = 1 - s_sel[idx].sum(axis=1) / s_oth[idx].sum(axis=1)
    return [round(float(v), 4) for v in np.quantile(skill, [0.025, 0.975])]


def train_threshold(C, L, n_weeks, threshold, epoch_ms=None):
    """Full selection + walk-forward evaluation for one threshold. Returns (estimator, meta, backtest)."""
    epoch_ms = ms(EPOCH) if epoch_ms is None else epoch_ms
    k_train_end, k_val_end = col_of(ms(TRAIN_END), epoch_ms), col_of(ms(VAL_END), epoch_ms)
    first = 52
    k_all = np.arange(first, n_weeks)                  # target column k exists for k <= n_weeks - 1
    k_train = k_all[k_all + 1 <= k_train_end]
    k_val = k_all[(k_all >= k_train_end) & (k_all + 1 <= k_val_end)]
    k_test = k_all[k_all >= k_val_end]

    train_mean = C[:, k_train].mean(axis=1)
    reg = np.flatnonzero(train_mean >= MIN_TRAIN_WEEKLY_MEAN)
    log.info("M%.1f: %d eligible regions; origins train %d, val %d, test %d", threshold, len(reg),
             len(k_train), len(k_val), len(k_test))

    Xtr, ytr, _, rtr = make_dataset(C, L, k_train, reg)
    Xva, yva, ova, rva = make_dataset(C, L, k_val, reg)

    # ---- validation: every candidate fit on train, scored on validation. The dispersion (constant alpha and the
    #      mean-dependent curve) is fitted on these validation predictions; the test period never touches it.
    cands = _candidates()
    val, alphas, curves, val_mu = {}, {}, {}, {}
    for name, factory in cands.items():
        _, mu = _fit_predict(factory, Xtr, ytr, rtr, Xva, rva)
        val_mu[name] = mu
        alphas[name] = FC.fit_dispersion(yva, mu)
        curves[name] = FC.fit_dispersion_curve(yva, mu)
        val[name] = FC.count_metrics(yva, mu, curves[name])
        log.info("  val %-26s dev %.4f  MAE %.4f  alpha %.3f  curve a %.3f b %.3f", name, val[name]["poisson_deviance"],
                 val[name]["mae"], alphas[name], curves[name]["a"], curves[name]["b"])

    def best_of(fam):
        names = [n for n in cands if family(n) == fam]
        return min(names, key=lambda n: val[n]["poisson_deviance"])
    best_glm, best_gbm = best_of("glm"), best_of("gbm")
    gbm_gain = 1 - val[best_gbm]["poisson_deviance"] / val[best_glm]["poisson_deviance"]
    selected = best_gbm if gbm_gain > SELECTION_TOLERANCE else best_glm
    best_baseline = min((n for n in cands if family(n) == "baseline"), key=lambda n: val[n]["poisson_deviance"])
    log.info("  selected %s (GBM validation deviance gain over GLM %.2f%%)", selected, 100 * gbm_gain)
    disp_evidence = dispersion_evidence(yva, val_mu[selected], ova, k_train_end + (k_val_end - k_train_end) // 2)
    log.info("  dispersion out-of-fold NB log score: constant %.5f, curve %.5f", disp_evidence["constant_alpha"],
             disp_evidence["mean_dependent_curve"])

    # ---- walk-forward test: yearly expanding-window refits
    evaluated = ["historical_mean", "rolling_mean_4w", "rolling_mean_52w", best_glm, best_gbm]
    years = sorted({datetime.fromtimestamp((epoch_ms + k * WEEK_MS) / 1000, tz=timezone.utc).year for k in k_test})
    preds = {n: [] for n in evaluated}
    y_test, k_rows, r_rows, folds = [], [], [], []
    for year in years:
        fold_k = np.array([k for k in k_test if datetime.fromtimestamp((epoch_ms + k * WEEK_MS) / 1000,
                                                                       tz=timezone.utc).year == year])
        k_fit = k_all[k_all + 1 <= fold_k.min()]
        Xf, yf, _, rf = make_dataset(C, L, k_fit, reg)
        Xt, yt, ot, rt = make_dataset(C, L, fold_k, reg)
        for n in evaluated:
            _, mu = _fit_predict(cands[n], Xf, yf, rf, Xt, rt)
            preds[n].append(mu)
        y_test.append(yt)
        k_rows.append(ot)
        r_rows.append(rt)
        folds.append({"year": int(year), "train_origins": int(len(k_fit)), "test_origins": int(len(fold_k)),
                      "train_to": iso_day(epoch_ms + (k_fit.max() + 1) * WEEK_MS),
                      "test_from": iso_day(epoch_ms + fold_k.min() * WEEK_MS),
                      "test_to": iso_day(epoch_ms + (fold_k.max() + 1) * WEEK_MS)})
        log.info("  walk-forward %d: fit on %d origins, test %d origins", year, len(k_fit), len(fold_k))
    y_test = np.concatenate(y_test)
    k_rows, r_rows = np.concatenate(k_rows), np.concatenate(r_rows)
    preds = {n: np.concatenate(v) for n, v in preds.items()}
    test = {n: FC.count_metrics(y_test, preds[n], curves[n], central=True) for n in evaluated}
    test_constant_alpha = FC.count_metrics(y_test, preds[selected], alphas[selected], central=True)
    year_of = np.array([datetime.fromtimestamp((epoch_ms + k * WEEK_MS) / 1000, tz=timezone.utc).year for k in k_rows])
    by_year = {int(y): {n: FC.count_metrics(y_test[year_of == y], preds[n][year_of == y], curves[n])
                        for n in (selected, "historical_mean", "rolling_mean_4w")} for y in years}

    sel_mu = preds[selected]
    region_rows = []
    for r in reg:
        m = r_rows == r
        sm = FC.count_metrics(y_test[m], sel_mu[m], curves[selected])
        bm = FC.count_metrics(y_test[m], preds["historical_mean"][m])
        b52 = FC.count_metrics(y_test[m], preds["rolling_mean_52w"][m])
        region_rows.append({"index": int(r), "id": regions.region_id(r), "label": regions.label(r),
                            "bounds": regions.bounds(r), "train_weekly_mean": round(float(train_mean[r]), 4),
                            "test": {k: sm[k] for k in ("n", "mean_observed", "mean_predicted", "mae", "rmse",
                                                        "poisson_deviance", "coverage_80", "coverage_95",
                                                        "model_implied_coverage_80", "mean_width_80")},
                            "baseline_historical_mean": {k: bm[k] for k in ("mae", "poisson_deviance")},
                            "baseline_rolling_mean_52w": {k: b52[k] for k in ("mae", "poisson_deviance")}})

    # Reliability by predicted-mean bin (selected model).
    edges = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, np.inf]
    reliability = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        m = (sel_mu >= a) & (sel_mu < b)
        if m.sum():
            lo80, hi80 = FC.interval(sel_mu[m], curves[selected], 0.8)
            reliability.append({"predicted_from": a, "predicted_to": None if np.isinf(b) else b, "n": int(m.sum()),
                                "mean_predicted": round(float(sel_mu[m].mean()), 4),
                                "mean_observed": round(float(y_test[m].mean()), 4),
                                "coverage_80": round(float(np.mean((y_test[m] >= lo80) & (y_test[m] <= hi80))), 4),
                                "model_implied_coverage_80": round(float(np.mean(
                                    FC.interval_mass(lo80, hi80, sel_mu[m], curves[selected]))), 4),
                                "mean_width_80": round(float(np.mean(hi80 - lo80)), 3)})

    # Regional and activity-level breakdown: does the aggregate gain hold everywhere?
    d_sel = FC.poisson_unit_deviance(y_test, sel_mu)
    regional = {"regions": len(reg)}
    for base in ("historical_mean", "rolling_mean_52w"):
        d_b = FC.poisson_unit_deviance(y_test, preds[base])
        better = [float(d_sel[r_rows == r].sum() < d_b[r_rows == r].sum()) for r in reg]
        regional[f"regions_better_than_{base}"] = int(sum(better))
    tiers = np.quantile(train_mean[reg], [1 / 3, 2 / 3])
    tier_of = np.digitize(train_mean[r_rows], tiers)
    region_tier = np.digitize(train_mean[reg], tiers)
    by_activity = []
    for t_, name in enumerate(("low", "middle", "high")):
        m = tier_of == t_
        sm = FC.count_metrics(y_test[m], sel_mu[m], curves[selected])
        hm = FC.count_metrics(y_test[m], preds["historical_mean"][m])
        in_tier = train_mean[reg][region_tier == t_]
        by_activity.append({"tier": name, "regions": int(len(np.unique(r_rows[m]))),
                            "train_weekly_mean_range": [round(float(in_tier.min()), 3), round(float(in_tier.max()), 3)],
                            "n": sm["n"], "share_zero_weeks": round(float(np.mean(y_test[m] == 0)), 4),
                            "poisson_deviance": sm["poisson_deviance"], "historical_mean_deviance": hm["poisson_deviance"],
                            "deviance_skill_vs_historical_mean": round(1 - sm["poisson_deviance"] / hm["poisson_deviance"], 4),
                            "coverage_80": sm["coverage_80"], "model_implied_coverage_80": sm["model_implied_coverage_80"],
                            "mean_width_80": sm["mean_width_80"]})
    regional["by_activity_tier"] = by_activity
    regional["note"] = ("Per-region test periods hold ~350 weeks with mostly small counts; per-region skill is noisy and "
                        "is not a reliable ranking of regions.")

    # ---- final served model: selected configuration on every sample
    Xall, yall, _, rall = make_dataset(C, L, k_all, reg)
    final, _ = _fit_predict(cands[selected], Xall, yall, rall, Xall[:1], rall[:1])

    coefficients = None
    if family(selected) == "glm":
        glm = final[-1]
        coefficients = {"intercept": round(float(glm.intercept_), 5),
                        "per_standardised_log1p_feature": {n: round(float(c), 5) for n, c in
                                                           zip(FEATURE_NAMES, glm.coef_, strict=True)}}

    # Backtest series for the UI: last BACKTEST_WEEKS test weeks per region.
    k_back = k_test[-BACKTEST_WEEKS:]
    backtest = {"weeks": [iso_day(epoch_ms + k * WEEK_MS) for k in k_back], "regions": {}}
    for r in reg:
        m = (r_rows == r) & np.isin(k_rows, k_back)
        order = np.argsort(k_rows[m])
        backtest["regions"][regions.region_id(r)] = {
            "observed": y_test[m][order].astype(int).tolist(),
            "forecast": np.round(sel_mu[m][order], 3).tolist(),
            "baseline_historical_mean": np.round(preds["historical_mean"][m][order], 3).tolist()}

    skill = {n: round(1 - test[selected]["poisson_deviance"] / test[n]["poisson_deviance"], 4)
             for n in evaluated if n != selected}
    skill_ci = {n: week_bootstrap_skill(y_test, sel_mu, preds[n], k_rows) for n in evaluated if n != selected}
    meta = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": (f"Number of USGS catalogue earthquakes with magnitude >= {threshold} and epicentre in a "
                   f"{regions.CELL_DEG}x{regions.CELL_DEG} degree cell during the 7 days [T, T+7d) after the "
                   "forecast origin T."),
        "horizon_days": 7, "region_grid_deg": regions.CELL_DEG,
        "features": FEATURES,
        "selected_model": selected, "selection_rule": (
            "Lowest validation mean Poisson deviance within each family; gradient boosting is chosen over the GLM "
            f"only if it is more than {SELECTION_TOLERANCE:.0%} better on validation."),
        "alpha": alphas[selected], "dispersion_curve": curves[selected],
        "dispersion_method": ("NB2 with alpha(mu) = exp(a + b log mu), maximum likelihood on validation predictions "
                              "(alpha: the constant-dispersion MLE, for reference)"),
        "dispersion_evidence": disp_evidence,
        "interval_method": ("Shortest integer interval with predictive probability >= the level (NB2, dispersion "
                            "curve fitted on validation data)"),
        "periods": {"epoch": iso_day(epoch_ms), "warmup_weeks": 52,
                    "train": {"from": iso_day(epoch_ms + k_train.min() * WEEK_MS),
                              "to": iso_day(epoch_ms + (k_train.max() + 1) * WEEK_MS), "origins": int(len(k_train))},
                    "validation": {"from": iso_day(epoch_ms + k_val.min() * WEEK_MS),
                                   "to": iso_day(epoch_ms + (k_val.max() + 1) * WEEK_MS), "origins": int(len(k_val))},
                    "test": {"from": iso_day(epoch_ms + k_test.min() * WEEK_MS),
                             "to": iso_day(epoch_ms + (k_test.max() + 1) * WEEK_MS), "origins": int(len(k_test)),
                             "method": "walk-forward, yearly expanding-window refits", "folds": folds},
                    "served_model_fit": {"from": iso_day(epoch_ms + k_all.min() * WEEK_MS),
                                         "to": iso_day(epoch_ms + (k_all.max() + 1) * WEEK_MS)}},
        "samples": {"train": int(len(ytr)), "validation": int(len(yva)), "test": int(len(y_test)),
                    "served_fit": int(len(yall))},
        "eligibility": f"train-period mean weekly count >= {MIN_TRAIN_WEEKLY_MEAN}",
        "regions": [{"index": int(r), "id": regions.region_id(r), "label": regions.label(r),
                     "bounds": regions.bounds(r), "train_weekly_mean": round(float(train_mean[r]), 4)} for r in reg],
        "validation_metrics": val, "test_metrics": test, "test_by_year": by_year,
        "test_by_region": region_rows, "test_reliability": reliability, "test_regional_summary": regional,
        "test_selected_with_constant_alpha": test_constant_alpha,
        "deviance_skill_vs": skill, "deviance_skill_ci95_week_bootstrap": skill_ci, "alphas": alphas,
        "dispersion_curves": curves,
        "best_baseline_on_validation": best_baseline, "glm_coefficients": coefficients,
        "seed": SEED,
        "limitations": [
            "Counts aggregate a whole 10-degree cell; the forecast says nothing about where in the cell, when in "
            "the week, or how large any individual earthquake will be.",
            "Most of the forecast skill comes from persistence of ongoing sequences (recent counts); "
            "the model cannot anticipate a sequence that has not started.",
            "Intervals assume a negative-binomial distribution whose dispersion depends only on the predicted mean "
            "(not on the region); coverage is checked on the test period and varies by region.",
            "Counts are integers, so an interval cannot hold exactly 80% or 95% probability; compare empirical "
            "coverage with the model-implied coverage, which is at least the nominal level.",
            "Catalogue completeness at M4.5 varies by region and over time; counts are compared only within "
            "each cell, never across cells.",
            "Live features use the USGS feed at the forecast origin; events posted late lower the most recent count.",
        ],
    }
    return final, meta, backtest
