"""
Offline evaluation of Modules A, C and D (Module B lives in forecast_training).

Nothing here is a verified real-world label set. Synthetic data are used where a
known answer is needed, and are reported as synthetic; held-out historical data
are used for behaviour (flag rates, stability, fit success), not for accuracy
against invented ground truth.
"""
import logging
import math
from datetime import datetime, timezone

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from analytics import aftershock as AS
from analytics import anomaly as AN
from analytics import geo, regions, sequences
from analytics.forecast_training import EPOCH, TRAIN_END, VAL_END, build_matrices, col_of, ms

log = logging.getLogger("terrawatcher.evaluation")
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
T0 = 1_700_000_000_000


# =============================================================================
# Module A — synthetic clustering benchmark
# =============================================================================
def destination(lat, lon, bearing, dist_km):
    """Point at dist_km along `bearing` from (lat, lon) on the sphere (handles poles and the antimeridian)."""
    d = np.asarray(dist_km) / geo.EARTH_RADIUS_KM
    p1, l1, b = math.radians(lat), math.radians(lon), np.radians(bearing)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(b))
    l2 = l1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540) % 360 - 180


def _cluster(rng, lat, lon, t0_h, n, sigma_km=15.0, sigma_h=24.0):
    la, lo = destination(lat, lon, rng.uniform(0, 360, n), np.abs(rng.normal(0, sigma_km, n)))
    return la, lo, t0_h + np.abs(rng.normal(0, sigma_h, n))


def _noise(rng, n, days=30):
    z = rng.uniform(-1, 1, n)
    return np.degrees(np.arcsin(z)), rng.uniform(-180, 180, n), rng.uniform(0, days * 24, n)


def scenario(name, seed):
    """Return (events, truth labels aligned to the events that survive validation)."""
    rng = np.random.default_rng(seed)
    parts = []                                  # (lat, lon, hours, label)
    if name == "dense_clusters_with_noise":
        for i, (la, lo) in enumerate([(38, 142), (-20, -70), (36, 28), (-6, 130), (61, -150)]):
            parts.append((*_cluster(rng, la, lo, rng.uniform(0, 600), 30), i))
        parts.append((*_noise(rng, 100), -1))
    elif name == "simultaneous_separated":
        parts.append((*_cluster(rng, 0, 100, 100, 25), 0))
        la, lo = destination(0, 100, 90, 400)
        parts.append((*_cluster(rng, float(la), float(lo), 100, 25), 1))
    elif name == "same_place_separated_in_time":
        parts.append((*_cluster(rng, 35, 140, 100, 25), 0))
        parts.append((*_cluster(rng, 35, 140, 100 + 20 * 24, 25), 1))
    elif name == "overlapping_sequences":
        parts.append((*_cluster(rng, 10, 125, 100, 25), 0))
        la, lo = destination(10, 125, 90, 60)
        parts.append((*_cluster(rng, float(la), float(lo), 130, 25), 1))
    elif name == "antimeridian":
        parts.append((*_cluster(rng, -17, 179.95, 100, 30), 0))
        parts.append((*_noise(rng, 30), -1))
    elif name == "high_latitude":
        parts.append((*_cluster(rng, 84.5, 30, 100, 30), 0))
        parts.append((*_cluster(rng, 89.9, -100, 300, 30), 1))
        parts.append((*_noise(rng, 30), -1))
    elif name == "duplicates_and_invalid":
        parts.append((*_cluster(rng, 52, 160, 100, 30), 0))
        parts.append((*_noise(rng, 20), -1))
    else:
        raise ValueError(name)
    lat = np.concatenate([p[0] for p in parts])
    lon = np.concatenate([p[1] for p in parts])
    hrs = np.concatenate([p[2] for p in parts])
    lab = np.concatenate([np.full(len(p[0]), p[3]) for p in parts])
    events = [{"id": f"{name}-{i}", "time": int(T0 + h * HOUR_MS), "latitude": float(a), "longitude": float(b),
               "depth_km": 10.0, "mag": 4.6} for i, (a, b, h) in enumerate(zip(lat, lon, hrs, strict=True))]
    if name == "duplicates_and_invalid":
        events += [dict(events[i]) for i in range(0, 20, 2)]                 # 10 exact duplicates
        events += [{"id": f"bad-{i}", "time": T0, "latitude": v, "longitude": 0.0, "mag": 5.0}
                   for i, v in enumerate([None, float("nan"), 95.0, -91.0])]  # 4 invalid records
    return events, {e["id"]: int(lab_) for e, lab_ in zip(events, lab, strict=False)}


SCENARIOS = ["dense_clusters_with_noise", "simultaneous_separated", "same_place_separated_in_time",
             "overlapping_sequences", "antimeridian", "high_latitude", "duplicates_and_invalid"]


def _naive_degrees(df, eps_km, eps_hours, min_samples):
    """Comparison method: Euclidean distance on raw degrees (no wrapping, no cos(latitude))."""
    eps_deg = eps_km / 111.195
    X = np.column_stack([df["latitude"] / eps_deg, df["longitude"] / eps_deg,
                         (df["time"] - df["time"].min()) / HOUR_MS / eps_hours])
    return DBSCAN(eps=1.0, min_samples=min_samples).fit(X).labels_


def _space_only(df, eps_km, min_samples):
    X = np.radians(df[["latitude", "longitude"]].to_numpy())
    return DBSCAN(eps=eps_km / geo.EARTH_RADIUS_KM, min_samples=min_samples, metric="haversine",
                  algorithm="ball_tree").fit(X).labels_


def _score(truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    noise_t, noise_p = truth == -1, pred == -1
    out = {"ari": round(float(adjusted_rand_score(truth, pred)), 4),
           "nmi": round(float(normalized_mutual_info_score(truth, pred)), 4),
           "true_sequences": int(len(set(truth.tolist()) - {-1})),
           "found_sequences": int(len(set(pred.tolist()) - {-1}))}
    if noise_t.any():
        out["noise_recall"] = round(float((noise_p & noise_t).sum() / noise_t.sum()), 4)
    if noise_p.any():
        out["noise_precision"] = round(float((noise_p & noise_t).sum() / noise_p.sum()), 4)
    return out


def evaluate_sequences(seeds=(0, 1, 2, 3, 4)):
    from analytics.catalog import events_to_frame
    p = sequences.DEFAULTS
    results = {}
    for name in SCENARIOS:
        rows = {"proposed": [], "naive_degrees": [], "space_only": []}
        data_quality = None
        for seed in seeds:
            events, truth_map = scenario(name, seed)
            out = sequences.detect(events, **p)
            df, data_quality = events_to_frame(events)
            truth = [truth_map[i] for i in df["id"]]
            label_of = {sid: k for k, sid in enumerate(sorted({s["id"] for s in out["sequences"]}))}
            pred = [label_of.get(out["assignments"].get(i), -1) for i in df["id"]]
            rows["proposed"].append(_score(truth, pred))
            rows["naive_degrees"].append(_score(truth, _naive_degrees(df, p["eps_km"], p["eps_hours"], p["min_samples"])))
            rows["space_only"].append(_score(truth, _space_only(df, p["eps_km"], p["min_samples"])))
        summary = {}
        for method, scores in rows.items():
            summary[method] = {"ari_mean": round(float(np.mean([s["ari"] for s in scores])), 4),
                               "ari_min": round(float(np.min([s["ari"] for s in scores])), 4),
                               "nmi_mean": round(float(np.mean([s["nmi"] for s in scores])), 4),
                               "found_sequences": [s["found_sequences"] for s in scores],
                               "true_sequences": scores[0]["true_sequences"]}
            if "noise_recall" in scores[0]:
                summary[method]["noise_recall_mean"] = round(float(np.mean([s.get("noise_recall", 0) for s in scores])), 4)
        summary["data_quality_last_seed"] = data_quality
        results[name] = summary
        log.info("sequences %-30s ARI proposed %.3f  naive %.3f  space-only %.3f", name,
                 summary["proposed"]["ari_mean"], summary["naive_degrees"]["ari_mean"], summary["space_only"]["ari_mean"])

    sensitivity = []
    for eps_km in (25.0, 50.0, 100.0):
        for eps_h in (24.0, 72.0, 168.0):
            aris = []
            for seed in seeds:
                events, truth_map = scenario("dense_clusters_with_noise", seed)
                out = sequences.detect(events, eps_km=eps_km, eps_hours=eps_h, min_samples=p["min_samples"])
                df, _ = events_to_frame(events)
                label_of = {sid: k for k, sid in enumerate(sorted({s["id"] for s in out["sequences"]}))}
                pred = [label_of.get(out["assignments"].get(i), -1) for i in df["id"]]
                aris.append(adjusted_rand_score([truth_map[i] for i in df["id"]], pred))
            sensitivity.append({"eps_km": eps_km, "eps_hours": eps_h, "ari_mean": round(float(np.mean(aris)), 4)})
    overlap = []
    for eps_km in (10.0, 15.0, 25.0, 50.0):
        aris = []
        for seed in seeds:
            events, truth_map = scenario("overlapping_sequences", seed)
            out = sequences.detect(events, eps_km=eps_km, eps_hours=p["eps_hours"], min_samples=p["min_samples"])
            df, _ = events_to_frame(events)
            label_of = {sid: k for k, sid in enumerate(sorted({s["id"] for s in out["sequences"]}))}
            pred = [label_of.get(out["assignments"].get(i), -1) for i in df["id"]]
            aris.append(adjusted_rand_score([truth_map[i] for i in df["id"]], pred))
        overlap.append({"eps_km": eps_km, "ari_mean": round(float(np.mean(aris)), 4)})
    return {
        "overlapping_sequences_vs_eps_km": overlap,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": "synthetic (clearly labelled); never shown as real catalogue events",
        "parameters": p, "seeds": list(seeds), "scenarios": results,
        "parameter_sensitivity_dense_scenario": sensitivity,
        "synthetic_generator": ("Clusters: epicentres at half-normal great-circle distances (sigma 15 km) in uniform "
                                "random directions from a centre, times centre + half-normal (sigma 24 h). Noise: "
                                "uniform on the sphere over 30 days. Noise events that happen to fall inside a "
                                "cluster still count as noise in the reference labels."),
        "metrics_note": ("ARI and NMI compare against the synthetic reference labels with noise as its own class. "
                         "They are only meaningful because the labels are known; on the real catalogue there is "
                         "no ground truth, and internal indices (e.g. silhouette) are not reported because they "
                         "assume convex clusters and a defined distance for noise."),
    }


# =============================================================================
# Module C — anomaly detection behaviour
# =============================================================================
NULL_RUNS = 2000
SETTING_RUNS = 1000
INJECTION_FACTORS = (0.5, 1.0, 2.0, 4.0)

# Detector variants compared on identical data: (2001-2015 dispersion floor, conditional test?, also take
# the reference window's moment estimate?). The plug-in variants use the NB with the estimated mean
# (the method before the conditional test); ADOPTED is exactly what AN.detect computes.
ADOPTED = "conditional_mle_floor"
VARIANTS = {
    "previous_plugin_moment_floor": ("moment", False, True),
    "plugin_mle_floor": ("mle", False, True),
    "plugin_empirical_q99_floor": ("q99", False, True),
    "plugin_empirical_q95_floor": ("q95", False, True),
    "conditional_moment_floor": ("moment", True, True),
    ADOPTED: ("mle", True, True),
}
VARIANT_NOTES = {
    "previous_plugin_moment_floor": "Previous method: NB with the estimated mean; floor = 2001-2015 moment dispersion.",
    "plugin_mle_floor": "NB with the estimated mean; floor = 2001-2015 per-cell maximum-likelihood dispersion.",
    "plugin_empirical_q99_floor": ("Empirical null: NB with the estimated mean; floor = smallest alpha whose 4-week "
                                   "total reaches the cell's 2001-2015 99th-percentile 4-week count with "
                                   "probability >= 1%."),
    "plugin_empirical_q95_floor": "Empirical null as above, fitted at the 95th percentile.",
    "conditional_moment_floor": "Conditional beta-binomial test; floor = 2001-2015 moment dispersion.",
    ADOPTED: "Adopted: conditional beta-binomial test; floor = 2001-2015 per-cell maximum-likelihood dispersion.",
    "oracle_true_alpha_estimated_mean": ("Diagnostic: the simulation's true 2016-2019 dispersion, mean estimated "
                                         "from the reference window (plug-in NB)."),
    "oracle_true_alpha_and_mean": "Diagnostic: the simulation's true 2016-2019 dispersion and mean.",
}


def _run(C, k, window, ref, fdr, long_alpha, region_idx=None):
    return AN.detect(C[:, k - window - ref:k], window, ref, fdr, region_idx=region_idx, long_alpha=long_alpha)


def _nb_sf(obs, mean, var):
    """P(X >= obs) for the NB with this mean and variance, Poisson when var <= mean (vectorised)."""
    obs, mean, var = (np.asarray(a, float) for a in (obs, mean, var))
    out = np.ones(mean.shape)
    pois = var <= mean * (1 + 1e-9)
    m = (obs > 0) & pois
    out[m] = stats.poisson.sf(obs[m] - 1, mean[m])
    m = (obs > 0) & ~pois
    out[m] = stats.nbinom.sf(obs[m] - 1, mean[m] ** 2 / (var[m] - mean[m]), mean[m] / var[m])
    return out


def empirical_quantile_dispersion(X, q, k=4):
    """
    Empirical-null floor (evaluated, not adopted): per region, the smallest weekly NB2 alpha whose
    k-week total reaches the region's empirical q-quantile of overlapping k-week totals with
    probability >= 1 - q. The NB tail at a fixed count is not monotone in alpha, so a grid is
    scanned; when the quantile is never reached, the alpha that maximises that tail is used.
    """
    grid = np.concatenate([[0.0], np.exp(np.linspace(-7, 7, 281))])
    out = np.zeros(X.shape[0])
    for r, x in enumerate(np.asarray(X, float)):
        mu = x.mean()
        if mu <= 0:
            continue
        t = np.quantile(np.convolve(x, np.ones(k), "valid"), q, method="higher")
        tails = _nb_sf(np.full(len(grid), t), np.full(len(grid), k * mu), k * (mu + grid * mu * mu))
        ok = np.flatnonzero(tails >= 1 - q)
        out[r] = grid[ok[0]] if len(ok) else grid[np.argmax(tails)]
    return out


def _variant_p(ref, obs, k, floor, conditional, recent=True, true_mu=None):
    """p-values of one detector variant for reference counts `ref` (rows) and window totals `obs`."""
    n = ref.shape[1]
    alpha = np.maximum(AN.moment_dispersion(ref), floor) if recent else np.asarray(floor, float)
    if conditional:
        return AN.conditional_p_value(obs, ref.sum(axis=1), k, n, alpha)
    mu = ref.mean(axis=1) if true_mu is None else true_mu
    w = mu + alpha * mu ** 2
    return _nb_sf(obs, k * mu, k * w + (k * k * w / n if true_mu is None else 0.0))


def _tested(W, k, n, cells=None):
    """Regions AN.detect tests (enough reference events): (indices, reference counts, window totals)."""
    cells = np.arange(W.shape[0]) if cells is None else np.asarray(cells)
    ref, obs = W[cells, :n], W[cells, n:].sum(axis=1)
    keep = ref.sum(axis=1) >= AN.DEFAULTS["min_reference_events"]
    return cells[keep], ref[keep], obs[keep]


def _elevated(p, obs, fdr):
    return (AN.benjamini_hochberg(p) <= fdr) & (obs >= AN.MIN_OBSERVED)


def _simulate(rng, mu, var, active, width):
    """Independent weekly NB counts per active region with this mean and variance (Poisson if var <= mean)."""
    S = np.zeros((regions.N_REGIONS, width), dtype=np.int64)
    for r in active:
        if var[r] > mu[r]:
            n_ = mu[r] ** 2 / (var[r] - mu[r])
            S[r] = rng.negative_binomial(n_, n_ / (n_ + mu[r]), width)
        else:
            S[r] = rng.poisson(mu[r], width)
    return S


def _injection(C, ks, k, n, fdr, floors, specs):
    """
    Exhaustive injection. In every window ending at a column of `ks`, each tested region that the variant
    does not already flag gets ceil(f x expected) extra events, one region at a time, and BH is redone
    across all tested regions. Returns {variant: {factor: [detected, unadjusted p <= 0.05, rolling rule, trials]}}.
    """
    res = {name: {f: np.zeros(4, np.int64) for f in INJECTION_FACTORS} for name in specs}
    for kk in ks:
        cells, ref, obs = _tested(C[:, kk - k - n:kk], k, n)
        expected = ref.mean(axis=1) * k
        for name, (floor, cond, recent) in specs.items():
            fl = floors[floor][cells]
            p = _variant_p(ref, obs, k, fl, cond, recent)
            cand = np.flatnonzero(~_elevated(p, obs, fdr))
            for f in INJECTION_FACTORS:
                new_obs = obs[cand] + np.maximum(1, np.ceil(f * expected[cand])).astype(np.int64)
                p_new = _variant_p(ref[cand], new_obs, k, fl[cand], cond, recent)
                tally = res[name][f]
                for j, i in enumerate(cand):
                    pp = p.copy()
                    pp[i] = p_new[j]
                    tally[0] += AN.benjamini_hochberg(pp)[i] <= fdr and new_obs[j] >= AN.MIN_OBSERVED
                tally[1] += int((p_new <= 0.05).sum())
                tally[2] += int((new_obs >= np.maximum(2.0 * expected[cand], AN.MIN_OBSERVED)).sum())
                tally[3] += len(cand)
    return res


def _injection_rates(tally):
    det, unadj, roll, trials = (int(v) for v in tally)
    t = max(trials, 1)
    return {"trials": trials, "detection_rate": round(det / t, 4), "unadjusted_p_below_0.05_rate": round(unadj / t, 4),
            "baseline_rolling_rule_detection_rate": round(roll / t, 4)}


def evaluate_anomaly(cat, meta, window=4, ref=52, fdr=0.05, seed=7):
    rng = np.random.default_rng(seed)
    epoch = ms(EPOCH)
    mats, _, n_weeks = build_matrices(cat, meta["end_ms"])
    k_val_end, k_train_end = col_of(ms(VAL_END), epoch), col_of(ms(TRAIN_END), epoch)
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "configuration": {"window_weeks": window, "reference_weeks": ref, "fdr": fdr,
                             "min_reference_events": AN.DEFAULTS["min_reference_events"],
                             "min_observed": AN.MIN_OBSERVED,
                             "test": "conditional beta-binomial (NB2 weekly counts; the mean is removed by conditioning)",
                             "long_term_dispersion": "2001-2015 per-cell NB2 maximum likelihood"},
           "held_out_period": {"from": datetime.fromtimestamp((epoch + k_val_end * regions.WEEK_MS) / 1000,
                                                              tz=timezone.utc).strftime("%Y-%m-%d"),
                               "to": meta["end"][:10]}}

    # Long-term dispersion per region and threshold, from the training period only. The MLE table is
    # served; the other tables exist only for the comparison in section 4.
    train = {thr: C[:, 52:k_train_end] for thr, C in mats.items()}
    floors = {thr: {"moment": AN.moment_dispersion(X), "mle": np.array([AN.mle_dispersion(x) for x in X])}
              for thr, X in train.items()}
    floors[4.5].update(q99=empirical_quantile_dispersion(train[4.5], 0.99, window),
                       q95=empirical_quantile_dispersion(train[4.5], 0.95, window))
    long_alpha = {thr: f["mle"] for thr, f in floors.items()}
    reference = {"period": {"from": "2001-01-01", "to": TRAIN_END.strftime("%Y-%m-%d")},
                 "method": "NB2 maximum-likelihood dispersion of weekly counts per 10-degree cell",
                 "thresholds": {f"{thr:.1f}": [round(float(a), 5) for a in arr] for thr, arr in long_alpha.items()}}

    # 1. Behaviour on held-out real data (non-overlapping windows), both thresholds and window lengths.
    real = {}
    for thr, C in mats.items():
        for w in AN.WINDOW_WEEKS:
            ks = np.arange(k_val_end + w, n_weeks + 1, w)
            flags = tested = runs_with_flag = base_flags = prev_flags = prev_runs = 0
            per_year, per_region = {}, {}
            for k in ks:
                res, n_tested = _run(C, k, w, ref, fdr, long_alpha[thr])
                f = [r for r in res if r["status"] == "elevated"]
                flags += len(f)
                tested += n_tested
                runs_with_flag += bool(f)
                base_flags += int(AN.rolling_rule(C[:, k - w - ref:k], w, ref).sum())
                cells, R, obs = _tested(C[:, k - w - ref:k], w, ref)
                prev = _elevated(_variant_p(R, obs, w, floors[thr]["moment"][cells], False), obs, fdr)
                prev_flags += int(prev.sum())
                prev_runs += bool(prev.any())
                yr = datetime.fromtimestamp((epoch + (k - 1) * regions.WEEK_MS) / 1000, tz=timezone.utc).year
                py = per_year.setdefault(yr, [0, 0])
                py[0] += len(f)
                py[1] += n_tested
                for r in f:
                    per_region[r["region"]] = per_region.get(r["region"], 0) + 1
            real[f"M{thr:.1f}_window{w}w"] = {
                "runs": int(len(ks)), "region_tests": int(tested), "flags": int(flags),
                "flag_rate_per_region_test": round(flags / max(tested, 1), 5),
                "share_of_runs_with_any_flag": round(runs_with_flag / max(len(ks), 1), 4),
                "baseline_rolling_rule_flags": base_flags,
                "previous_method_flags": prev_flags,
                "previous_method_share_of_runs_with_any_flag": round(prev_runs / max(len(ks), 1), 4),
                "flag_rate_by_year": {str(y): round(a / max(b, 1), 5) for y, (a, b) in sorted(per_year.items())},
                "regions_ever_flagged": len(per_region),
                "most_flagged_regions": sorted(per_region.items(), key=lambda kv: -kv[1])[:8]}
    out["held_out_real_behaviour"] = real
    out["held_out_note"] = ("Real flags are not verified anomalies: there is no ground-truth label. These numbers "
                            "describe how often and where the detector fires on unseen periods.")

    # 2. Null simulation: stationary negative-binomial weekly counts with each region's VALIDATION-period
    #    (2016-2019) mean and variance, i.e. not the period the long-term dispersion was estimated on.
    #    Any flag here is a false alert by construction. Every variant sees the same simulated runs;
    #    the first 200 runs are identical to the earlier 200-run evaluation (same seed and generator).
    C = mats[4.5]
    la45 = long_alpha[4.5]
    valid = C[:, k_train_end:k_val_end]
    mu, var = valid.mean(axis=1), valid.var(axis=1)
    active = np.flatnonzero(mu * ref >= AN.DEFAULTS["min_reference_events"])
    true_alpha = np.where(mu > 0, np.maximum(var - mu, 0) / np.where(mu > 0, mu, 1.0) ** 2, 0.0)
    oracles = {"oracle_true_alpha_estimated_mean": False, "oracle_true_alpha_and_mean": True}
    names = list(VARIANTS) + list(oracles)
    any_flag = {nm: np.zeros(NULL_RUNS, bool) for nm in names}
    n_flags = dict.fromkeys(names, 0)
    runs_flagged = np.zeros(NULL_RUNS, bool)
    false_flags = base_false = tests = unadjusted = 0
    false_by_region = {}
    for s in range(NULL_RUNS):
        S = _simulate(rng, mu, var, active, ref + window)
        res, n_tested = AN.detect(S, window, ref, fdr, region_idx=active, long_alpha=la45)
        f = sum(r["status"] == "elevated" for r in res)
        for r in res:
            if r["status"] == "elevated":
                false_by_region[r["index"]] = false_by_region.get(r["index"], 0) + 1
        unadjusted += sum((r.get("p_value") or 1) <= 0.05 for r in res)
        false_flags += f
        runs_flagged[s] = bool(f)
        tests += n_tested
        base_false += int(AN.rolling_rule(S[active], window, ref).sum())
        cells, R, obs = _tested(S, window, ref, active)
        for nm in names:
            if nm in VARIANTS:
                floor, cond, recent = VARIANTS[nm]
                p = _variant_p(R, obs, window, floors[4.5][floor][cells], cond, recent)
            else:
                p = _variant_p(R, obs, window, true_alpha[cells], False, False, mu[cells] if oracles[nm] else None)
            el = _elevated(p, obs, fdr)
            any_flag[nm][s] = el.any()
            n_flags[nm] += int(el.sum())
    if not np.array_equal(any_flag[ADOPTED], runs_flagged):
        raise AssertionError("comparison helper disagrees with AN.detect for the adopted variant")
    share = float(runs_flagged.mean())
    out["null_simulation"] = {
        "simulated_runs": NULL_RUNS, "regions_per_run": int(len(active)), "region_tests": int(tests),
        "false_alert_rate_per_region_test": round(false_flags / max(tests, 1), 6),
        "share_of_runs_with_any_false_alert": round(share, 4),
        "share_of_runs_with_any_false_alert_se": round(float(np.sqrt(share * (1 - share) / NULL_RUNS)), 4),
        "share_of_runs_with_any_false_alert_first_200_runs": round(float(runs_flagged[:200].mean()), 4),
        "unadjusted_p_below_0.05_rate": round(unadjusted / max(tests, 1), 5),
        "baseline_rolling_rule_false_alert_rate": round(base_false / max(NULL_RUNS * len(active), 1), 6),
        "regions_with_most_false_alerts": [
            {"region": regions.region_id(r), "false_alerts": c, "true_alpha_2016_2019": round(float(true_alpha[r]), 3),
             "long_term_alpha_2001_2015": round(float(la45[r]), 3), "weekly_mean_2016_2019": round(float(mu[r]), 3)}
            for r, c in sorted(false_by_region.items(), key=lambda kv: -kv[1])[:6]],
        "construction": ("Independent weekly NB counts per region with mean and variance from 2016-2019 (M4.5); the "
                         "detector's long-term dispersion comes from 2001-2015. With every region null, BH at FDR "
                         "0.05 should leave at most ~5% of runs with any flag. Real catalogues are clustered, so "
                         "real false-alert rates can differ."),
    }

    # 3. Injection: add synthetic excess events to real windows, one region at a time (see _injection).
    ks_val = np.arange(k_train_end + window, k_val_end + 1, window)        # observation windows in 2016-2019
    ks_ho = np.arange(k_val_end + window, n_weeks + 1, window)             # held-out observation windows
    inj_val = _injection(C, ks_val, window, ref, fdr, floors[4.5], VARIANTS)
    inj_ho = _injection(C, ks_ho, window, ref, fdr, floors[4.5], VARIANTS)
    out["synthetic_injection"] = {f"+{f:g}x_expected": _injection_rates(t) for f, t in inj_ho[ADOPTED].items()}
    out["injection_construction"] = ("In every non-overlapping held-out window (M4.5, 4-week observation), each tested "
                                     "region not already flagged gets ceil(factor x expected) extra events in its "
                                     "last observation week, one region at a time; detection = flagged after BH "
                                     "across all regions.")

    # 4. Variant comparison on the same null runs and injection trials, plus held-out flag counts.
    #    Selection used only the null simulation and the 2016-2019 injection; held-out columns came after.
    ho_flags = {nm: [0, 0, set()] for nm in VARIANTS}
    for k in ks_ho:
        cells, R, obs = _tested(C[:, k - window - ref:k], window, ref)
        for nm, (floor, cond, recent) in VARIANTS.items():
            el = _elevated(_variant_p(R, obs, window, floors[4.5][floor][cells], cond, recent), obs, fdr)
            ho_flags[nm][0] += int(el.sum())
            ho_flags[nm][1] += bool(el.any())
            ho_flags[nm][2] |= set(cells[el].tolist())

    def detection(tallies):
        return {"trials": int(tallies[INJECTION_FACTORS[0]][3]),
                **{f"+{f:g}x_expected": round(int(t[0]) / max(int(t[3]), 1), 4) for f, t in tallies.items()}}

    comparison = {}
    for nm in names:
        row = {"description": VARIANT_NOTES[nm],
               "null_runs_with_false_alert": round(float(any_flag[nm].mean()), 4),
               "null_runs_with_false_alert_first_200_runs": round(float(any_flag[nm][:200].mean()), 4),
               "null_false_alert_rate_per_region_test": round(n_flags[nm] / max(tests, 1), 6)}
        if nm in VARIANTS:
            row.update(injection_validation_2016_2019=detection(inj_val[nm]), injection_held_out=detection(inj_ho[nm]),
                       held_out_M4_5_window4w={"flags": ho_flags[nm][0],
                                               "share_of_runs_with_any_flag": round(ho_flags[nm][1] / len(ks_ho), 4),
                                               "regions_ever_flagged": len(ho_flags[nm][2])})
        comparison[nm] = row
    out["method_comparison"] = {
        "adopted": ADOPTED, "variants": comparison,
        "floor_quartiles_null_regions": {
            k: [round(float(v), 3) for v in np.percentile(floors[4.5][k][active], [25, 50, 75])]
            for k in ("moment", "mle", "q99", "q95")},
        "note": ("All variants see the same null runs and injection trials (M4.5, 4-week window, 52-week reference). "
                 "Selection used the null simulation and the 2016-2019 injection only; held-out columns were "
                 "computed afterwards. Floor quartiles: 25/50/75th percentiles of each 2001-2015 weekly "
                 "dispersion table over the regions in the null simulation."),
    }

    # 5. Null calibration in every user-selectable setting (separate random stream, previous vs adopted).
    rng_settings = np.random.default_rng(seed + 1)
    compared = ("previous_plugin_moment_floor", "conditional_moment_floor", ADOPTED)
    by_setting = []
    for thr in (4.5, 5.0):
        valid = mats[thr][:, k_train_end:k_val_end]
        mu_t, var_t = valid.mean(axis=1), valid.var(axis=1)
        for w in AN.WINDOW_WEEKS:
            for n in AN.REFERENCE_WEEKS:
                act = np.flatnonzero(mu_t * n >= AN.DEFAULTS["min_reference_events"])
                hits = dict.fromkeys(compared, 0)
                for _ in range(SETTING_RUNS):
                    cells, R, obs = _tested(_simulate(rng_settings, mu_t, var_t, act, n + w), w, n, act)
                    for nm in compared:
                        floor, cond, recent = VARIANTS[nm]
                        p = _variant_p(R, obs, w, floors[thr][floor][cells], cond, recent)
                        hits[nm] += bool(_elevated(p, obs, fdr).any())
                by_setting.append({"threshold": thr, "window_weeks": w, "reference_weeks": n,
                                   "regions": int(len(act)), "runs": SETTING_RUNS,
                                   **{f"{nm}_runs_with_false_alert": round(v / SETTING_RUNS, 4)
                                      for nm, v in hits.items()}})
    out["null_simulation_by_setting"] = by_setting

    # 6. Missing data: real global zero-event days, and the effect of removing days from the observation window.
    t = cat["time"].to_numpy(np.int64)
    zero_days = AN.gap_days(t, meta["start_ms"], meta["end_ms"])
    lost, gap_fired, trials = [], 0, 0
    reg_all = regions.region_index(cat["latitude"].to_numpy(), cat["longitude"].to_numpy())
    mags = cat["mag"].to_numpy(float)
    for k in rng.choice(ks_ho, size=min(60, len(ks_ho)), replace=False):
        end = epoch + int(k) * regions.WEEK_MS
        start = end - window * regions.WEEK_MS
        res, _ = AN.detect(C[:, k - window - ref:k], window, ref, fdr, long_alpha=la45)
        flagged = {r["index"] for r in res if r["status"] == "elevated"}
        drop_days = rng.choice(window * 7, size=3, replace=False)
        in_win = (t >= start) & (t < end)
        dropped = in_win & np.isin((t - start) // DAY_MS, drop_days)
        W = C[:, k - window - ref:k].copy()
        sub = dropped & (mags >= 4.5 - 1e-9)
        np.subtract.at(W[:, -window:], (reg_all[sub], (t[sub] - start) // regions.WEEK_MS), 1)
        gaps = AN.gap_days(t[in_win & ~dropped], start, end)
        res2, _ = AN.detect(W, window, ref, fdr, gaps_in_window=len(gaps), long_alpha=la45)
        still = {r["index"] for r in res2 if r["status"] == "elevated"}
        if flagged:
            lost.append(len(flagged - still) / len(flagged))
        gap_fired += len(gaps) >= 3
        trials += 1
    out["missing_data"] = {
        "global_zero_event_days_2000_to_end": len(zero_days),
        "zero_event_days": [datetime.fromtimestamp(d / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for d in zero_days][:20],
        "removal_trials": trials, "days_removed_per_trial": 3,
        "gap_check_detected_all_removed_days_share": round(gap_fired / max(trials, 1), 4),
        "mean_share_of_flags_lost_when_3_days_missing": round(float(np.mean(lost)), 4) if lost else None,
        "note": ("Removing whole days lowers observed counts, so some elevated regions are missed; the gap check "
                 "reports those windows as data_incomplete instead of normal."),
    }
    return out, reference


# =============================================================================
# Module D — Omori-Utsu across historical sequences
# =============================================================================
def evaluate_aftershocks(cat, meta, min_main=7.0, fit_days=30.0, test_days=90.0, start_days=0.25, pre_days=365):
    t = cat["time"].to_numpy(np.int64)
    lat, lon, mag = cat["latitude"].to_numpy(), cat["longitude"].to_numpy(), cat["mag"].to_numpy(float)
    cand = np.flatnonzero((mag >= min_main) & (t >= meta["start_ms"] + pre_days * DAY_MS)
                          & (t <= meta["end_ms"] - test_days * DAY_MS))
    rows, excluded = [], {"larger_event_in_prior_30_days": 0, "larger_event_within_window": 0}
    for i in cand:
        R = AS.default_radius_km(mag[i])
        win = (t >= t[i] - pre_days * DAY_MS) & (t <= t[i] + test_days * DAY_MS)
        idx = np.flatnonzero(win)
        d = geo.haversine_km(lat[i], lon[i], lat[idx], lon[idx])
        near = idx[d <= R]
        other = near[near != i]
        if np.any((t[other] < t[i]) & (t[other] >= t[i] - 30 * DAY_MS) & (mag[other] >= mag[i])):
            excluded["larger_event_in_prior_30_days"] += 1
            continue
        if np.any((t[other] > t[i]) & (mag[other] > mag[i])):
            excluded["larger_event_within_window"] += 1
            continue
        td = (t[other] - t[i]) / DAY_MS
        pre_n = int(((td < 0) & (td >= -pre_days)).sum())
        mu = pre_n / pre_days
        post = td[td > 0]
        fit = AS.fit_omori(post, start_days, fit_days, mu)
        full = AS.fit_omori(post, start_days, test_days, mu)
        obs_future = int(((post > fit_days) & (post <= test_days)).sum())
        row = {"id": cat["id"].iloc[i], "mag": float(mag[i]), "radius_km": R,
               "time": datetime.fromtimestamp(t[i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
               "n_fit": fit["n_events"], "status_fit30": fit["status"], "status_full90": full["status"],
               "p_full90": full.get("parameters", {}).get("p"), "ks_p_full90": full.get("diagnostics", {}).get("ks_p_value"),
               "observed_30_90": obs_future}
        if fit.get("parameters") and fit["status"] == "fitted":
            pred = AS.expected_count(fit, fit_days, test_days)
            last_week = ((post > fit_days - 7) & (post <= fit_days)).sum() / 7.0
            row.update(pred_omori_30_90=round(pred, 2), pred_last_week_rate=round(last_week * (test_days - fit_days), 2),
                       pred_background=round(mu * (test_days - fit_days), 2))
        rows.append(row)
    fitted = [r for r in rows if "pred_omori_30_90" in r]

    def lerr(key):
        e = [abs(math.log((r["observed_30_90"] + 1) / (r[key] + 1))) for r in fitted]
        return round(float(np.median(e)), 4) if e else None

    def coverage():
        from scipy import stats
        c = [stats.poisson.ppf(0.025, r["pred_omori_30_90"]) <= r["observed_30_90"]
             <= stats.poisson.ppf(0.975, r["pred_omori_30_90"])
             for r in fitted]
        return round(float(np.mean(c)), 4) if c else None

    status = {}
    for r in rows:
        status[r["status_full90"]] = status.get(r["status_full90"], 0) + 1
    ps = [r["p_full90"] for r in rows if r["status_full90"] == "fitted"]
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection": (f"USGS M>={min_main} events {meta['start'][:10]}..{meta['end'][:10]} with >= {pre_days} days of "
                      f"prior data and {test_days:g} days of follow-up; events within the Wells-Coppersmith rupture-"
                      f"length radius, M>=4.5 (the downloaded catalogue). Fit starts {start_days} d after the mainshock."),
        "candidates": int(len(cand)), "excluded": excluded, "sequences_analysed": len(rows),
        "full_window_status_counts": status,
        "p_of_reliable_fits": ({"n": len(ps), "median": round(float(np.median(ps)), 3),
                                "q25": round(float(np.quantile(ps, 0.25)), 3),
                                "q75": round(float(np.quantile(ps, 0.75)), 3)} if ps else None),
        "out_of_sample_30_to_90_days": {
            "sequences_with_reliable_30_day_fit": len(fitted),
            "median_abs_log_ratio_omori": lerr("pred_omori_30_90"),
            "median_abs_log_ratio_last_week_rate": lerr("pred_last_week_rate"),
            "median_abs_log_ratio_background_only": lerr("pred_background"),
            "poisson_95_coverage_omori": coverage(),
            "metric_note": "|log((observed+1)/(predicted+1))|; 0 is perfect. Coverage ignores parameter uncertainty.",
        },
        "limitations": ["M4.5+ global catalogue: only the largest aftershocks are seen, and early aftershocks are "
                        "missing while the mainshock coda masks them.",
                        "Radius from rupture-length scaling is a heuristic; nearby independent activity is included.",
                        "Sequences with a larger later event were excluded, which favours simple sequences.",
                        "This evaluation is offline; the application never extrapolates fits beyond the data."],
        "sequences": rows,
    }
