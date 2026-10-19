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
PRED_DRAWS = 4000


def wilson_interval(successes, trials, z=1.96):
    """Wilson score 95% interval for a binomial proportion; [None, None] without trials."""
    if trials <= 0:
        return [None, None]
    p = successes / trials
    d = 1 + z * z / trials
    c = (p + z * z / (2 * trials)) / d
    h = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return [round(c - h, 4), round(c + h, 4)]


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
    elif name.startswith("overlap_"):                         # overlap_<km>km_<hours>h
        km, hours = (float(v[:-2] if v.endswith("km") else v[:-1]) for v in name.split("_")[1:3])
        parts.append((*_cluster(rng, 10, 125, 100, 25), 0))
        la, lo = destination(10, 125, 90, km)
        parts.append((*_cluster(rng, float(la), float(lo), 100 + hours, 25), 1))
    elif name == "density_contrast":
        parts.append((*_cluster(rng, -20, -70, 100, 80, sigma_km=20), 0))
        la, lo = destination(-20, -70, 0, 150)
        parts.append((*_cluster(rng, float(la), float(lo), 140, 8, sigma_km=10, sigma_h=12), 1))
        parts.append((*_noise(rng, 60), -1))
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
EXTRA_SCENARIOS = ["overlap_30km_30h", "overlap_100km_30h", "overlap_150km_30h", "overlap_60km_120h", "density_contrast"]
# Clustering methods compared on identical synthetic data: (algorithm, eps_km or scale km, eps_hours or scale h,
# min_samples or min_cluster_size). HDBSCAN runs on the same scaled coordinates as DBSCAN.
CLUSTER_METHODS = {"dbscan_50km_72h (default)": ("dbscan", 50, 72, 4), "dbscan_25km_72h": ("dbscan", 25, 72, 4),
                   "dbscan_15km_72h": ("dbscan", 15, 72, 4), "dbscan_50km_24h": ("dbscan", 50, 24, 4),
                   "dbscan_25km_36h": ("dbscan", 25, 36, 4), "hdbscan_mcs5": ("hdbscan", 50, 72, 5),
                   "hdbscan_mcs8": ("hdbscan", 50, 72, 8)}
CLUSTER_SELECTION_RULE = ("Replace the default only if a method has a higher mean ARI over all scenarios AND no "
                          "scenario's mean ARI falls by more than 0.02 (fixed before the comparison was run).")


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


def merges_splits(truth, pred):
    """
    False merges: true sequences whose majority predicted sequence is also another true sequence's majority.
    False splits: true sequences with >= 20% of members in each of two or more predicted sequences.
    Unassigned share: members of true sequences labelled noise.
    """
    truth, pred = np.asarray(truth), np.asarray(pred)
    major, splits, unassigned = [], 0, 0
    for s in sorted(set(truth.tolist()) - {-1}):
        p = pred[truth == s]
        unassigned += int((p < 0).sum())
        lab = p[p >= 0]
        if len(lab):
            vals, cnt = np.unique(lab, return_counts=True)
            major.append(int(vals[np.argmax(cnt)]))
            splits += int(np.sum(cnt >= 0.2 * len(p)) > 1)
    return len(major) - len(set(major)), splits, unassigned / max(int((truth >= 0).sum()), 1)


def _cluster_predict(method, df):
    algo, a, b, c = CLUSTER_METHODS[method]
    if algo == "dbscan":
        return sequences.cluster_labels(df["latitude"].to_numpy(), df["longitude"].to_numpy(), df["time"].to_numpy(),
                                        a, b, c)[0]
    from sklearn.cluster import HDBSCAN
    X = sequences.scaled_coordinates(df["latitude"], df["longitude"], df["time"], a, b)
    return HDBSCAN(min_cluster_size=c, min_samples=4).fit(X).labels_


def compare_cluster_methods(seeds):
    from analytics.catalog import events_to_frame
    out = {}
    for name in SCENARIOS + EXTRA_SCENARIOS:
        frames = []
        for seed in seeds:
            events, truth_map = scenario(name, seed)
            df, _ = events_to_frame(events)
            frames.append((df, [truth_map[i] for i in df["id"]]))
        out[name] = {}
        for method in CLUSTER_METHODS:
            ari, mg, sp, un = [], [], [], []
            for df, truth in frames:
                pred = _cluster_predict(method, df)
                ari.append(adjusted_rand_score(truth, pred))
                m, s, u = merges_splits(truth, pred)
                mg.append(m)
                sp.append(s)
                un.append(u)
            out[name][method] = {"ari_mean": round(float(np.mean(ari)), 4), "ari_min": round(float(np.min(ari)), 4),
                                 "false_merges_mean": round(float(np.mean(mg)), 3),
                                 "false_splits_mean": round(float(np.mean(sp)), 3),
                                 "unassigned_member_share": round(float(np.mean(un)), 4)}
    default = next(iter(CLUSTER_METHODS))
    overall = {m: round(float(np.mean([out[s][m]["ari_mean"] for s in out])), 4) for m in CLUSTER_METHODS}
    worst_drop = {m: round(float(max(out[s][default]["ari_mean"] - out[s][m]["ari_mean"] for s in out)), 4)
                  for m in CLUSTER_METHODS}
    passing = sorted((m for m in CLUSTER_METHODS
                      if m != default and overall[m] > overall[default] and worst_drop[m] <= 0.02),
                     key=lambda m: -overall[m])
    return {"scenarios": out, "mean_ari_over_scenarios": overall, "largest_ari_drop_vs_default": worst_drop,
            "selection_rule": CLUSTER_SELECTION_RULE, "methods_passing_rule": passing,
            "decision": f"replace default with {passing[0]}" if passing else "keep the default DBSCAN parameters"}


SUBSTRUCTURE_MIN_EVENTS = 20


def substructure_groups(X):
    """
    Candidate merged-sequence check (evaluated, NOT adopted): HDBSCAN inside one DBSCAN sequence (rows of X: its
    scaled coordinates); sizes of subgroups holding >= 20% of the members. Two or more would suggest a merge.
    """
    from sklearn.cluster import HDBSCAN
    n = len(X)
    if n < SUBSTRUCTURE_MIN_EVENTS:
        return []
    sub = HDBSCAN(min_cluster_size=max(5, n // 10), min_samples=4).fit(X).labels_
    return sorted((int((sub == s).sum()) for s in set(sub.tolist()) - {-1} if (sub == s).sum() >= 0.2 * n), reverse=True)


def substructure_flag_rates(seeds, cat=None, meta=None):
    """
    How often the candidate check fires on DBSCAN sequences that truly merge 2+ synthetic sequences, on genuine single
    synthetic sequences, and on real M7+ mainshock sequences (mostly single aftershock sequences, so mostly false flags).
    """
    from analytics.catalog import events_to_frame
    p = sequences.DEFAULTS
    rows = {}
    for name in SCENARIOS + EXTRA_SCENARIOS:
        hit = merged = false = single = 0
        for seed in seeds:
            events, truth_map = scenario(name, seed)
            df, _ = events_to_frame(events)
            lab, _ = sequences.cluster_labels(df["latitude"].to_numpy(), df["longitude"].to_numpy(),
                                              df["time"].to_numpy(), p["eps_km"], p["eps_hours"], p["min_samples"])
            X = sequences.scaled_coordinates(df["latitude"], df["longitude"], df["time"], p["eps_km"], p["eps_hours"])
            truth = np.array([truth_map[i] for i in df["id"]])
            for c in set(lab.tolist()) - {-1}:
                m = lab == c
                if m.sum() < SUBSTRUCTURE_MIN_EVENTS:
                    continue
                flag = len(substructure_groups(X[m])) >= 2
                if len(set(truth[m].tolist()) - {-1}) >= 2:
                    merged += 1
                    hit += flag
                else:
                    single += 1
                    false += flag
        rows[name] = {"merged_sequences": merged, "flagged_merged": hit, "single_sequences": single, "flagged_single": false}
    tot = {k: sum(r[k] for r in rows.values()) for k in ("merged_sequences", "flagged_merged", "single_sequences",
                                                         "flagged_single")}
    out = {"by_scenario": rows, "seeds": list(seeds), "total": {**tot,
           "hit_rate": round(tot["flagged_merged"] / max(tot["merged_sequences"], 1), 4),
           "hit_rate_wilson95": wilson_interval(tot["flagged_merged"], tot["merged_sequences"]),
           "false_flag_rate": round(tot["flagged_single"] / max(tot["single_sequences"], 1), 4),
           "false_flag_rate_wilson95": wilson_interval(tot["flagged_single"], tot["single_sequences"])}}
    if cat is not None:
        t = cat["time"].to_numpy(np.int64)
        la, lo, mg = cat["latitude"].to_numpy(), cat["longitude"].to_numpy(), cat["mag"].to_numpy(float)
        checked = flagged = 0
        for i in np.flatnonzero((mg >= 7.0) & (t <= meta["end_ms"] - 30 * DAY_MS)):
            w = np.flatnonzero((t >= t[i] - 3 * DAY_MS) & (t <= t[i] + 30 * DAY_MS))
            w = w[geo.haversine_km(la[i], lo[i], la[w], lo[w]) <= AS.default_radius_km(mg[i])]
            lab, _ = sequences.cluster_labels(la[w], lo[w], t[w], p["eps_km"], p["eps_hours"], p["min_samples"])
            own = lab[w == i][0]
            mem = w[lab == own] if own >= 0 else w[:0]
            if len(mem) < SUBSTRUCTURE_MIN_EVENTS:
                continue
            checked += 1
            flagged += len(substructure_groups(sequences.scaled_coordinates(la[mem], lo[mem], t[mem], p["eps_km"],
                                                                            p["eps_hours"]))) >= 2
        out["real_m7_mainshock_sequences"] = {"checked": checked, "flagged": flagged,
                                              "flag_rate": round(flagged / max(checked, 1), 4),
                                              "flag_rate_wilson95": wilson_interval(flagged, checked)}
    out["decision"] = ("Not adopted. It was dropped after a live check flagged a single M7.7 aftershock sequence; the "
                       "real-catalogue rate above measures how often it fires on mainshock sequences, which are mostly "
                       "single aftershock sequences with internal structure along the rupture, not merges.")
    return out


def real_fragmentation(cat, meta, methods, min_main=7.5, days=30):
    """
    Real-catalogue split check (no ground truth): for each M>=min_main mainshock, cluster the M4.5+ events within its
    rupture-length radius during the following `days` days and measure the share that falls in the mainshock's own
    sequence. Most of these events are aftershocks, so a low share indicates a false split of one sequence.
    """
    t = cat["time"].to_numpy(np.int64)
    la, lo, mg = cat["latitude"].to_numpy(), cat["longitude"].to_numpy(), cat["mag"].to_numpy(float)
    mains = np.flatnonzero((mg >= min_main) & (t <= meta["end_ms"] - days * DAY_MS))
    out = {}
    for method in methods:
        _, e_km, e_h, ms_ = CLUSTER_METHODS[method]
        shares = []
        for i in mains:
            w = np.flatnonzero((t >= t[i]) & (t <= t[i] + days * DAY_MS))
            w = w[geo.haversine_km(la[i], lo[i], la[w], lo[w]) <= AS.default_radius_km(mg[i])]
            if len(w) < 20:
                continue
            lab, _ = sequences.cluster_labels(la[w], lo[w], t[w], e_km, e_h, ms_)
            own = lab[w == i][0]
            shares.append(float(np.mean(lab == own)) if own >= 0 else 0.0)
        s = np.array(shares)
        out[method] = {"mainshocks": len(s), "median_share_in_mainshock_sequence": round(float(np.median(s)), 4),
                       "share_of_mainshocks_below_half": round(float(np.mean(s < 0.5)), 4),
                       "share_of_mainshocks_below_half_wilson95": wilson_interval(int(np.sum(s < 0.5)), len(s))}
    return out


def evaluate_sequences(seeds=(0, 1, 2, 3, 4), cat=None, meta=None):
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
    comparison = compare_cluster_methods(range(10))
    default = next(iter(CLUSTER_METHODS))
    if cat is not None and comparison["methods_passing_rule"]:
        real = real_fragmentation(cat, meta, [default] + comparison["methods_passing_rule"])
        worse = [m for m in comparison["methods_passing_rule"]
                 if real[m]["share_of_mainshocks_below_half"] > real[default]["share_of_mainshocks_below_half"]]
        comparison["real_catalogue_split_check"] = {
            "results": real, "methods_splitting_real_sequences_more_often": worse,
            "note": ("Added after the synthetic comparison: the synthetic clusters are compact (15 km spread) and contain "
                     "no long ruptures, so a smaller eps was also checked on real M7.5+ aftershock zones. A method that "
                     "leaves more mainshocks with under half of their 30-day zone events in their own sequence "
                     "splits real sequences more often.")}
        keep = [m for m in comparison["methods_passing_rule"] if m not in worse]
        comparison["decision"] = (f"replace default with {keep[0]}" if keep else
                                  "keep the default DBSCAN parameters: every method that passed the synthetic rule "
                                  "splits real aftershock sequences more often")
    elif comparison["methods_passing_rule"]:
        comparison["decision"] += " (real-catalogue split check not run: no catalogue)"
    return {
        "overlapping_sequences_vs_eps_km": overlap,
        "method_comparison": comparison,
        "substructure_check_candidate": substructure_flag_rates(range(20), cat, meta),
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


EXPECTED_STRATA = (0.0, 1.0, 2.0, 5.0, 10.0, np.inf)     # expected events in the observation window


def _injection(C, ks, k, n, fdr, floors, specs):
    """
    Exhaustive injection. In every window ending at a column of `ks`, each tested region that the variant
    does not already flag gets ceil(f x expected) extra events, one region at a time, and BH is redone
    across all tested regions. Returns {variant: {factor: [detected, unadjusted p <= 0.05, rolling rule, trials]}}
    and {variant: {factor: (strata, 2) array of [detected, trials] by expected count}}.
    """
    res = {name: {f: np.zeros(4, np.int64) for f in INJECTION_FACTORS} for name in specs}
    strata = {name: {f: np.zeros((len(EXPECTED_STRATA) - 1, 2), np.int64) for f in INJECTION_FACTORS} for name in specs}
    for kk in ks:
        cells, ref, obs = _tested(C[:, kk - k - n:kk], k, n)
        expected = ref.mean(axis=1) * k
        band = np.searchsorted(EXPECTED_STRATA, expected, side="right") - 1
        for name, (floor, cond, recent) in specs.items():
            fl = floors[floor][cells]
            p = _variant_p(ref, obs, k, fl, cond, recent)
            cand = np.flatnonzero(~_elevated(p, obs, fdr))
            for f in INJECTION_FACTORS:
                new_obs = obs[cand] + np.maximum(1, np.ceil(f * expected[cand])).astype(np.int64)
                p_new = _variant_p(ref[cand], new_obs, k, fl[cand], cond, recent)
                tally, st = res[name][f], strata[name][f]
                for j, i in enumerate(cand):
                    pp = p.copy()
                    pp[i] = p_new[j]
                    hit = AN.benjamini_hochberg(pp)[i] <= fdr and new_obs[j] >= AN.MIN_OBSERVED
                    tally[0] += hit
                    st[band[i]] += (hit, 1)
                tally[1] += int((p_new <= 0.05).sum())
                tally[2] += int((new_obs >= np.maximum(2.0 * expected[cand], AN.MIN_OBSERVED)).sum())
                tally[3] += len(cand)
    return res, strata


def _strata_rates(st):
    out = {}
    for b, (det, trials) in enumerate(st):
        lo, hi = EXPECTED_STRATA[b], EXPECTED_STRATA[b + 1]
        out[f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}"] = {
            "trials": int(trials), "detection_rate": round(int(det) / max(int(trials), 1), 4),
            "wilson95": wilson_interval(int(det), int(trials))}
    return out


def _injection_rates(tally):
    det, unadj, roll, trials = (int(v) for v in tally)
    t = max(trials, 1)
    return {"trials": trials, "detection_rate": round(det / t, 4), "detection_rate_wilson95": wilson_interval(det, trials),
            "unadjusted_p_below_0.05_rate": round(unadj / t, 4),
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
        "share_of_runs_with_any_false_alert_wilson95": wilson_interval(int(runs_flagged.sum()), NULL_RUNS),
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
    inj_val, strata_val = _injection(C, ks_val, window, ref, fdr, floors[4.5], VARIANTS)
    inj_ho, strata_ho = _injection(C, ks_ho, window, ref, fdr, floors[4.5], VARIANTS)
    out["synthetic_injection"] = {f"+{f:g}x_expected": _injection_rates(t) for f, t in inj_ho[ADOPTED].items()}
    out["synthetic_injection_by_expected_count"] = {
        "note": ("Detection by the region's expected count in the 4-week window. A +4x excess where 1 event is "
                 "expected is 4 extra events, which no calibrated test can separate from chance among ~140 cells."),
        "held_out": {nm: {f"+{f:g}x_expected": _strata_rates(st) for f, st in strata_ho[nm].items()}
                     for nm in (ADOPTED, "conditional_moment_floor")},
        "validation_2016_2019": {nm: {f"+{f:g}x_expected": _strata_rates(st) for f, st in strata_val[nm].items()}
                                 for nm in (ADOPTED, "conditional_moment_floor")}}
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
                **{f"+{f:g}x_expected": round(int(t[0]) / max(int(t[3]), 1), 4) for f, t in tallies.items()},
                **{f"+{f:g}x_expected_wilson95": wilson_interval(int(t[0]), int(t[3])) for f, t in tallies.items()}}

    comparison = {}
    for nm in names:
        row = {"description": VARIANT_NOTES[nm],
               "null_runs_with_false_alert": round(float(any_flag[nm].mean()), 4),
               "null_runs_with_false_alert_wilson95": wilson_interval(int(any_flag[nm].sum()), NULL_RUNS),
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
                                      for nm, v in hits.items()},
                                   **{f"{nm}_wilson95": wilson_interval(v, SETTING_RUNS) for nm, v in hits.items()}})
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
    out["false_alert_diagnosis"] = _diagnose_false_alerts(C[:, k_train_end:k_val_end], mu, var, active, true_alpha,
                                                          floors[4.5], window, ref, fdr, seed + 2)
    return out, reference


DIAG_RUNS = 2000


def _diagnose_false_alerts(valid, mu, var, active, true_alpha, floors, window, ref, fdr, seed):
    """
    Why does the adopted detector exceed 5% runs-with-a-false-alert in the NB null? Three nulls on a separate random
    stream, each with the adopted test, the moment floor and the oracle test that knows the simulated dispersion:
      nb_2016_2019                  - the null of section 2 (NB, 2016-2019 mean and variance per cell);
      nb_dispersion_capped_at_floor - the same, but no cell is more overdispersed than its 2001-2015 floor;
      empirical_weekly_bootstrap    - each cell's real 2016-2019 weekly counts resampled independently (real tails,
                                      no NB assumption, no temporal clustering).
    Also: the share of tested cells whose smallest attainable p-value already exceeds the FDR level (a discrete
    multiplicity filter could only drop these).
    """
    rng = np.random.default_rng(seed)
    capped_var = np.minimum(var, mu + floors["mle"] * mu ** 2)
    tests = {ADOPTED: lambda R, obs, cells: _variant_p(R, obs, window, floors["mle"][cells], True, True),
             "conditional_moment_floor": lambda R, obs, cells: _variant_p(R, obs, window, floors["moment"][cells], True, True),
             "oracle_conditional_true_alpha": lambda R, obs, cells: _variant_p(R, obs, window, true_alpha[cells], True, False)}
    out, unattainable, tested = {}, 0, 0
    for null in ("nb_2016_2019", "nb_dispersion_capped_at_floor", "empirical_weekly_bootstrap_2016_2019"):
        hits = dict.fromkeys(tests, 0)
        for _ in range(DIAG_RUNS):
            if null == "empirical_weekly_bootstrap_2016_2019":
                S = np.zeros((regions.N_REGIONS, ref + window), np.int64)
                S[active] = valid[active][np.arange(len(active))[:, None],
                                          rng.integers(0, valid.shape[1], (len(active), ref + window))]
            else:
                S = _simulate(rng, mu, capped_var if null == "nb_dispersion_capped_at_floor" else var, active,
                              ref + window)
            cells, R, obs = _tested(S, window, ref, active)
            for nm, fn in tests.items():
                hits[nm] += bool(_elevated(fn(R, obs, cells), obs, fdr).any())
            if null == "nb_2016_2019":
                N = R.sum(axis=1) + obs
                unattainable += int((AN.conditional_p_value(N, np.zeros_like(N), window, ref, floors["mle"][cells])
                                     > fdr).sum())
                tested += len(cells)
        out[null] = {nm: {"runs_with_false_alert": round(h / DIAG_RUNS, 4),
                          "wilson95": wilson_interval(h, DIAG_RUNS)} for nm, h in hits.items()}
    over = active[true_alpha[active] > floors["mle"][active]]
    return {
        "runs_per_null": DIAG_RUNS, "nulls": out,
        "cells_where_2016_2019_dispersion_exceeds_floor": int(len(over)), "cells_simulated": int(len(active)),
        "share_of_tested_cells_that_cannot_reach_significance": round(unattainable / max(tested, 1), 5),
        "interpretation": (
            "If the capped null and the oracle stay near or below 5% while the uncapped null does not, the test itself "
            "is calibrated and the excess comes from cells whose 2016-2019 dispersion exceeded the 2001-2015 floor "
            "(a large sequence in a previously quieter cell). The empirical bootstrap places real sequence weeks at "
            "random positions; a burst landing in the observation window is what the detector is meant to flag, so "
            "its rate measures sensitivity to real bursts, not a false-alarm rate of a stationary process."),
    }


# =============================================================================
# Module D — Omori-Utsu across historical sequences
# =============================================================================
INTERVAL_METHODS = ("poisson_plugin", "parameter_uncertainty", "parameter_uncertainty_nb")


def calibrate_count_dispersion(observed, rate_draws_list):
    """
    NB2 dispersion d maximising sum_i log mean_s NB(obs_i | lam_is, d): the likelihood of each observed count under
    the predictive mixture over parameter draws. 0 when a Poisson stage fits at least as well.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln, logsumexp
    pairs = [(float(y), np.asarray(lam, float)) for y, lam in zip(observed, rate_draws_list, strict=True)
             if lam is not None and len(lam)]
    if not pairs:
        return 0.0

    def loglik(d):
        total = 0.0
        for y, lam in pairs:
            lam = np.maximum(lam, 1e-9)
            if d <= 0:
                lp = y * np.log(lam) - lam - gammaln(y + 1)
            else:
                r = 1.0 / d
                lp = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + lam)) + y * np.log(lam / (r + lam))
            total += logsumexp(lp) - math.log(len(lam))
        return total

    res = minimize_scalar(lambda ld: -loglik(math.exp(ld)), bounds=(-9.0, 3.0), method="bounded")
    return float(math.exp(res.x)) if -res.fun > loglik(0.0) else 0.0


def evaluate_aftershocks(cat, meta, min_main=7.0, fit_days=30.0, test_days=90.0, start_days=0.25, pre_days=365):
    t = cat["time"].to_numpy(np.int64)
    lat, lon, mag = cat["latitude"].to_numpy(), cat["longitude"].to_numpy(), cat["mag"].to_numpy(float)
    cand = np.flatnonzero((mag >= min_main) & (t >= meta["start_ms"] + pre_days * DAY_MS)
                          & (t <= meta["end_ms"] - test_days * DAY_MS))
    rows, excluded = [], {"larger_event_in_prior_30_days": 0, "larger_event_within_window": 0}
    fits = {}
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
                       pred_background=round(mu * (test_days - fit_days), 2),
                       n_fit_30=fit["n_events"], p_fit30=fit["parameters"]["p"], se_p_fit30=fit["standard_errors"]["p"])
            for name, unc in (("poisson_plugin", False), ("parameter_uncertainty", True)):
                pc = AS.predictive_count(fit, fit_days, test_days, levels=(0.8, 0.95), seed=int(i), n_draws=PRED_DRAWS,
                                         include_parameter_uncertainty=unc)
                if pc:
                    row[f"interval_{name}"] = {"80": pc["interval_80"], "95": pc["interval_95"]}
            fits[row["id"]] = (fit, int(i))
        rows.append(row)
    fitted = [r for r in rows if "pred_omori_30_90" in r]

    # Count dispersion for the negative-binomial stage, calibrated ONLY on mainshocks before split_year
    # and applied unchanged to later mainshocks (chronological separation).
    split_year = 2013
    calib = [r for r in fitted if int(r["time"][:4]) < split_year and fits[r["id"]][0].get("covariance")]
    disp = calibrate_count_dispersion([r["observed_30_90"] for r in calib],
                                      [AS.rate_draws(fits[r["id"]][0], fit_days, test_days, 1000, fits[r["id"]][1])
                                       for r in calib])
    for r in fitted:
        fit, seed = fits[r["id"]]
        pc = AS.predictive_count(fit, fit_days, test_days, levels=(0.8, 0.95), seed=seed, n_draws=PRED_DRAWS,
                                 dispersion=disp)
        if pc:
            r["interval_parameter_uncertainty_nb"] = {"80": pc["interval_80"], "95": pc["interval_95"]}

    def lerr(key):
        e = [abs(math.log((r["observed_30_90"] + 1) / (r[key] + 1))) for r in fitted]
        return round(float(np.median(e)), 4) if e else None

    def coverage():
        from scipy import stats
        c = [stats.poisson.ppf(0.025, r["pred_omori_30_90"]) <= r["observed_30_90"]
             <= stats.poisson.ppf(0.975, r["pred_omori_30_90"])
             for r in fitted]
        return round(float(np.mean(c)), 4) if c else None

    def interval_summary(subset):
        """Empirical coverage (Wilson 95% CI) and width of each predictive interval method on `subset`."""
        out = {"sequences": len(subset)}
        for name in INTERVAL_METHODS:
            rs = [r for r in subset if f"interval_{name}" in r]
            m = {}
            for lv in ("80", "95"):
                hit = [r[f"interval_{name}"][lv][0] <= r["observed_30_90"] <= r[f"interval_{name}"][lv][1] for r in rs]
                widths = [r[f"interval_{name}"][lv][1] - r[f"interval_{name}"][lv][0] for r in rs]
                rel = [(r[f"interval_{name}"][lv][1] + 1) / (r[f"interval_{name}"][lv][0] + 1) for r in rs]
                m[lv] = {"coverage": round(float(np.mean(hit)), 4) if hit else None,
                         "coverage_wilson95": wilson_interval(int(np.sum(hit)), len(hit)),
                         "median_width_events": float(np.median(widths)) if widths else None,
                         "median_ratio_upper_to_lower": round(float(np.median(rel)), 2) if rel else None}
            out[name] = {"n": len(rs), **m}
        return out

    by_size = {"20-49 fit events": [r for r in fitted if r["n_fit_30"] < 50],
               "50-149 fit events": [r for r in fitted if 50 <= r["n_fit_30"] < 150],
               ">=150 fit events": [r for r in fitted if r["n_fit_30"] >= 150]}
    by_mag = {"M7.0-7.4": [r for r in fitted if r["mag"] < 7.5], "M7.5+": [r for r in fitted if r["mag"] >= 7.5]}
    intervals = {
        "all": interval_summary(fitted),
        f"mainshocks_before_{split_year}": interval_summary([r for r in fitted if int(r["time"][:4]) < split_year]),
        f"mainshocks_{split_year}_and_later": interval_summary([r for r in fitted if int(r["time"][:4]) >= split_year]),
        "by_fit_size": {k: interval_summary(v) for k, v in by_size.items()},
        "by_magnitude": {k: interval_summary(v) for k, v in by_mag.items()},
        "signed_median_log_ratio_observed_vs_omori": (
            round(float(np.median([math.log((r["observed_30_90"] + 1) / (r["pred_omori_30_90"] + 1)) for r in fitted])), 4)
            if fitted else None),
        "share_observed_above_omori": (round(float(np.mean([r["observed_30_90"] > r["pred_omori_30_90"] for r in fitted])), 4)
                                       if fitted else None),
        "methods": {
            "poisson_plugin": "Poisson quantiles around the fitted expected count (parameters treated as exact).",
            "parameter_uncertainty": (f"Parametric simulation: {PRED_DRAWS} draws of (log K, [log c], p) from the MLE's "
                                      "normal approximation (inverse Hessian), Poisson count per draw; nothing is tuned "
                                      "on the evaluated sequences."),
            "parameter_uncertainty_nb": ("As parameter_uncertainty, but the count per draw is negative binomial with "
                                         "Var = lam + d * lam^2; d is fitted by maximum likelihood (mixture over the "
                                         f"parameter draws) on mainshocks before {split_year} only."),
        },
        "count_dispersion_d": round(float(disp), 4),
        "count_dispersion_calibration": {"mainshocks_before": split_year, "sequences": len(calib)},
        "note": (f"poisson_plugin and parameter_uncertainty use only each sequence's own 0.25-30 day fit, so every "
                 f"sequence is out of sample. parameter_uncertainty_nb borrows d from mainshocks before {split_year}: "
                 f"only the mainshocks_{split_year}_and_later rows are out of sample for it (the earlier rows are "
                 "in-sample and shown for comparison)."),
    }

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
            "predictive_intervals": intervals,
        },
        "limitations": ["M4.5+ global catalogue: only the largest aftershocks are seen, and early aftershocks are "
                        "missing while the mainshock coda masks them.",
                        "Radius from rupture-length scaling is a heuristic; nearby independent activity is included.",
                        "Sequences with a larger later event were excluded, which favours simple sequences.",
                        "This evaluation is offline; the application never extrapolates fits beyond the data."],
        "sequences": rows,
    }
