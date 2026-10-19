"""
Build docs/ml_evaluation.md from the JSON results in models/.

The report contains only numbers read from those files; nothing is typed in by
hand, so the document cannot drift from the artifacts the application serves.
"""
import json
import os


def _load(model_dir, name):
    path = os.path.join(model_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _f(v, d=3):
    return "—" if v is None else f"{v:.{d}f}" if isinstance(v, float) else str(v)


def _forecast(lines, meta):
    thr = meta["threshold"]
    p = meta["periods"]
    lines += [f"### Threshold M{thr:.1f}", "",
              f"* Target: {meta['target']}",
              f"* Eligible regions: {len(meta['regions'])} ({meta['eligibility']}).",
              f"* Train {p['train']['from']} → {p['train']['to']} ({meta['samples']['train']} samples); validation "
              f"{p['validation']['from']} → {p['validation']['to']} ({meta['samples']['validation']}); test "
              f"{p['test']['from']} → {p['test']['to']} ({meta['samples']['test']}, {p['test']['method']}).",
              f"* Selected model: **{meta['selected_model']}** — {meta['selection_rule']}",
              f"* Predictive distribution: NB2 with alpha(mu) = exp({meta['dispersion_curve']['a']:.3f} "
              f"{meta['dispersion_curve']['b']:+.3f} log mu); constant-alpha MLE for reference {meta['alpha']:.3f}. "
              f"Out-of-fold validation NB log score: constant {meta['dispersion_evidence']['constant_alpha']:.4f}, "
              f"curve {meta['dispersion_evidence']['mean_dependent_curve']:.4f} (lower is better).",
              f"* Intervals: {meta['interval_method']}.", "",
              "Validation (models fit on the training period):", "",
              "| Model | MAE | RMSE | Poisson deviance | NB log score |", "|---|---:|---:|---:|---:|"]
    for name, m in meta["validation_metrics"].items():
        lines.append(f"| {name} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['poisson_deviance']:.4f} | {m['nb_log_score']:.4f} |")
    lines += ["", "**Final test period (walk-forward, never used for any choice):**", "",
              "| Model | MAE | RMSE | Poisson deviance | NB log score | MAE (zero weeks) |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, m in meta["test_metrics"].items():
        star = " (selected)" if name == meta["selected_model"] else ""
        lines.append(f"| {name}{star} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['poisson_deviance']:.4f} | "
                     f"{m['nb_log_score']:.4f} | {_f(m.get('mae_zero_weeks'), 4)} |")
    ci = meta.get("deviance_skill_ci95_week_bootstrap", {})
    lines += ["", "Test deviance skill of the selected model (1 − deviance / other model's deviance; 95% interval from "
              "resampling whole forecast weeks): " +
              ", ".join(f"{k} {v:+.3f} [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]" if k in ci else f"{k} {v:+.3f}"
                        for k, v in meta["deviance_skill_vs"].items()) + ".", ""]
    sel_m, const = meta["test_metrics"][meta["selected_model"]], meta.get("test_selected_with_constant_alpha")
    lines += ["**Predictive intervals on the test period** (selected model). Counts are integers, so no interval holds "
              "exactly the nominal probability; *model-implied* is the probability the predictive distribution assigns "
              "to its own intervals (always ≥ nominal). Calibration means empirical ≈ model-implied.", "",
              "| Interval construction | Nominal | Empirical coverage | Model-implied | Mean width (events) |",
              "|---|---:|---:|---:|---:|"]
    for lv in ("80", "95"):
        lines.append(f"| Shortest, dispersion curve (served) | {lv}% | {sel_m[f'coverage_{lv}']:.3f} | "
                     f"{sel_m[f'model_implied_coverage_{lv}']:.3f} | {sel_m[f'mean_width_{lv}']:.2f} |")
        lines.append(f"| Equal-tailed, dispersion curve | {lv}% | {sel_m[f'central_coverage_{lv}']:.3f} | "
                     f"{sel_m[f'central_model_implied_coverage_{lv}']:.3f} | {sel_m[f'central_mean_width_{lv}']:.2f} |")
        if const:
            lines.append(f"| Equal-tailed, constant alpha (previous) | {lv}% | {const[f'central_coverage_{lv}']:.3f} | "
                         f"{const[f'central_model_implied_coverage_{lv}']:.3f} | {const[f'central_mean_width_{lv}']:.2f} |")
    if const:
        lines += ["", f"NB log score of the selected model on test: dispersion curve {sel_m['nb_log_score']:.4f}, "
                  f"constant alpha {const['nb_log_score']:.4f}."]
    if sel_m.get("pit_deciles"):
        lines += ["", "Randomised PIT, share of test weeks per decile (0.100 each if calibrated): " +
                  ", ".join(f"{v:.3f}" for v in sel_m["pit_deciles"]) + "."]
    lines += [""]
    lines += [
              "Test by year (selected model vs historical mean, Poisson deviance):", "",
              "| Year | n | Selected | Historical mean | 80% coverage |", "|---|---:|---:|---:|---:|"]
    for year, d in meta["test_by_year"].items():
        s, b = d[meta["selected_model"]], d["historical_mean"]
        lines.append(f"| {year} | {s['n']} | {s['poisson_deviance']:.4f} | {b['poisson_deviance']:.4f} | "
                     f"{s['coverage_80']:.3f} |")
    lines += ["", "Reliability on test (selected model, by predicted mean):", "",
              "| Predicted | n | Mean predicted | Mean observed | 80% coverage | Model-implied | Mean width |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in meta["test_reliability"]:
        hi = "∞" if r["predicted_to"] is None else r["predicted_to"]
        lines.append(f"| {r['predicted_from']}–{hi} | {r['n']} | {r['mean_predicted']:.3f} | "
                     f"{r['mean_observed']:.3f} | {r['coverage_80']:.3f} | {_f(r.get('model_implied_coverage_80'))} | "
                     f"{_f(r.get('mean_width_80'), 2)} |")
    worse = sum(r["test"]["poisson_deviance"] > r["baseline_historical_mean"]["poisson_deviance"] for r in meta["test_by_region"])
    lines += ["", f"Regions where the selected model has higher test deviance than the historical mean: "
              f"{worse} of {len(meta['test_by_region'])}."]
    reg = meta.get("test_regional_summary")
    if reg:
        lines += [f"Regions where it beats the historical mean: {reg['regions_better_than_historical_mean']} of "
                  f"{reg['regions']}; the 52-week rolling mean: {reg['regions_better_than_rolling_mean_52w']} of "
                  f"{reg['regions']}. {reg['note']}", "",
                  "By training-period activity (equal thirds of regions):", "",
                  "| Tier | Regions | Weekly mean (train) | Zero weeks | Deviance | Historical mean | Skill | "
                  "80% coverage | Model-implied | Mean width |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for a in reg["by_activity_tier"]:
            lo, hi = a["train_weekly_mean_range"]
            lines.append(f"| {a['tier']} | {a['regions']} | {lo}–{hi} | {a['share_zero_weeks']:.3f} | "
                         f"{a['poisson_deviance']:.4f} | {a['historical_mean_deviance']:.4f} | "
                         f"{a['deviance_skill_vs_historical_mean']:+.3f} | {a['coverage_80']:.3f} | "
                         f"{a['model_implied_coverage_80']:.3f} | {a['mean_width_80']:.2f} |")
    lines += [""]


def write(model_dir, out_path):
    lines = ["# TerraWatcher ML evaluation", "",
             "_Generated by `python ml_pipeline.py report` from the JSON files in `models/`. Do not edit by hand._", ""]
    m45 = _load(model_dir, "forecast_m4.5.json")
    if m45:
        d = m45.get("data", {})
        lines += ["## Data", "", f"USGS FDSN event service, M ≥ {d.get('min_mag')}, {d.get('start')} → {d.get('end')}, "
                  f"{d.get('events')} events (downloaded {d.get('downloaded_at')}).", ""]
    lines += ["## Module B — seven-day regional count forecast", ""]
    any_fc = False
    for thr in ("4.5", "5.0"):
        meta = _load(model_dir, f"forecast_m{thr}.json")
        if meta:
            any_fc = True
            _forecast(lines, meta)
    if not any_fc:
        lines += ["Not trained yet (`python ml_pipeline.py forecast`).", ""]
    if m45:
        lines += ["Limitations:", ""] + [f"* {x}" for x in m45["limitations"]] + [""]

    seq = _load(model_dir, "sequence_eval.json")
    lines += ["## Module A — sequence clustering (synthetic labelled scenarios)", ""]
    if seq:
        p = seq["parameters"]
        lines += [f"Parameters: eps_km = {p['eps_km']}, eps_hours = {p['eps_hours']}, min_samples = {p['min_samples']}; "
                  f"seeds {seq['seeds']}. {seq['synthetic_generator']}", "",
                  "| Scenario | ARI (proposed) | ARI min | ARI naive degrees | ARI space-only | sequences true / found |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, s in seq["scenarios"].items():
            lines.append(f"| {name} | {s['proposed']['ari_mean']:.3f} | {s['proposed']['ari_min']:.3f} | "
                         f"{s['naive_degrees']['ari_mean']:.3f} | {s['space_only']['ari_mean']:.3f} | "
                         f"{s['proposed']['true_sequences']} / {s['proposed']['found_sequences']} |")
        lines += ["", "Overlapping sequences (centres 60 km apart, 30 h apart) vs eps_km: " +
                  ", ".join(f"{r['eps_km']:g} km → ARI {r['ari_mean']:.3f}"
                            for r in seq["overlapping_sequences_vs_eps_km"]) + ".",
                  "", seq["metrics_note"], ""]
        mc = seq.get("method_comparison")
        if mc:
            methods = list(mc["mean_ari_over_scenarios"])
            lines += ["**Alternative clustering methods** (10 seeds per scenario; HDBSCAN runs on the same scaled "
                      "coordinates). Mean ARI, with false merges / false splits per run where non-zero:", "",
                      "| Scenario | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
            for sc, row in mc["scenarios"].items():
                cells = []
                for m in methods:
                    v = row[m]
                    extra = []
                    if v["false_merges_mean"]:
                        extra.append(f"M{v['false_merges_mean']:g}")
                    if v["false_splits_mean"]:
                        extra.append(f"S{v['false_splits_mean']:g}")
                    if v["unassigned_member_share"] >= 0.01:
                        extra.append(f"U{v['unassigned_member_share']:.0%}")
                    cells.append(f"{v['ari_mean']:.3f}" + (f" ({', '.join(extra)})" if extra else ""))
                lines.append(f"| {sc} | " + " | ".join(cells) + " |")
            lines += ["| **mean over scenarios** | " + " | ".join(f"{mc['mean_ari_over_scenarios'][m]:.3f}"
                                                            for m in methods) + " |",
                      "| largest drop vs default | " + " | ".join(f"{mc['largest_ari_drop_vs_default'][m]:.3f}"
                                                         for m in methods) + " |", "",
                      "M = false merges, S = false splits, U = members of true sequences left unassigned.", "",
                      f"Selection rule: {mc['selection_rule']} Passing: {', '.join(mc['methods_passing_rule']) or 'none'}."]
            rc = mc.get("real_catalogue_split_check")
            if rc:
                lines += ["", f"Real-catalogue split check. {rc['note']}", "",
                          "| Method | M7.5+ mainshocks | Median share in own sequence | Mainshocks with < half "
                          "(95% CI) |", "|---|---:|---:|---:|"]
                for m, v in rc["results"].items():
                    lo, hi = v["share_of_mainshocks_below_half_wilson95"]
                    lines.append(f"| {m} | {v['mainshocks']} | {v['median_share_in_mainshock_sequence']:.3f} | "
                                 f"{v['share_of_mainshocks_below_half']:.3f} ({lo:.3f}–{hi:.3f}) |")
            lines += ["", f"**Decision: {mc['decision']}.**", ""]
        sf = seq.get("substructure_check_candidate")
        if sf:
            t, real = sf["total"], sf.get("real_m7_mainshock_sequences")
            lines += [f"**Rejected candidate: a “possibly merged sequences” check** (HDBSCAN inside each DBSCAN sequence of "
                      f"≥ 20 events; {len(sf['seeds'])} seeds per scenario). Synthetic: flagged {t['flagged_merged']} of "
                      f"{t['merged_sequences']} sequences that merge two true sequences (95% CI "
                      f"{t['hit_rate_wilson95'][0]:.3f}–{t['hit_rate_wilson95'][1]:.3f}) and {t['flagged_single']} of "
                      f"{t['single_sequences']} genuine single sequences ({t['false_flag_rate']:.3f}, 95% CI "
                      f"{t['false_flag_rate_wilson95'][0]:.3f}–{t['false_flag_rate_wilson95'][1]:.3f})." +
                      (f" Real catalogue: flagged {real['flagged']} of {real['checked']} M7+ mainshock sequences "
                       f"({real['flag_rate']:.3f}, 95% CI {real['flag_rate_wilson95'][0]:.3f}–"
                       f"{real['flag_rate_wilson95'][1]:.3f})." if real else "") + f" {sf['decision']}", ""]
    else:
        lines += ["Not evaluated yet (`python ml_pipeline.py sequences`).", ""]

    an = _load(model_dir, "anomaly_eval.json")
    lines += ["## Module C — elevated observed activity", ""]
    if an:
        c, ns, md = an["configuration"], an["null_simulation"], an["missing_data"]
        lines += [f"Configuration: {c['window_weeks']}-week observation window, {c['reference_weeks']}-week reference, "
                  f"Benjamini–Hochberg FDR {c['fdr']}, at least {c['min_reference_events']} reference events, "
                  f"at least {c['min_observed']} observed events. Held-out period {an['held_out_period']['from']} → "
                  f"{an['held_out_period']['to']}.", "",
                  "**Null simulation (every flag is false by construction):** "
                  f"{ns['simulated_runs']} runs × {ns['regions_per_run']} regions; false-alert rate per region test "
                  f"{ns['false_alert_rate_per_region_test']:.4f}; runs with at least one false alert "
                  f"{ns['share_of_runs_with_any_false_alert']:.3f}"
                  + (" (95% CI {:.3f}–{:.3f})".format(*ns["share_of_runs_with_any_false_alert_wilson95"])
                     if "share_of_runs_with_any_false_alert_wilson95" in ns else "") +
                  "; unadjusted p ≤ 0.05 rate "
                  f"{ns['unadjusted_p_below_0.05_rate']:.4f}; rolling-rule baseline false-alert rate "
                  f"{ns['baseline_rolling_rule_false_alert_rate']:.4f}. {ns['construction']}", "",
                  "**Synthetic injection:** " + an["injection_construction"], "",
                  "| Injected excess | Trials | Detected (BH) | Unadjusted p ≤ 0.05 | Rolling-rule baseline |",
                  "|---|---:|---:|---:|---:|"]
        for k, v in an["synthetic_injection"].items():
            ci = v.get("detection_rate_wilson95")
            lines.append(f"| {k} | {v['trials']} | {v['detection_rate']:.3f}" +
                         (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else "") +
                         f" | {v['unadjusted_p_below_0.05_rate']:.3f} | {v['baseline_rolling_rule_detection_rate']:.3f} |")
        strata = an.get("synthetic_injection_by_expected_count")
        if strata:
            ho = strata["held_out"]
            names = list(ho)
            bands = list(next(iter(ho[names[0]].values())))
            lines += ["", f"**Detection by expected count** (held-out windows). {strata['note']}", "",
                      "| Variant | Excess | " + " | ".join(f"expected {b}" for b in bands) + " |",
                      "|---|---|" + "---:|" * len(bands)]
            for nm in names:
                for exc in ("+2x_expected", "+4x_expected"):
                    row = ho[nm][exc]
                    lines.append(f"| {nm} | {exc} | " + " | ".join(
                        f"{row[b]['detection_rate']:.3f} (n={row[b]['trials']})" for b in bands) + " |")
        mc = an.get("method_comparison")
        if mc:
            lines += ["", f"**Method comparison** (adopted: `{mc['adopted']}`). {mc['note']}", "",
                      "| Variant | Null runs with a false alert | +2× / +4× detected, 2016–2019 | "
                      "+2× / +4× detected, held-out | Held-out flags (M4.5, 4 w) |",
                      "|---|---:|---:|---:|---:|"]
            for k, v in mc["variants"].items():
                iv, ih, hf = (v.get(x) for x in ("injection_validation_2016_2019", "injection_held_out",
                                                 "held_out_M4_5_window4w"))
                lines.append(f"| {k} | {v['null_runs_with_false_alert']:.4f} | "
                             + (f"{iv['+2x_expected']:.4f} / {iv['+4x_expected']:.4f} | "
                                f"{ih['+2x_expected']:.4f} / {ih['+4x_expected']:.4f} | {hf['flags']} |" if iv else
                                "— | — | — |"))
        by_setting = an.get("null_simulation_by_setting")
        if by_setting:
            keys = [k for k in by_setting[0] if k.endswith("_runs_with_false_alert")]
            lines += ["", "**Null calibration by setting** (share of runs with at least one false alert):", "",
                      "| Threshold | Window | Reference | Cells | " + " | ".join(k[:-len("_runs_with_false_alert")]
                                                                       for k in keys) + " |",
                      "|---|---:|---:|---:|" + "---:|" * len(keys)]
            for s in by_setting:
                def cell(k, s=s):
                    ci = s.get(k[:-len("_runs_with_false_alert")] + "_wilson95")
                    return f"{s[k]:.3f}" + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else "")
                lines.append(f"| M{s['threshold']:.1f} | {s['window_weeks']} w | {s['reference_weeks']} w | "
                             f"{s['regions']} | " + " | ".join(cell(k) for k in keys) + " |")
            lines += ["", f"{len(by_setting)} settings × {by_setting[0]['runs']} runs; 95% Wilson intervals. With every "
                      "cell null, BH at FDR 0.05 should leave at most ~5% of runs with any flag."]
        fa = an.get("false_alert_diagnosis")
        if fa:
            nulls = list(fa["nulls"])
            tests = list(fa["nulls"][nulls[0]])
            lines += ["", f"**Why false alerts remain** ({fa['runs_per_null']} runs per null, separate random stream). "
                      f"Runs with at least one false alert (95% CI):", "",
                      "| Null | " + " | ".join(tests) + " |", "|---|" + "---:|" * len(tests)]
            for nl in nulls:
                lines.append(f"| {nl} | " + " | ".join(
                    f"{fa['nulls'][nl][t]['runs_with_false_alert']:.3f} ({fa['nulls'][nl][t]['wilson95'][0]:.3f}–"
                    f"{fa['nulls'][nl][t]['wilson95'][1]:.3f})" for t in tests) + " |")
            lines += ["", f"Cells whose 2016–2019 dispersion exceeds their 2001–2015 floor: "
                      f"{fa['cells_where_2016_2019_dispersion_exceeds_floor']} of {fa['cells_simulated']}. Share of tested "
                      f"cells that cannot reach significance at all (a discrete multiplicity filter could only drop "
                      f"these): {fa['share_of_tested_cells_that_cannot_reach_significance']:.4f}. {fa['interpretation']}"]
        lines += ["", "**Held-out real data** (flags are not verified anomalies):", "",
                  "| Threshold / window | Runs | Flags | Flag rate per region test | Runs with a flag "
                  "| Rolling-rule flags | Regions ever flagged |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in an["held_out_real_behaviour"].items():
            lines.append(f"| {k} | {v['runs']} | {v['flags']} | {v['flag_rate_per_region_test']:.4f} | "
                         f"{v['share_of_runs_with_any_flag']:.3f} | {v['baseline_rolling_rule_flags']} | "
                         f"{v['regions_ever_flagged']} |")
        lines += ["", f"**Missing data:** global days with no M4.5+ event in the whole catalogue: "
                  f"{md['global_zero_event_days_2000_to_end']}. Removing {md['days_removed_per_trial']} random days from "
                  f"{md['removal_trials']} real windows: the gap check flagged the window in "
                  f"{md['gap_check_detected_all_removed_days_share']:.0%} of trials; mean share of elevated flags lost "
                  f"{_f(md['mean_share_of_flags_lost_when_3_days_missing'])}.", ""]
    else:
        lines += ["Not evaluated yet (`python ml_pipeline.py anomaly`).", ""]

    af = _load(model_dir, "aftershock_eval.json")
    lines += ["## Module D — Omori–Utsu across historical sequences", ""]
    if af:
        o = af["out_of_sample_30_to_90_days"]
        pr = af["p_of_reliable_fits"] or {}
        lines += [af["selection"], "",
                  f"Candidates {af['candidates']}, excluded {af['excluded']}, analysed {af['sequences_analysed']}. "
                  f"90-day fit status: {af['full_window_status_counts']}. Reliable fits: p median {_f(pr.get('median'))} "
                  f"(IQR {_f(pr.get('q25'))}–{_f(pr.get('q75'))}).", "",
                  f"Out of sample (fit days 0.25–30, predict the count in days 30–90; {o['sequences_with_reliable_30_day_fit']} "
                  "sequences with a reliable 30-day fit):", "",
                  "| Predictor | Median abs log ratio |", "|---|---:|",
                  f"| Omori–Utsu + background | {_f(o['median_abs_log_ratio_omori'])} |",
                  f"| Last-week rate held constant | {_f(o['median_abs_log_ratio_last_week_rate'])} |",
                  f"| Background rate only | {_f(o['median_abs_log_ratio_background_only'])} |", "",
                  f"Poisson 95% interval coverage of the Omori prediction: {_f(o['poisson_95_coverage_omori'])} "
                  f"({o['metric_note']})", ""]
        pi = o.get("predictive_intervals")
        if pi:
            lines += [f"**Predictive intervals for the day 30–90 count.** NB count dispersion d = "
                      f"{pi['count_dispersion_d']} (fitted on {pi['count_dispersion_calibration']['sequences']} mainshocks "
                      f"before {pi['count_dispersion_calibration']['mainshocks_before']}). {pi['note']} Signed median log "
                      f"ratio observed vs Omori {pi['signed_median_log_ratio_observed_vs_omori']:+.3f}; observed above "
                      f"the prediction in {pi['share_observed_above_omori']:.1%} of sequences.", "",
                      "| Sequences | Method | n | 80% coverage (95% CI) | 95% coverage (95% CI) | Median 95% width | "
                      "Median 95% upper/lower |", "|---|---|---:|---:|---:|---:|---:|"]
            groups = [("all", pi["all"])] + [(k, v) for k, v in pi.items() if k.startswith("mainshocks_")]
            groups += [(f"fit size {k}", v) for k, v in pi["by_fit_size"].items()]
            groups += [(k, v) for k, v in pi["by_magnitude"].items()]
            for gname, g in groups:
                for m in pi["methods"]:
                    v = g[m]
                    c8, c9 = v["80"], v["95"]
                    lines.append(f"| {gname} | {m} | {v['n']} | {_f(c8['coverage'])} ({_f(c8['coverage_wilson95'][0])}–"
                                 f"{_f(c8['coverage_wilson95'][1])}) | {_f(c9['coverage'])} ("
                                 f"{_f(c9['coverage_wilson95'][0])}–{_f(c9['coverage_wilson95'][1])}) | "
                                 f"{_f(c9['median_width_events'], 1)} | {_f(c9['median_ratio_upper_to_lower'], 1)} |")
            lines += ["", "Methods: " + " ".join(f"`{k}`: {v}" for k, v in pi["methods"].items()), ""]
        lines += ["Limitations:", ""] + [f"* {x}" for x in af["limitations"]] + [""]
    else:
        lines += ["Not evaluated yet (`python ml_pipeline.py aftershock`).", ""]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
