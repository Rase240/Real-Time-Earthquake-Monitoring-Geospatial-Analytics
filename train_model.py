"""
Offline training + evaluation pipeline for TerraWatcher's conditional severity model.

Question the model answers
--------------------------
    Given that an earthquake of magnitude >= CATALOG_MIN_MAG (4.5) is recorded at
    (latitude, longitude, depth), with a given amount of recent local activity and
    a given distance to the nearest plate boundary, what is the probability that
    its magnitude is >= SIGNIFICANT_MAG (5.0)?

This is a *conditional* classification of recorded events. It does NOT predict
whether, when or where an earthquake will happen.

Pipeline
--------
  1. Download the USGS catalogue (M >= 4.5) for the last N months, in chunks.
  2. Compute features with features.py (density uses only the 30 days *before*
     each event; nothing from the future).
  3. Split chronologically: oldest 70% train, next 15% validation, newest 15% test.
  4. Fit StandardScaler on train only; train the MLP with early stopping on validation.
  5. Evaluate on the untouched test period against simple baselines.
  6. Write earthquake_model_v1.pth, scaler.pkl, model_meta.json and
     docs/model_evaluation.md.

Usage
-----
  python train_model.py                 # download fresh data and train
  python train_model.py --offline       # reuse data/training_catalog.csv
  python train_model.py --months 18
"""
import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

import config
import features as F
import usgs_client
from ml_model import EarthquakeNN

log = logging.getLogger("terrawatcher.train")

CATALOG_CSV = os.path.join(config.BASE_DIR, "data", "training_catalog.csv")
EVAL_REPORT = os.path.join(config.BASE_DIR, "docs", "model_evaluation.md")
CHUNK_DAYS = 30
GRID_DEG = 5.0          # cell size for the regional-rate baseline
GRID_PRIOR = 20.0       # pseudo-counts shrinking sparse cells towards the base rate


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def download_catalog(months, min_mag):
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30 * months)
    events, cursor = {}, start
    log.info("Downloading USGS catalogue M>=%.1f from %s to %s", min_mag, start.date(), end.date())
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        batch, stats = usgs_client.fetch_events(cursor, chunk_end, min_mag=min_mag, limit=20000,
                                                timeout=90, retries=3)
        if len(batch) >= 20000:
            raise RuntimeError("A chunk hit the USGS 20000-event limit; reduce CHUNK_DAYS")
        for e in batch:
            events[e["id"]] = e      # chunks are half-open, but de-duplicate anyway
        log.info("  %s -> %s: %d events (%d dropped)", cursor.date(), chunk_end.date(), len(batch),
                 stats["dropped_invalid"])
        cursor = chunk_end
    df = pd.DataFrame(events.values())
    os.makedirs(os.path.dirname(CATALOG_CSV), exist_ok=True)
    df.to_csv(CATALOG_CSV, index=False)
    log.info("Saved %d events to %s", len(df), CATALOG_CSV)
    return df


def build_dataset(catalog, segments):
    """Return a DataFrame of samples with FEATURE_NAMES + label, sorted by time."""
    cat = catalog.dropna(subset=["time", "latitude", "longitude"]).copy()
    cat["time"] = cat["time"].astype(np.int64)
    cat = cat.sort_values("time").reset_index(drop=True)

    # Density counts every catalogue event (even ones lacking magnitude/depth).
    cat["local_density_30d"] = F.causal_density(cat["time"], cat["latitude"], cat["longitude"],
                                                cat["time"], cat["latitude"], cat["longitude"])

    samples = cat.dropna(subset=["mag", "depth_km"]).copy()
    # Warm-up: the first 30 days have an incomplete look-back window.
    warmup_end = cat["time"].min() + F.DENSITY_WINDOW_DAYS * F.DAY_MS
    samples = samples[samples["time"] >= warmup_end].copy()
    samples["plate_distance_km"] = F.plate_distance_km(samples["latitude"], samples["longitude"], segments)
    samples["label"] = (samples["mag"] >= F.SIGNIFICANT_MAG).astype(int)
    return samples.reset_index(drop=True)


def chronological_split(df, train_frac=0.70, val_frac=0.15):
    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)
    i_train, i_val = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_nn(X_tr, y_tr, X_va, y_va, seed, epochs=200, batch_size=256, lr=1e-3, patience=15):
    set_seed(seed)
    model = EarthquakeNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    Xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
    Xv, yv = torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32).view(-1, 1)
    gen = torch.Generator().manual_seed(seed)

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for start in range(0, len(Xt), batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:          # BatchNorm needs >1 sample in training mode
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 10 == 0:
            log.info("  epoch %3d  val_loss %.4f  (best %.4f @ %d)", epoch, val_loss, best_loss, best_epoch)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs_run": epoch, "best_epoch": best_epoch, "best_val_log_loss": round(best_loss, 5),
                   "batch_size": batch_size, "learning_rate": lr, "weight_decay": 1e-4,
                   "patience": patience, "optimizer": "Adam", "loss": "binary cross-entropy", "seed": seed}


def nn_proba(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().reshape(-1)


def grid_rate_baseline(train, base_rate):
    """Smoothed historical share of M>=5 events per 5-degree cell, learned from train only."""
    def cell(df):
        return (np.floor(df["latitude"] / GRID_DEG).astype(int).astype(str) + ":" +
                np.floor(F.wrap_lon(df["longitude"]) / GRID_DEG).astype(int).astype(str))
    g = train.assign(cell=cell(train)).groupby("cell")["label"].agg(["sum", "count"])
    rates = ((g["sum"] + GRID_PRIOR * base_rate) / (g["count"] + GRID_PRIOR)).to_dict()
    return lambda df: cell(df).map(rates).fillna(base_rate).to_numpy()


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def prob_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1])}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, p)
        out["average_precision"] = average_precision_score(y, p)
    return {k: round(float(v), 4) for k, v in out.items()}


def best_f1_threshold(y, p):
    candidates = np.unique(np.round(p, 3))
    scores = [(f1_score(y, p >= t, zero_division=0), t) for t in candidates]
    return float(max(scores)[1]) if scores else 0.5


def threshold_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(t, 3),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "accuracy": round(float((pred == y).mean()), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def calibration_table(y, p, bins=(0, .1, .2, .3, .4, .5, .6, 1.0)):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "mean_predicted": round(float(p[m].mean()), 4),
                         "observed_rate": round(float(y[m].mean()), 4)})
    return rows


def write_report(meta):
    m, b = meta["metrics"], meta["baselines"]
    rows = [("MLP (this model)", m["test"])] + [(name, v) for name, v in b.items()]
    lines = [
        "# TerraWatcher model evaluation",
        "",
        "_Generated automatically by `train_model.py` on " + meta["trained_at"] + ". Do not edit by hand._",
        "",
        "## What is being predicted",
        "",
        meta["interpretation"],
        "",
        f"* Catalogue: USGS FDSN event service, M >= {meta['label']['catalog_min_mag']}, "
        f"{meta['data']['start']} to {meta['data']['end']} ({meta['data']['catalog_events']} events, "
        f"{meta['data']['samples']} usable samples after the {F.DENSITY_WINDOW_DAYS}-day warm-up).",
        f"* Positive label: magnitude >= {meta['label']['significant_mag']}. "
        f"Training base rate: {meta['label']['train_positive_rate']:.3f}.",
        "",
        "## Chronological split",
        "",
        "| Split | Samples | Positive rate | From | To |",
        "|---|---:|---:|---|---|",
    ]
    for name in ("train", "validation", "test"):
        s = meta["split"][name]
        lines.append(f"| {name} | {s['n']} | {s['positive_rate']:.3f} | {s['from']} | {s['to']} |")
    lines += ["", "## Test-period results (newest 15%, never used for training or early stopping)", "",
              "| Model | ROC-AUC | Average precision | Brier | Log loss |", "|---|---:|---:|---:|---:|"]
    for name, v in rows:
        lines.append(f"| {name} | {v.get('roc_auc', float('nan')):.4f} | {v.get('average_precision', float('nan')):.4f} "
                     f"| {v['brier']:.4f} | {v['log_loss']:.4f} |")
    t = m["test_at_threshold"]
    cm = t["confusion_matrix"]
    lines += ["", f"Brier skill score vs. constant base rate: **{m['brier_skill_vs_base_rate']:+.4f}** "
              "(0 = no better than always predicting the base rate; negative = worse).", "",
              f"### Decision threshold {t['threshold']} (chosen to maximise F1 on validation, applied to test)", "",
              f"Precision {t['precision']:.3f} · Recall {t['recall']:.3f} · F1 {t['f1']:.3f} · Accuracy {t['accuracy']:.3f}",
              "", "| | Predicted < M5 | Predicted >= M5 |", "|---|---:|---:|",
              f"| Actual < M5 | {cm['tn']} | {cm['fp']} |", f"| Actual >= M5 | {cm['fn']} | {cm['tp']} |", "",
              f"Accuracy of always predicting '< M5' on test: {m['majority_class_accuracy']:.3f}.", "",
              "### Calibration on test", "", "| Predicted bin | n | Mean predicted | Observed rate |",
              "|---|---:|---:|---:|"]
    for r in m["test_calibration"]:
        lines.append(f"| {r['bin']} | {r['n']} | {r['mean_predicted']:.3f} | {r['observed_rate']:.3f} |")
    lines += ["", "## Limitations", ""] + [f"* {x}" for x in meta["limitations"]] + [""]
    os.makedirs(os.path.dirname(EVAL_REPORT), exist_ok=True)
    with open(EVAL_REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# -----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=24, help="months of history to download (default 24)")
    ap.add_argument("--offline", action="store_true", help=f"reuse {os.path.relpath(CATALOG_CSV, config.BASE_DIR)}")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)

    segments = F.load_plate_segments(config.TECTONIC_FILE)
    if len(segments) == 0:
        log.error("Plate boundaries are required for training")
        return 1

    if args.offline:
        if not os.path.exists(CATALOG_CSV):
            log.error("%s not found; run without --offline first", CATALOG_CSV)
            return 1
        catalog = pd.read_csv(CATALOG_CSV)
    else:
        catalog = download_catalog(args.months, F.CATALOG_MIN_MAG)
    catalog = catalog[catalog["mag"].isna() | (catalog["mag"] >= F.CATALOG_MIN_MAG)]

    df = build_dataset(catalog, segments)
    if len(df) < 1000 or df["label"].sum() < 50:
        log.error("Not enough data to train (%d samples, %d positives)", len(df), int(df["label"].sum()))
        return 1
    train, val, test = chronological_split(df)
    assert train["time"].max() <= val["time"].min() <= val["time"].max() <= test["time"].min()

    X = {k: d[F.FEATURE_NAMES].to_numpy(float) for k, d in (("train", train), ("val", val), ("test", test))}
    y = {k: d["label"].to_numpy(int) for k, d in (("train", train), ("val", val), ("test", test))}
    scaler = StandardScaler().fit(X["train"])
    Xs = {k: scaler.transform(v) for k, v in X.items()}
    base_rate = float(y["train"].mean())

    log.info("Samples: train %d, val %d, test %d. Train positive rate %.3f",
             len(train), len(val), len(test), base_rate)
    model, train_info = train_nn(Xs["train"], y["train"], Xs["val"], y["val"], args.seed)

    p_val, p_test = nn_proba(model, Xs["val"]), nn_proba(model, Xs["test"])
    threshold = best_f1_threshold(y["val"], p_val)

    logreg = LogisticRegression(max_iter=1000).fit(Xs["train"], y["train"])
    grid = grid_rate_baseline(train, base_rate)
    baselines = {
        "Constant train base rate": prob_metrics(y["test"], np.full(len(test), base_rate)),
        "5-degree regional rate (train history)": prob_metrics(y["test"], grid(test)),
        "Logistic regression (same features)": prob_metrics(y["test"], logreg.predict_proba(Xs["test"])[:, 1]),
    }
    test_metrics = prob_metrics(y["test"], p_test)
    brier_base = baselines["Constant train base rate"]["brier"]

    def fmt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    def span(d):
        return {"n": int(len(d)), "positive_rate": round(float(d["label"].mean()), 4),
                "from": fmt(int(d["time"].min())), "to": fmt(int(d["time"].max()))}

    meta = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_names": F.FEATURE_NAMES,
        "interpretation": (
            f"For an earthquake that has been recorded with magnitude >= {F.CATALOG_MIN_MAG}, the model estimates "
            f"the probability that its magnitude is >= {F.SIGNIFICANT_MAG}, from its location, depth, the number "
            f"of M>={F.CATALOG_MIN_MAG} events within +/-{F.DENSITY_RADIUS_DEG} degree in the preceding "
            f"{F.DENSITY_WINDOW_DAYS} days, and its distance to the nearest mapped plate boundary. It does not "
            "estimate whether, when or where an earthquake will occur."
        ),
        "label": {"catalog_min_mag": F.CATALOG_MIN_MAG, "significant_mag": F.SIGNIFICANT_MAG,
                  "train_positive_rate": round(base_rate, 4)},
        "data": {"source": "USGS FDSN event web service (eventtype=earthquake)",
                 "start": span(df)["from"], "end": span(df)["to"],
                 "catalog_events": int(len(catalog)), "samples": int(len(df))},
        "split": {"method": "chronological 70/15/15", "train": span(train), "validation": span(val), "test": span(test)},
        "training": train_info,
        "metrics": {
            "test": test_metrics,
            "validation": prob_metrics(y["val"], p_val),
            "test_at_threshold": threshold_metrics(y["test"], p_test, threshold),
            "brier_skill_vs_base_rate": round(1 - test_metrics["brier"] / brier_base, 4),
            "majority_class_accuracy": round(float((y["test"] == 0).mean()), 4),
            "test_calibration": calibration_table(y["test"], p_test),
        },
        "baselines": baselines,
        "limitations": [
            "Conditional on an event having occurred: the score is not a probability that an earthquake will happen.",
            "Magnitude types differ across networks (mb, Mww, ml...); the label mixes them as reported by USGS.",
            "Aftershock sequences that straddle split boundaries make adjacent periods correlated.",
            "Latitude/longitude are raw inputs, so the model partly memorises regional behaviour; it is not "
            "validated on regions absent from training.",
            "Evaluated on a single recent test period; results can change with each retraining.",
        ],
    }

    torch.save(model.state_dict(), config.MODEL_FILE)
    joblib.dump(scaler, config.SCALER_FILE)
    with open(config.MODEL_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    write_report(meta)

    log.info("Test (MLP): %s", test_metrics)
    for name, v in baselines.items():
        log.info("Test (%s): %s", name, v)
    log.info("Brier skill vs base rate: %+.4f", meta["metrics"]["brier_skill_vs_base_rate"])
    log.info("Wrote %s, %s, %s, %s", *(os.path.relpath(p, config.BASE_DIR) for p in
             (config.MODEL_FILE, config.SCALER_FILE, config.MODEL_META_FILE, EVAL_REPORT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
