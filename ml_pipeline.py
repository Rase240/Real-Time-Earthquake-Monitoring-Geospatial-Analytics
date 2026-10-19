"""
Offline training and evaluation for TerraWatcher's analytical modules.

  python ml_pipeline.py download     # USGS M4.5+ catalogue 2000-01-01 .. last Monday -> data/history_catalog.csv
  python ml_pipeline.py forecast     # Module B: select, walk-forward evaluate and save the forecast models
  python ml_pipeline.py anomaly      # Module C: held-out, null-simulation and injection evaluation
  python ml_pipeline.py sequences    # Module A: synthetic-label clustering evaluation
  python ml_pipeline.py aftershock   # Module D: Omori-Utsu fits across many historical sequences
  python ml_pipeline.py all          # forecast + anomaly + sequences + aftershock (+ report)
  python ml_pipeline.py report       # rebuild docs/ml_evaluation.md from models/*.json

Every command except `download` and `sequences` reads data/history_catalog.csv
(downloaded once). Outputs go to models/ and docs/ml_evaluation.md. Training is
never done by the web application.
"""
import argparse
import json
import logging
import os
import sys

import config
from analytics import catalog

log = logging.getLogger("terrawatcher.pipeline")
MODEL_DIR = os.path.join(config.BASE_DIR, "models")
REPORT = os.path.join(config.BASE_DIR, "docs", "ml_evaluation.md")


def _write_json(name, obj):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    log.info("Wrote %s", os.path.relpath(path, config.BASE_DIR))


def cmd_forecast(cat, meta):
    from analytics import forecast as FC
    from analytics import forecast_training as T
    mats, large, n_weeks = T.build_matrices(cat, meta["end_ms"])
    backtests = {}
    for thr in T.THRESHOLDS:
        est, m, back = T.train_threshold(mats[thr], large, n_weeks, thr)
        m["data"] = {k: meta[k] for k in ("source", "min_mag", "start", "end", "events", "downloaded_at")}
        FC.save_artifact(MODEL_DIR, thr, est, m)
        backtests[f"{thr:.1f}"] = back
    _write_json("forecast_backtest.json", backtests)


def cmd_anomaly(cat, meta):
    from analytics import evaluation
    result, reference = evaluation.evaluate_anomaly(cat, meta)
    _write_json("anomaly_reference.json", reference)
    _write_json("anomaly_eval.json", result)


def cmd_sequences(cat=None, meta=None):
    from analytics import evaluation
    _write_json("sequence_eval.json", evaluation.evaluate_sequences())


def cmd_aftershock(cat, meta):
    from analytics import evaluation
    _write_json("aftershock_eval.json", evaluation.evaluate_aftershocks(cat, meta))


def cmd_report(*_):
    from analytics import report
    report.write(MODEL_DIR, REPORT)
    log.info("Wrote %s", os.path.relpath(REPORT, config.BASE_DIR))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "forecast", "anomaly", "sequences", "aftershock", "all", "report"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)

    if args.command == "download":
        catalog.download_history()
        return 0
    if args.command == "report":
        cmd_report()
        return 0
    if args.command == "sequences":
        cmd_sequences()
        cmd_report()
        return 0
    try:
        cat, meta = catalog.load_history()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    steps = {"forecast": [cmd_forecast], "anomaly": [cmd_anomaly], "aftershock": [cmd_aftershock],
             "all": [cmd_forecast, cmd_anomaly, cmd_sequences, cmd_aftershock]}[args.command]
    for step in steps:
        step(cat, meta)
    cmd_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
