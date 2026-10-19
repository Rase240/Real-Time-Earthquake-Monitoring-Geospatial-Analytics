"""
Validated event tables and the historical catalogue used for training and evaluation.

Every analytical module consumes events through `events_to_frame`, which applies
one set of rules:

  * rows without a finite time, latitude in [-90, 90] or longitude in [-180, 180]
    are dropped (counted in `stats["dropped_invalid"]`), never repaired;
  * duplicate ids keep the most recently updated copy (counted in
    `stats["duplicates_removed"]`);
  * missing magnitude / depth stay NaN. Rows without magnitude are kept in the
    table but never satisfy a magnitude threshold;
  * the result is sorted by time (ascending), times are integer UTC epoch ms.

The historical catalogue (`download_history`) is fetched in time chunks from the
USGS FDSN service. A chunk that fails or that reaches the 20 000-event response
limit is retried or split; the download aborts rather than leaving a hole, so a
gap is never silently turned into zero activity.
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
import usgs_client

log = logging.getLogger("terrawatcher.catalog")

COLUMNS = ["id", "time", "latitude", "longitude", "depth_km", "mag", "mag_type"]
HISTORY_CSV = os.path.join(config.BASE_DIR, "data", "history_catalog.csv")
HISTORY_META = os.path.join(config.BASE_DIR, "data", "history_catalog.json")
HISTORY_MIN_MAG = 4.5
HISTORY_START = datetime(2000, 1, 1, tzinfo=timezone.utc)
USGS_LIMIT = 20000
DAY_MS = 86_400_000


def events_to_frame(events):
    """Normalised event dicts (or a DataFrame) -> (validated DataFrame, stats)."""
    df = pd.DataFrame(events if not isinstance(events, pd.DataFrame) else events.copy())
    for col in COLUMNS + ["updated"]:
        if col not in df.columns:
            df[col] = np.nan
    n0 = len(df)
    for col in ("time", "latitude", "longitude", "depth_km", "mag", "updated"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ok = (np.isfinite(df["time"]) & df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180))
    df = df[ok].copy()
    dropped = n0 - len(df)
    df["id"] = df["id"].astype(str)
    df["_upd"] = df["updated"].fillna(-1)
    df = df.sort_values(["id", "_upd"]).drop_duplicates("id", keep="last")
    duplicates = n0 - dropped - len(df)
    df["time"] = df["time"].astype(np.int64)
    df = df.sort_values(["time", "id"], kind="stable").reset_index(drop=True)
    return df[COLUMNS], {"dropped_invalid": int(dropped), "duplicates_removed": int(duplicates)}


def _iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_chunk(start, end, min_mag):
    events, _ = usgs_client.fetch_events(start, end, min_mag=min_mag, limit=USGS_LIMIT, timeout=120, retries=4)
    if len(events) >= USGS_LIMIT:
        if end - start <= timedelta(days=1):
            raise RuntimeError(f"More than {USGS_LIMIT} events in one day starting {start}; cannot download completely")
        mid = start + (end - start) / 2
        log.info("  chunk %s..%s hit the response limit; splitting", start.date(), end.date())
        return _fetch_chunk(start, mid, min_mag) + _fetch_chunk(mid, end, min_mag)
    return events


def download_history(end=None, start=HISTORY_START, min_mag=HISTORY_MIN_MAG, chunk_days=120,
                     csv_path=HISTORY_CSV, meta_path=HISTORY_META):
    """
    Download the USGS catalogue M >= min_mag for [start, end) and write a compact CSV
    plus a JSON description of exactly what was requested. `end` defaults to the
    most recent Monday 00:00 UTC, so the catalogue always ends on a week boundary.
    """
    if end is None:
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now - timedelta(days=now.weekday())
    rows, cursor, chunks = [], start, 0
    log.info("Downloading USGS M>=%.1f catalogue %s -> %s", min_mag, start.date(), end.date())
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        batch = _fetch_chunk(cursor, chunk_end, min_mag)
        rows.extend(batch)
        chunks += 1
        log.info("  %s -> %s: %d events", cursor.date(), chunk_end.date(), len(batch))
        cursor = chunk_end
    df, stats = events_to_frame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    meta = {
        "source": "USGS FDSN event web service (eventtype=earthquake)",
        "min_mag": min_mag, "start": _iso(start.timestamp() * 1000), "end": _iso(end.timestamp() * 1000),
        "start_ms": int(start.timestamp() * 1000), "end_ms": int(end.timestamp() * 1000),
        "events": int(len(df)), "chunks": chunks, **stats,
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Saved %d events to %s", len(df), csv_path)
    return df, meta


def load_history(csv_path=HISTORY_CSV, meta_path=HISTORY_META):
    """Load the downloaded catalogue and its description. Raises FileNotFoundError if absent."""
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"{os.path.relpath(csv_path, config.BASE_DIR)} not found; "
                                "run `python ml_pipeline.py download` first")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    df, _ = events_to_frame(pd.read_csv(csv_path, dtype={"id": str}))
    return df, meta


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    download_history()
