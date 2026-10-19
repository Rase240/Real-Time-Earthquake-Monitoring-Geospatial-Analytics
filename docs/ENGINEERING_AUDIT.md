# TerraWatcher engineering audit (2026-10-19)

This document records the defects found in the repository as it stood at commit `99915da`,
what was changed, and how each change was verified. Older reports from a previous pass are in
[`archive/`](archive/) and no longer describe the code.

## 1. Defects found

| # | Area | Defect | Evidence |
|---|---|---|---|
| D1 | Data feed | USGS `endtime` was formatted as a date (`YYYY-MM-DD`), i.e. today 00:00 UTC, so the "24H" view showed *yesterday* and missed the most recent ~day of events. | Live call on 2026-10-18 21:18 UTC: newest returned event was 23 h old. |
| D2 | Data feed | Hard `limit=300` with no indication; a 30-day M2.5+ request silently returned exactly 300 events. | Live call returned 300 records. |
| D3 | Data feed | Missing coordinates were replaced with `0.0` (plotting fake events at 0°N 0°E) and missing depth with `10.0`. | `app.py` coordinate parsing. |
| D4 | Caching | Unbounded dict keyed by arbitrary floats, no lock, cached objects mutated in place; cached data indistinguishable from fresh data. | `USGS_CACHE` in old `app.py`. |
| D5 | ML – leakage | Training `local_density` counted all events in the whole 6-month window, including aftershocks *after* each event (future information). | `compute_density` used the full training frame. |
| D6 | ML – train/serve skew | At inference, density was a live USGS count of *all magnitudes* in the last 30 days within the box; in training it was M2.5+ over 6 months. In `/api/earthquakes` it was computed from the ≤300 returned events. Three different definitions of one feature. | `get_live_local_density`, `compute_density`. |
| D7 | ML – evaluation | Random `train_test_split` on time-ordered data, accuracy as the only metric (misleading with ~10–25 % positives), no baselines, no calibration, no stored metrics. | old `train_model.py`. |
| D8 | ML – label validity | Training used M2.5+, but the global catalogue is only complete to roughly M4.5; below that, coverage depends on regional networks (dense in the US, sparse elsewhere). An M≥5 label on an M2.5+ catalogue therefore partly encodes network coverage. This was reasoned from catalogue properties; it was not measured here. | Design review. |
| D9 | Hotspots | `/api/hotspots` returned 7 hard-coded coordinates scored with a constant `live_density=5`, and a fabricated `0.5` when the model was missing. | old `hotspots()` route. |
| D10 | Wasted / misleading work | Every observed event was scored by the model (`props['prediction']`) although its magnitude is already known; the UI never displayed it. ~300 single-row forward passes per request. | old `/api/earthquakes`. |
| D11 | Inference | `/api/predict` made a blocking USGS call per click and silently used density `0` on failure. | `get_live_local_density` swallowed exceptions. |
| D12 | Model metadata | `/api/model-info` returned hard-coded `"epochs": 60`, `"version": "v1.1"`. Older docs described a different architecture (5-64-32-16-1). | old route; archived reports. |
| D13 | Error handling | `str(e)` of arbitrary exceptions returned to clients; `warnings.filterwarnings("ignore")` hid all warnings globally; non-JSON 404/405 for API paths. | old `app.py`. |
| D14 | Frontend | Map showed only part of the world at load; hotspot rings labelled "model hotspot" for hard-coded points; optional `?carto_key=` URL parameter put an API key in the URL; CSS was a 2-line minified file that could not be maintained. | Browser inspection, `dashboard.js`. |
| D15 | Dependencies | `torch 2.5.1` (24 advisories, fixed ≥ 2.13), `requests 2.32.5` (1), `pytest 8.4.2` (1) per `pip-audit`. `tqdm` unused. | `pip-audit -r requirements.txt`. |
| D16 | Tests | One test hit the live USGS API; tests asserted the old contract (e.g. exactly 7 hotspots). | old `tests/test_application.py`. |
| D17 | Docs | README claimed "100% automated test coverage", "spherical distances", a "deep neural network", an MIT `LICENSE` file that does not exist, and listed `dataset_fetcher.py`, whose output nothing used. | README. |

## 2. Changes

* **Modules.** `app.py` was split into `config.py` (environment configuration), `usgs_client.py`
  (HTTP, retries, validation), `features.py` (shared feature definitions), `ml_model.py` (network and
  artifact loading) and `app.py` (routes, caching, errors).
* **Feed (D1–D4).** Queries now use full ISO timestamps, up to 2000 events, `eventtype=earthquake`,
  truncation reporting via the FDSN `count` endpoint, records with invalid coordinates or times are
  dropped (and counted), and duplicates are removed by id. The cache is a bounded, thread-safe LRU with
  TTL, and a stale copy (≤30 min) is served only when USGS fails, labelled `cache.stale=true`.
* **Features (D5, D6).** One definition is used for training and serving: M4.5+ events within ±1° in the
  30 days *strictly before* the reference time. At serving time the reference catalogue is one cached
  30-day M4.5+ download (10 min TTL), so there is no per-click USGS call. Plate distance is vectorised
  and expressed in km (local equirectangular approximation).
* **Training (D7, D8).** The model is retrained on 24 months of M4.5+ data. The split is chronological
  70/15/15, early stopping uses the validation period, and the test period is untouched. The model is
  compared with a constant base rate, a smoothed 5° regional rate and logistic regression. ROC-AUC,
  average precision, Brier, log loss, a thresholded confusion matrix and a calibration table are written
  to `model_meta.json` and `docs/model_evaluation.md`.
* **Hotspots (D9).** Hotspots are now *observed activity clusters*: the 2° cells with the most M4.5+
  events in the last 30 days. Each cluster's conditional severity score is `null` when the model is
  unavailable.
* **Inference (D10, D11).** Observed events are no longer scored. `/api/predict` returns 503 with an
  explicit reason if the model or the recent-activity context is unavailable. Responses include the
  base rate and a ratio band instead of absolute "high/medium/low risk" labels.
* **Model loading (D12).** Loading is all-or-nothing and validates the feature schema stored in
  `model_meta.json`, the scaler width and the state_dict. `/api/model-info` reports the real metadata
  and the load error, if any.
* **Errors and security (D13).** Errors use one JSON shape (`{error, code, status}`). Unhandled
  exceptions are logged server-side and return a generic 500. The app sends a Content-Security-Policy,
  `nosniff` and `Referrer-Policy`, and no CORS headers (the UI is same-origin).
* **Frontend (D14).** New layout with a controls sidebar, large map, inspector, charts and model card.
  It has observed-event, activity-cluster and plate-boundary layers, a client-side depth filter, a
  selected-event detail view, point analysis with keyboard-usable inputs, stale/cached/fresh labels,
  error banners with Retry, superseded-request cancellation, request timeouts and responsive layouts.
  The pure logic lives in `static/core.js` and is unit-tested.
* **Dependencies (D15).** Pins are updated (`torch>=2.13,<2.15`, `requests>=2.33`) and test tooling is
  moved to `requirements-dev.txt`.
* **Tests (D16).** There are 91 offline pytest tests (fake USGS transport) and 10 Node tests.
* **Cleanup (D17).** `dataset_fetcher.py` was removed (it was dead code) and the stale reports were moved
  to `docs/archive/`.

## 3. Verification performed

See the README section "Verification" for the commands and observed results.
