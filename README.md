# TerraWatcher — Seismic Analytics on the USGS Earthquake Catalogue

TerraWatcher is a Flask + Leaflet application that combines live data from the public **USGS FDSN event
web service** with four evaluated analytical modules:

1. **Sequence Explorer**: unsupervised space-time clustering (DBSCAN on Earth-centred coordinates).
2. **Activity Forecasts**: seven-day regional event-count forecasts (Poisson GLM or gradient boosting with
   negative-binomial intervals), evaluated walk-forward against baselines.
3. **Anomaly Detection**: elevated observed activity per region against its own history
   (negative-binomial reference, Benjamini–Hochberg correction).
4. **Aftershock Analysis**: event-centred statistics and a modified Omori–Utsu fit, shown only when it
   passes validity checks.

A **Model Performance** view shows the evaluation results of every module, read from the generated
files. A **Live Monitor** keeps the original event feed, map and point analysis.

![TerraWatcher dashboard](docs/images/dashboard-desktop.png)

> **What TerraWatcher does not do.** It does not predict the time, place or magnitude of any individual
> earthquake. Forecasts are aggregate counts for 10° cells over seven days. An "elevated activity" flag
> describes what was observed, usually an ongoing sequence, and is not a warning. For alerts, use
> official agencies (USGS, national geological surveys, tsunami warning centres).

📘 **Learning guide:** [`docs/learning-guide.pdf`](docs/learning-guide.pdf) (source:
[`docs/learning-guide.md`](docs/learning-guide.md)). **Evaluation report:**
[`docs/ml_evaluation.md`](docs/ml_evaluation.md), generated from `models/*.json`.

---

## Contents

[Features](#features) · [Architecture](#architecture) · [Setup](#setup) · [Configuration](#configuration) ·
[API](#api) · [Methods](#methods) · [Evaluation results](#evaluation-results) · [Limitations](#limitations) ·
[Testing](#testing) · [Troubleshooting](#troubleshooting) · [Future work](#future-work)

## Features

| View | What it shows | Kind of output |
|---|---|---|
| **Live Monitor** | Events from the last 24 h / 7 d / 30 d, filters, statistics, freshness, plate boundaries, observed M4.5+ activity cells, event detail. | Observed data |
| **Sequence Explorer** | DBSCAN sequences of the current feed with adjustable ε (km, hours) and minimum core size. For each sequence: statistics, magnitude and depth histograms, timeline, member events, and why the events were grouped. | Unsupervised analysis |
| **Activity Forecasts** | For each 10° cell: expected M4.5+ (or M5.0+) count for the next 7 days, 80 % / 95 % intervals, P(≥ 1), baselines, inputs at the origin, the cell's test-period metrics, and observed weekly counts next to out-of-sample forecasts. | Model estimate |
| **Anomaly Detection** | Cells with elevated observed activity versus their reference period: observed vs expected, p and q values, dispersion, contributing events, explanation, data-gap status. | Statistical test on observed data |
| **Aftershock Analysis** | For a chosen reference event (from the feed, a sequence, an anomaly or a typed USGS id): timeline, cumulative count, rate on log–log axes, magnitude, depth and distance histograms, pre/post comparison, an Omori–Utsu fit with uncertainty and diagnostics (or the reasons it was rejected), and comparison of up to four sequences. | Descriptive statistics + statistical model |
| **Model Performance** | Validation vs final-test metrics, baseline comparisons, by-year stability and reliability for both forecast thresholds; synthetic clustering scores; anomaly false-alert, injection and missing-data results; Omori–Utsu results across historical sequences; the legacy classifier. | Evaluation results |

| Activity Forecasts | Anomaly Detection |
|---|---|
| ![Activity Forecasts](docs/images/forecast-view.png) | ![Anomaly Detection](docs/images/anomaly-view.png) |

Shared behaviour:

* Every view has explicit loading, empty, error, stale and retry states.
* A newer request supersedes an older one, so a stale response never overwrites a newer view.
* A failed USGS request or missing model gives an error message. It is never turned into empty data or a fabricated result.
* The live monitor keeps working if any analytical model cannot be loaded (`/api/health` → `degraded`).

## Architecture

```mermaid
flowchart LR
    B[Browser<br/>index.html · core.js · charts.js<br/>analytics.js · dashboard.js · ml.js<br/>Leaflet] -- "GET /api/* (same origin, JSON)" --> F[Flask app.py]
    F -- "feed · 400-day M4.5+ history<br/>event lookup · radius query" --> U[(USGS FDSN)]
    F --> C[[feed_cache: raw downloads, TTL + stale fallback<br/>analysis_cache: derived results keyed by data version + parameters]]
    F --> A[analytics/<br/>sequences · regional_features · forecast<br/>anomaly · aftershock · geo · regions]
    A --> M[(models/<br/>forecast_m4.5/5.0 .joblib + .json<br/>anomaly_reference.json<br/>*_eval.json · forecast_backtest.json)]
    P[ml_pipeline.py — offline only] -- "download 2000→now" --> U
    P -- writes --> M
    P -- writes --> R[docs/ml_evaluation.md]
```

* **Live data.** The feed (`/api/earthquakes`, also clustered by `/api/sequences`) and a 400-day global
  M4.5+ history (forecast features and anomaly windows) are each downloaded once per cache lifetime. The
  download is single-flight: concurrent cold requests share it. A truncated history download (20 000-event
  limit) is refused, not silently used.
* **No training or refitting per request.** Forecast models are trained offline and loaded at start-up
  with schema and version checks. Anomaly tests and Omori–Utsu fits are cheap and deterministic: they are
  computed on demand and memoised under keys that include every parameter and the data version.
* **One feature definition.** `analytics/regional_features.features_at` builds the forecast inputs for
  both training (many origins) and serving (one live origin). A test checks that both paths give
  identical features.

### Repository layout

```text
app.py                      Flask routes, validation, caching, errors, security headers
config.py                   Tunables (environment-overridable)
usgs_client.py              USGS client: timeouts, retry, validation, de-duplication, event lookup, radius search
analytics/
  geo.py                    Great-circle distance, Earth-centred vectors, spherical centroid, longitude span
  catalog.py                One table-validation rule set; historical catalogue download (chunked, never gappy)
  regions.py                10° grid, region ids, neighbours (antimeridian-aware), weekly count matrices
  regional_features.py      Forecast-origin features: definitions, units, leakage contract
  sequences.py              Module A — space-time DBSCAN and sequence summaries
  forecast.py               Module B — models, NB predictive distribution, metrics, artifact loading
  forecast_training.py      Module B — dataset, validation selection, walk-forward test, served fit
  anomaly.py                Module C — NB reference, BH correction, gap detection
  aftershock.py             Module D — selection, binned statistics, Omori–Utsu MLE and diagnostics
  evaluation.py             Offline evaluation of modules A, C and D
  report.py                 Builds docs/ml_evaluation.md from models/*.json
ml_pipeline.py              CLI: download / forecast / anomaly / sequences / aftershock / all / report
models/                     Trained forecast models, evaluation results, backtests (generated, committed)
features.py, ml_model.py,   Legacy conditional severity classifier (point analysis), unchanged
train_model.py, *.pth, scaler.pkl, model_meta.json
static/core.js              Feed validation, statistics, formatting (Node-tested)
static/charts.js            Dependency-free SVG charts with labelled axes (Node-tested)
static/analytics.js         View helpers: response validation, request supersession, explanations (Node-tested)
static/dashboard.js         Live Monitor, map, shared fetch layer
static/ml.js                The five analytical views
tests/                      pytest (offline, fake USGS) + tests/frontend (node:test)
docs/                       Learning guide (+ build script), ml_evaluation.md, audit, images
```

## Setup

Prerequisites: **Python 3.12** and Git. **Node.js 18+** is needed only for the frontend unit tests. The
app needs internet access to reach `earthquake.usgs.gov`, the Esri tiles and the Leaflet CDN.

```bash
git clone https://github.com/Rase240/Real-Time-Earthquake-Monitoring-Geospatial-Analytics.git
```

```bash
cd Real-Time-Earthquake-Monitoring-Geospatial-Analytics
```

```bash
python -m venv .venv
```

Activate the virtual environment (Windows PowerShell: `.\.venv\Scripts\Activate.ps1`; Linux/macOS:
`source .venv/bin/activate`), then install the dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the app and open <http://127.0.0.1:5000>:

```bash
python app.py
```

The trained models and evaluation files are committed, so no training is needed. The first visit to
Activity Forecasts or Anomaly Detection downloads about a year of global M4.5+ events (about 9 000
records, typically 5–10 s). After that it is served from memory for an hour.

### Reproducing the models and evaluations

Download the historical catalogue once (≈ 188 000 events, 2000-01-01 to the last Monday, about 10 MB,
not committed):

```bash
python ml_pipeline.py download
```

Then train and evaluate everything and rebuild `docs/ml_evaluation.md` (≈ 6 minutes on a laptop CPU):

```bash
python ml_pipeline.py all
```

Single steps: `forecast`, `anomaly`, `sequences` (synthetic data only, no download needed), `aftershock`
and `report`. Seeds are fixed. Two consecutive `all` runs on the same catalogue produced identical
metrics. The remaining source of variation is the catalogue itself: USGS revises magnitudes and
locations over time, so a later download can change the numbers slightly. The served forecast
artifacts depend on the scikit-learn version (pinned `<1.8`). A different version is refused at load
time, not used.

## Configuration

All settings are optional environment variables. No secrets or API keys are used.

| Variable | Default | Meaning |
|---|---|---|
| `TERRAWATCHER_HOST` / `TERRAWATCHER_PORT` | `127.0.0.1` / `5000` | Bind address and port. |
| `TERRAWATCHER_DEBUG` | `0` | Flask debug mode (never on a shared network). |
| `TERRAWATCHER_LOG_LEVEL` | `INFO` | Logging level. |
| `TERRAWATCHER_USGS_URL` | `https://earthquake.usgs.gov/fdsnws/event/1` | FDSN base URL. |
| `TERRAWATCHER_USGS_TIMEOUT` / `_RETRIES` | `15` s / `1` | Per-request timeout and retries on timeout / 429 / 5xx. |
| `TERRAWATCHER_FEED_CACHE_TTL` | `60` s | Event-feed cache lifetime. |
| `TERRAWATCHER_CONTEXT_CACHE_TTL` | `600` s | 30-day context for the legacy point analysis. |
| `TERRAWATCHER_STALE_MAX_AGE` | `1800` s | Maximum age of a labelled stale copy when USGS fails. |
| `TERRAWATCHER_HISTORY_CACHE_TTL` / `_HISTORY_STALE_MAX_AGE` | `3600` / `21600` s | 400-day analytical history. |
| `TERRAWATCHER_AFTERSHOCK_CACHE_TTL` | `600` s | Event lookup and radius queries. |
| `TERRAWATCHER_MODEL_DIR` | `models/` | Location of forecast and evaluation artifacts. |

## API

All endpoints are `GET` and return JSON. Errors always look like
`{"error": "<message>", "code": "<machine code>", "status": <http status>}`, and raw exception text is never returned.

| Endpoint | Parameters (validated) | Returns |
|---|---|---|
| `/api/health` | — | Overall status, plus per-module readiness (`forecast` per threshold, `anomaly_long_term_dispersion`, `evaluations`) and cache ages. |
| `/api/earthquakes` | `days` 1–30, `min_mag` −1–10 | Observed events (≤ 2000, newest first), truncation, query window, freshness. |
| `/api/sequences` | `days`, `min_mag` as above; `eps_km` 5–500 (50), `eps_hours` 1–720 (72), `min_samples` 2–50 (4) | Sequences (id, counts, start/end, duration, magnitude and depth distributions and histograms, centroid, extent incl. antimeridian flag, linking statistics, member ids), compact events with assignments, summary, explanation, limitations. |
| `/api/forecast` | `threshold` ∈ {4.5, 5.0} | Origin, valid window, target definition, model and test metrics, data window, and for each eligible cell: expected count, 80 % / 95 % intervals, P(≥1), baselines. |
| `/api/forecast/<region>` | `threshold`; region id like `R+30+140` | The cell's forecast, 52 observed weekly counts, input features with units and definitions, walk-forward backtest series, and the cell's test metrics. |
| `/api/anomalies` | `threshold` ∈ {4.5, 5.0}, `window_weeks` ∈ {1, 2, 4}, `reference_weeks` ∈ {26, 52}, `fdr` ∈ {0.01, 0.05, 0.1} | Observation and reference windows, data-quality block (possible gap days, dispersion source), method text, and for each cell: status, observed, expected, p, q, ratio, dispersion, largest event. |
| `/api/anomalies/<region>` | same | The cell's result, weekly counts (reference + observation) and contributing events. |
| `/api/aftershocks` | `event_id` (USGS id); `radius_km` 5–1000 (default: rupture-length rule); `pre_days` 0–365 (30); `post_days` 1–365 (60); `min_mag` 0–8 (4.5); `min_depth`, `max_depth` 0–800; `fit_start_hours` 0–72 (6) | Reference event, parameters, selection rules, statistics (daily, hourly, log-time bins, cumulative, histograms, pre/post test), fit (status, parameters, SE, 95 % CI, diagnostics, reasons/notes, curve inside the fit window), member events, data window. |
| `/api/model-performance` | — | Evaluation results of every module, read from `models/*.json`, and the legacy classifier. |
| `/api/hotspots`, `/api/predict`, `/api/model-info`, `/api/plates` | as before | Observed activity cells; legacy conditional severity score and its metadata; plate boundaries. |

Example (real response, 2026-10-19, lists shortened):

```http
GET /api/anomalies?threshold=4.5&window_weeks=4
```
```json
{
  "observation_window": {"from": "2026-09-21T08:00:00Z", "to": "2026-10-19T08:00:00Z"},
  "reference_window": {"from": "2025-09-22T08:00:00Z", "to": "2026-09-21T08:00:00Z"},
  "regions_tested": 145, "regions_elevated": 2,
  "results": [{"region": "R+00-090", "label": "0°–10°N, 90°W–80°W", "status": "elevated", "observed": 67,
               "expected": 4.154, "ratio": 16.13, "p_value": 1.671e-12, "q_value": 2.407e-10,
               "largest_event": {"id": "us6000u18k", "mag": 7.7, "place": "12 km WSW of Pitaloza Arriba, Panama"}}],
  "...": "..."
}
```

Error examples: `GET /api/forecast?threshold=4.7` gives 400 `invalid_parameter`.
`GET /api/forecast/R+35+140` gives 404 `unknown_region`. A cell without enough history gives 404
`insufficient_history`. Missing model artifacts give 503 `model_unavailable`. A failed or truncated
history download gives 503 `history_unavailable` ("No result was computed"). An unknown USGS event gives
404 `event_not_found`. `min_depth > max_depth` gives 400 `out_of_range`.

## Methods

### Data

* **Source and roles.** The USGS FDSN event service (`eventtype=earthquake`) supplies everything:
  * the live feed (any magnitude, ≤ 30 days);
  * a 400-day global M4.5+ history for forecasts and anomalies;
  * radius queries around a reference event;
  * the offline catalogue M4.5+ 2000-01-01 → 2026-10-19 (187 784 events, 82 chunks of 120 days; no
    invalid records or duplicates in this download).
* **Completeness.** M4.5 is roughly where the global catalogue is complete. Below it, coverage
  depends on regional networks. Completeness also improved over 2000–2026 and is worse right after
  large earthquakes. This is why forecasts and anomaly tests compare each cell only with its own history.
* **Validation rules** (`analytics/catalog.events_to_frame`, shared by every module):
  * Rows without finite time and valid coordinates are dropped and counted.
  * Duplicate ids keep the latest `updated` copy.
  * A missing magnitude or depth stays missing. It never becomes 0, and it never satisfies a magnitude threshold.
* **Truncation.** Queries reaching the 20 000-event limit are split (offline) or refused (history,
  aftershock fit). The feed reports `truncated` with the USGS total.
* **Gaps vs zeros.** The offline download aborts on a failed chunk instead of leaving a hole. In live
  data, a UTC day with no M4.5+ event anywhere on Earth (expected about 20 per day; none occurred in
  2000–2026) is treated as a possible feed gap. It is not treated as zero activity.
* **Time.** Every timestamp is UTC epoch milliseconds or ISO-8601 with `Z`, and every Python datetime
  is timezone-aware. Week bins are half-open `[start, start + 7 d)`.

### Module A — sequence clustering

* **Distance.** Events are mapped to Earth-centred unit vectors scaled by the chord of ε_km, and time
  is scaled by ε_hours. DBSCAN then runs with ε = 1, so two events are neighbours when
  (chord/chord(ε_km))² + (Δt/ε_h)² ≤ 1. The spatial part is exact great-circle distance at Δt = 0, and
  the antimeridian and the poles need no special cases.
* **Why DBSCAN.** It needs no cluster count, labels isolated events as noise, handles irregular shapes
  and is deterministic. HDBSCAN would adapt to density differences, but it adds a dependency and is
  harder to tune for users.
* **What is not in the distance.** Depth and magnitude are excluded and only summarised afterwards.
* **Outputs per sequence.** A stable id (hash of the earliest member's id), counts, time span,
  magnitude and depth statistics, a spherical centroid, extent with an antimeridian-aware longitude
  span, linking statistics, and ambiguous border events (border events reachable from two sequences;
  DBSCAN assigns them to one).
* **Terminology.** Results are always called *detected sequences*, with no causal claim.

### Module B — seven-day regional count forecast

* **Target.** y(r, T) is the number of catalogue events with M ≥ m (4.5 or 5.0) and epicentre in the
  10° cell r, in [T, T + 7 d).
* **Grid.** 648 equal-angle cells. A cell is eligible if its 2001–2015 mean is ≥ 0.1 events per week
  (131 cells at M4.5, 68 at M5.0).
* **Features at origin T.** Only the weeks before T are used (`analytics/regional_features.py`):
  * counts in the last week and the week before;
  * mean weekly counts over 4, 13 and 52 weeks;
  * 13-week standard deviation;
  * the 4-week mean summed over the 8 neighbouring cells (longitude wraps);
  * the number of M ≥ 6 events in the last 4 weeks;
  * weeks since the last qualifying event (capped at 52).

  There is no imputation, because a zero week in a complete download is a real zero.
* **Leakage control.** Features read only columns before the origin. Tests change the target week and
  later weeks and check that the features are unchanged. Training-side and serving-side features are
  tested for equality.
* **Temporal design.** Weekly origins on Mondays (epoch 2000-01-03, 52-week warm-up):
  * train targets ≤ 2016-01-04;
  * validation 2016-01-04 → 2020-01-06, used for model and hyperparameter choice and the NB dispersion;
  * final test 2020-01-06 → 2026-10-19, evaluated **walk-forward**: for each year, every model is refit
    on all targets that end before that year, then predicts it.

  No shuffling. Preprocessing (log1p + standardisation for the GLM) is fitted inside each training window.
* **Candidates.**
  * Baselines: the historical regional mean (training window), and the rolling 4-week and 52-week means.
  * A Poisson GLM (L2 grid).
  * Poisson-loss gradient-boosted trees (two sizes; deterministic, no early stopping).

  Rule: the lowest validation mean Poisson deviance within each family. Trees are chosen over the GLM
  only if they are more than 1 % better (interpretability). The result: trees at M4.5 (2.3 % better on
  validation), the GLM at M5.0 (0.75 %).
* **Uncertainty.** NB2 predictive distribution, Var = μ + αμ², with α fitted by maximum likelihood on
  the validation predictions. Intervals are NB quantiles, and their test coverage is reported.
* **Served model.** The selected configuration is refit on every available week. Its expected quality is
  the walk-forward test result.

### Module C — anomaly detection

For each cell (threshold 4.5 or 5.0) the observed count y in the last k weeks (1/2/4) is compared with
the preceding n weeks (26/52) of the same cell:

* weekly counts are modelled as independent NB2 with an unknown mean μ and dispersion α;
* α is the larger of the reference-window method-of-moments estimate and the cell's 2001–2015
  maximum-likelihood dispersion (`models/anomaly_reference.json`);
* the one-sided p-value is conditional on the total N of both windows: y given N is
  BetaBinomial(N, k/α, n/α), or Binomial(N, k/(n+k)) when α = 0. Conditioning removes the unknown mean
  exactly, so no estimate of μ enters the p-value;
* `expected` (kμ̂) and `reference_sd` (√(k·w + k²·w/n), w = μ̂ + αμ̂²) are reported for context.

Benjamini–Hochberg q-values are computed across all tested cells. A cell is **elevated** when q ≤ FDR
and at least 3 events were observed. Other statuses:

* `not_elevated`;
* `insufficient_reference`: fewer than 5 reference events;
* `data_incomplete`: possible feed gaps, so a normal-looking count is not confirmed as normal.

Only excess activity is tested. The design came from the null simulation:

* With the reference window alone, 35 % of null runs produced at least one false alert.
* The plug-in NB (estimated mean, 2001–2015 moment-dispersion floor) still gave 13.65 %.
* Given the true dispersion but the estimated mean, 11.3 % of runs still had a false alert. The
  estimated mean was the main cause, because quiet reference windows underestimate μ. Conditioning
  removes μ.
* The moment floor (median α 4.4 over the 140 simulated cells, dominated by single megathrust
  sequences) left almost no power. The MLE floor (median 2.4) restores some, and the conditional test
  keeps the false alerts down.

Five alternatives were compared on the same null runs and injection trials: an empirical null at the
95th and 99th percentiles, the MLE floor with and without conditioning, and conditioning with the old
floor (details in [docs/learning-guide.md](docs/learning-guide.md) §8.5). Declustering was not
evaluated, because it would remove the aftershock sequences this module mainly finds.

Isolation Forest was not used: sparse, overdispersed counts need a count model with an explicit
reference, and its scores would not be interpretable as p-values.

### Module D — aftershock analysis

* **Selection rules** (deterministic):
  * great-circle distance ≤ radius (default: Wells & Coppersmith 1994 rupture length, clipped to 30–1000 km);
  * magnitude ≥ the threshold;
  * optional depth limits;
  * a pre window and a post window;
  * the reference event itself is excluded.

  Nearby events are not assumed to be caused by the reference event.
* **Model.** λ(t) = μ + K/(t + c)^p, with μ fixed to the pre-event rate. K, c and p are estimated by
  inhomogeneous-Poisson MLE (L-BFGS-B, nine starts) after the first `fit_start_hours`, because
  catalogues miss early aftershocks. If c is shorter than the data can resolve, it is fixed at its bound
  and only K and p are estimated, and the UI says so.
* **When a fit is attempted and shown.** It needs at least 20 events. It is shown only if all of
  these hold:
  * the optimiser converged;
  * no parameter is at a bound;
  * the Hessian is positive definite (standard errors and 95 % CIs exist);
  * the transformed-time KS test gives p ≥ 0.05;
  * ΔAIC ≥ 2 against a constant-rate model.

  Otherwise the observed data are shown with the reasons. The fitted curve is drawn only inside the fit
  window, with no extrapolation.

## Evaluation results

All numbers below are from `models/*.json` (generated 2026-10-19 by `python ml_pipeline.py all`, data
to 2026-10-19). Full tables: [`docs/ml_evaluation.md`](docs/ml_evaluation.md).

**Forecast, M4.5, final test 2020-01-06 → 2026-10-19 (walk-forward, 46 374 cell-weeks):**

| Model | MAE | RMSE | Mean Poisson deviance | NB log score | 80 % cov. | 95 % cov. |
|---|---:|---:|---:|---:|---:|---:|
| **Gradient boosting (selected)** | **0.933** | 4.644 | **1.985** | **1.091** | 0.958 | 0.985 |
| Poisson GLM | 0.963 | **4.609** | 2.027 | 1.101 | 0.958 | 0.985 |
| Historical mean | 0.966 | 4.787 | 2.304 | 1.129 | 0.952 | 0.982 |
| Rolling 52-week mean | 0.971 | 4.765 | 2.175 | 1.114 | 0.952 | 0.981 |
| Rolling 4-week mean | 1.003 | 5.019 | 4.030 | 2.015 | 0.916 | 0.936 |

* The selected model has 13.9 % lower deviance than the historical mean and 8.7 % lower than the
  52-week mean.
* It is better than the historical mean in every test year.
* The MAE gain is small (0.933 vs 0.966).
* By cell it is mixed: the model has higher deviance than the historical mean in 64 of 131 cells.
  The overall gain comes mostly from active cells with ongoing sequences.

**Forecast, M5.0 (24 072 cell-weeks):**

* The selected Poisson GLM has deviance 1.310 against 1.402 for the historical mean (6.6 % better),
  and a better NB log score (0.740 vs 0.756).
* Its **MAE is slightly worse** (0.553 vs 0.546 for the historical mean and 0.530 for the 52-week mean).
* It is worse than the historical mean in 44 of 68 cells.
* Trees, not selected on validation, were 1.5 % better on test.

**Intervals.** At both thresholds the intervals are **conservative**: 80 % intervals covered about
95–96 % of test weeks. This is because counts are discrete and mostly 0 or 1, and one α is shared by
all cells.

**Sequence clustering (synthetic, 5 seeds; the labels are synthetic, not real events):**

* ARI 1.000 for dense clusters with noise, simultaneous separated clusters, same place at different
  times, antimeridian, high latitude (85°–89.9°), and duplicates/invalid rows (4 invalid and 10
  duplicates removed).
* Plain Euclidean distance on degrees scores ARI 0.814 at the antimeridian and 0.474 at high latitude.
  Ignoring time scores 0 on same place, different times.
* **Two overlapping sequences 60 km apart merge** at the default ε = 50 km (ARI 0). They separate only
  at ε ≈ 15 km (ARI 0.82).

**Anomaly detection:**

* **Null simulation** (2 000 runs × 140 cells, stationary NB counts from 2016–2019, M4.5, 4-week window):
  * 5.6 % of runs had at least one false alert (standard error 0.5 %; nominal 5 %), against 13.65 % for
    the previous plug-in test. On the first 200 runs, identical to the earlier evaluation, the rate fell
    from 12 % to 4.5 %.
  * The false-alert rate per cell test was 0.046 %. The rolling 2× rule's was 10.2 %.
  * Across all 12 threshold/window/reference settings the rate is below the previous test's, but above
    5 % in 7 of them (worst 8.1 %, M4.5 with a 4-week window and 26-week reference).
* **Injected excess** (every unflagged cell of each of the 88 held-out windows, 12 592 trials per
  factor):
  * detection after BH was 0.1 %, 0.4 %, 3.3 % and 15.1 % at +0.5×, +1×, +2× and +4× the expected count
    (previous test: 1.6 % at +2× and 9.1 % at +4×);
  * for comparison, unadjusted p ≤ 0.05 reached 60 % at +4× and the rolling rule 91 %.
  * The detector is conservative and flags only strong excesses.
* **Held-out 2020–2026, M4.5, 4-week windows:** 80 flags in 88 runs (previous test: 70). There is no
  ground truth, so more flags is not evidence of better detection.
* **Missing data:** removing 3 days was always detected by the gap check.

**Omori–Utsu (USGS M ≥ 7.0 2001–2026, M4.5+ events in the rupture-length radius):**

* 356 sequences analysed; 90-day fits: 103 fitted, 43 unreliable, 210 insufficient data. Median p of
  reliable fits 1.32 (IQR 1.09–1.62).
* Out of sample (fit days 0.25–30, predict days 30–90, 103 sequences), median |log ratio| was 0.41
  for Omori–Utsu, 0.94 for the last-week rate and 1.25 for the background rate alone.
* The Poisson 95 % coverage was only 0.58, because parameter uncertainty and secondary sequences are ignored.

**Legacy classifier** (point analysis, unchanged): test ROC-AUC 0.570 against 0.500 for the base rate.
It is exploratory.

## Limitations

* Forecasts are cell-level aggregates. Most of their skill is persistence of ongoing sequences, and they
  cannot anticipate a sequence that has not started. One α for all cells makes intervals conservative
  overall and miscalibrated in some cells.
* Cells are equal-angle (smaller near the poles). Catalogue completeness differs by region and over time.
* Live forecast features use the USGS feed up to the start of the current UTC day. Events posted late
  lower the most recent count.
* DBSCAN uses one density threshold everywhere, merges overlapping sequences at large ε, and cuts
  sequences at the window edges. There is no ground truth for real sequences.
* The anomaly test is conservative (15 % detection at +4× the expected count). Its run-level
  false-alert rate under the null is 5.6 % at the default setting and up to 8.1 % with a 26-week
  reference. The excess comes from cells whose 2016–2019 dispersion far exceeds their 2001–2015
  record. It flags sequences after large earthquakes, by design. It does not test for quiescence.
* Omori–Utsu fits at M4.5+ see few events per sequence. The default radius is a heuristic, and
  overlapping sequences and the exclusion of complex sequences bias the historical evaluation.
* Caches are per process (single-process development server). The analytical history is a ~9 000-event
  in-memory table.

## Testing

```bash
python -m pytest -q
```

```bash
node --test "tests/frontend/*.test.js"
```

* **Offline.** pytest never contacts USGS: `tests/conftest.py` injects a fake transport, with per-query
  handlers for the event lookup and the radius search.
* **Backend coverage** (188 tests):
  * USGS parsing, retries and failures;
  * geometry: antimeridian, poles, centroid, longitude span;
  * the region grid and week-bin boundaries;
  * feature definitions against hand calculations, no look-ahead, and training/serving equality;
  * dispersion MLE recovery, interval and deviance maths;
  * a full walk-forward training run on synthetic data, checking period ordering per fold;
  * artifact loading and refusal of incompatible artifacts;
  * DBSCAN edge cases (antimeridian, high latitude, time separation, ambiguous border events, missing
    magnitudes, stable ids);
  * Benjamini–Hochberg values, the conditional test (exact against the NB convolution, level held for
    any mean), dispersion MLE robustness, insufficient reference, gap handling;
  * Omori–Utsu parameter recovery on simulated sequences, insufficient data, and rejection of
    constant-rate data;
  * the contract of every endpoint, including error codes, caching, truncation and model-unavailable behaviour.
* **Frontend coverage** (26 tests): feed validation and statistics, validation of every analytical
  response, superseded requests, sequence-layer highlighting, explanation texts, comparison rules, and
  SVG chart output (accessibility, empty states, log axes).

### Verification performed (2026-10-19, Windows 11, Python 3.12.10, Node 24.18)

| Check | Result |
|---|---|
| `python -m pytest -q` | 188 passed |
| `node --test "tests/frontend/*.test.js"` | 26 passed |
| `ruff check .` (ruff 0.17.1 from a separate tool venv; config in `ruff.toml`) | no findings |
| `pip-audit -r requirements.txt -r requirements-dev.txt -r requirements-docs.txt` | no known vulnerabilities |
| `python docs/build_guide.py` | all source excerpts verified verbatim; PDF rebuilt |
| `python ml_pipeline.py all` twice | identical metrics (timestamps aside) |
| Live server against real USGS | All views computed real results: 131 forecasts, 145 cells tested (2 elevated: the Panama M7.7 and New Caledonia M6.7 sequences), 60 sequences in 30 days of M2.5+, Omori–Utsu fit for us6000u18k (p = 1.21, 95 % CI 0.92–1.49) |
| Browser (in-app Chromium, 1440×900 and 375×812) | Every view renders without console errors. Selection, highlighting, charts, retry and hand-offs (anomaly → aftershock, sequence → aftershock) work. No horizontal page scroll on mobile. With an empty model directory: forecast shows "unavailable", performance shows "not evaluated yet", anomaly falls back with a notice, live monitor unaffected. |

Not verified: sustained USGS rate limiting (simulated only), memory under long uptime, browsers other
than Chromium, and screen-reader behaviour beyond ARIA roles and labels.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Forecast unavailable: … Missing forecast artifact" | Run `python ml_pipeline.py download`, then `python ml_pipeline.py forecast`. `/api/health` → `modules` shows the reason. |
| "trained with scikit-learn X, running Y" | Install from `requirements.txt`, or retrain. |
| "Historical USGS data could not be loaded" | USGS is unreachable or slow (the history query can take > 15 s). The app retries on the next request; raise `TERRAWATCHER_USGS_TIMEOUT` if needed. |
| A new view looks broken after an update | Static files are versioned by modification time, so a normal reload is enough. |
| Blank map, lists work | The Leaflet CDN or the Esri tiles are blocked. |
| Port in use | `TERRAWATCHER_PORT=5001`. |

## Future work

* Rate-based (per-area) comparisons, and an equal-area grid or tectonic regionalisation.
* Per-cell or covariate-dependent dispersion for better-calibrated forecast intervals.
* ETAS modelling (space-time branching) instead of separate DBSCAN and Omori–Utsu steps.
* Declustering before anomaly testing, to separate background-rate changes from aftershock sequences.
* Magnitude-of-completeness estimation (e.g. maximum curvature) per region and time.
* A shared cache (e.g. Redis) and a production WSGI server for multi-process deployment.

---

Data: USGS Earthquake Hazards Program (public domain). Tiles © Esri. Plate boundaries: Mueller et al. (1987),
as bundled in `data/tectonic_plates.geojson`. The repository has no license file; add one before reuse.
