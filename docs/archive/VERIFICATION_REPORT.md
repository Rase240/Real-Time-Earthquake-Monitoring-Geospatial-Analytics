# Verification & Operational Audit Report: Earthquake-Forecaster

**Date:** October 17, 2026  
**Target Branch:** `audit-repair`  
**Repository:** `https://github.com/Rase240/Earthquake-Forecaster`  
**Auditor / Verification Agent:** Senior Python Engineer & Antigravity IDE  

---

## 1. Executive Verdict

### **VERDICT: PASS**

The application starts without warnings or errors, loads the neural network model and standard scaler cleanly on CPU, parses all 241 global tectonic plate boundaries, connects successfully to the live USGS earthquake feed, and serves the modernized Seismic Intelligence dashboard. All 19 automated regression tests pass (100%), and end-to-end HTTP smoke tests confirm all API endpoints and error handlers return expected status codes and payloads.

---

## 2. Environment & Runtime Specifications

| Attribute | Value / Specification |
| :--- | :--- |
| **Operating System** | Windows 11 (AMD64) |
| **Python Version** | Python 3.12.10 (`.venv\Scripts\python.exe`) |
| **Virtual Environment** | Dedicated local virtualenv at `.venv` |
| **Core Dependencies** | `torch==2.5.1+cpu`, `scikit-learn==1.7.2`, `flask==3.1.3`, `pandas==2.2.3`, `numpy==2.2.3`, `requests==2.32.3`, `joblib==1.4.2` |
| **Test Runner** | `pytest==8.4.2`, `pytest-mock` |
| **Startup Command** | `.\.venv\Scripts\python.exe app.py` |
| **Execution URL** | `http://127.0.0.1:5000/` |

---

## 3. Artifact & Component Integrity Check

| Component / Artifact | File Location | Status | Details |
| :--- | :--- | :--- | :--- |
| **Model Weights** | `earthquake_model_v1.pth` | **LOADED** | PyTorch sequential MLP (5 &rarr; 64 &rarr; 32 &rarr; 16 &rarr; 1). Set to `eval()` mode. |
| **StandardScaler** | `scaler.pkl` | **LOADED** | Restored from Git history and re-serialized with scikit-learn 1.7.2 (0 unpickle warnings). |
| **Tectonic Plates GeoJSON**| `data/tectonic_plates.geojson`| **LOADED** | 241 fault boundary line segments loaded with antimeridian-safe distance algorithm. |
| **Offline Training Pipeline**| `train_model.py` | **OPERATIONAL** | Decoupled training pipeline with USGS historical chunked fetching. |
| **Dataset Fetcher** | `dataset_fetcher.py` | **OPERATIONAL** | Standardized offline earthquake CSV download utility. |
| **Frontend Dashboard** | `templates/index.html` | **OPERATIONAL** | Modern glassmorphic seismic dashboard (Leaflet 1.9.4 + ESRI Dark Canvas). |
| **Client Assets** | `static/dashboard.css`, `static/dashboard.js` | **OPERATIONAL** | Modern styles & map/KPI client logic. |

---

## 4. Test Suite Execution & Results

### Automated Tests (`pytest -v`)
Executed command: `.\.venv\Scripts\pytest.exe -v`

| Test Name | Component Tested | Result | Duration |
| :--- | :--- | :---: | :---: |
| `test_model_and_scaler_load_successfully` | Model & Scaler Artifact Loading | **PASSED** | 0.05s |
| `test_missing_model_files_behavior` | Graceful Degradation on Missing Artifacts | **PASSED** | 0.02s |
| `test_model_eval_mode` | BatchNorm1d Inference Mode Validation | **PASSED** | 0.01s |
| `test_antimeridian_wraparound_distance` | Antimeridian (180°/-180°) Wraparound Math | **PASSED** | 0.01s |
| `test_normal_segment_distance` | Cartesian Geometry Segment Distance | **PASSED** | 0.01s |
| `test_polar_latitude_density_clamping` | Polar Latitude USGS Bounds [-90, 90] | **PASSED** | 0.02s |
| `test_predict_returns_valid_probabilities`| AI Model Risk & Probability Output | **PASSED** | 0.03s |
| `test_index_route` | Root Route `/` Template & Asset Verification | **PASSED** | 0.02s |
| `test_dashboard_v2_route` | Alias Route `/dashboard-v2` Verification | **PASSED** | 0.02s |
| `test_predict_api_valid_input` | Endpoint `/api/predict` (Valid Coordinates) | **PASSED** | 0.02s |
| `test_predict_api_missing_parameters` | Input Validation: Missing `lat`/`lon` (HTTP 400) | **PASSED** | 0.02s |
| `test_predict_api_invalid_non_numeric` | Input Validation: Non-numeric Strings (HTTP 400)| **PASSED** | 0.02s |
| `test_predict_api_nan_inf_handling` | Input Validation: NaN and Infinity (HTTP 400) | **PASSED** | 0.02s |
| `test_predict_api_out_of_bounds` | Input Validation: Lat/Lon Range Check (HTTP 400)| **PASSED** | 0.02s |
| `test_earthquakes_api_input_validation`| Endpoint `/api/earthquakes` Input Bounds | **PASSED** | 0.02s |
| `test_earthquakes_api_populates_depth_and_handles_2d_coords` | GeoJSON 2D/3D Depth Parsing & Feature Grading | **PASSED** | 0.03s |
| `test_hotspots_api` | Endpoint `/api/hotspots` (7 Tectonic Zones) | **PASSED** | 0.02s |
| `test_model_info_api` | Endpoint `/api/model-info` Metadata & Schema | **PASSED** | 0.02s |
| `test_tectonic_plates_loading` | Global Tectonic Boundaries Loading (241 lines) | **PASSED** | 0.01s |
| **TOTAL** | **19 test cases** | **19 PASSED** | **4.08s** |

---

## 5. Live HTTP Endpoint Smoke Tests

Executed against running Flask server at `http://127.0.0.1:5000`:

| Endpoint / URL | Method | Expected Status | Actual Status | Payload Size | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `/` | GET | 200 | **200 OK** | 11,616 bytes | Serves modern `index.html` |
| `/dashboard-v2` | GET | 200 | **200 OK** | 11,616 bytes | Preserved alias route |
| `/static/dashboard.css` | GET | 200 | **200 OK** | 23,488 bytes | Clean modern styles |
| `/static/dashboard.js` | GET | 200 | **200 OK** | 20,246 bytes | Interactive client logic |
| `/api/model-info` | GET | 200 | **200 OK** | 173 bytes | Status: `"loaded"`, PyTorch CPU |
| `/api/hotspots` | GET | 200 | **200 OK** | 434 bytes | 7 zones with model probabilities |
| `/api/earthquakes?days=1&min_mag=4.5`| GET | 200 | **200 OK** | 9,284 bytes | Live USGS GeoJSON with depth & AI grades |
| `/api/predict?lat=35.0&lon=-120.0&depth=10`| GET | 200 | **200 OK** | 105 bytes | Risk: `low`, Prob: `0.029` |
| `/api/predict?lat=foo` | GET | 400 | **400 Bad Request** | 56 bytes | Proper validation error message |
| `/api/earthquakes?days=-5` | GET | 400 | **400 Bad Request** | 73 bytes | Proper parameter validation |

---

## 6. Files Changed & Summary of Modifications

| File Path | Nature of Change | Description & Rationale |
| :--- | :--- | :--- |
| `app.py` | **Modified** | Fixed polar density bounds, antimeridian wraparound distance, input validation, USGS GeoJSON 2D/3D depth bug, decoupled model training, and consolidated routes. |
| `scaler.pkl` | **Restored & Reserialized** | Restored missing scaler required for inference; re-pickled using scikit-learn 1.7.2. |
| `data/tectonic_plates.geojson` | **Renamed** | Standardized from `tectonic_plates.Geojson` to `tectonic_plates.geojson` for cross-platform Linux/Docker safety. |
| `templates/index.html` | **Replaced** | Upgraded from legacy layout to modern glassmorphic dashboard with ESRI dark canvas maps. |
| `static/dashboard.css` | **Added** | Design system tokens, responsive sidebar, glassmorphic cards, and Leaflet overrides. |
| `static/dashboard.js` | **Added** | Client state machine, USGS feed loader, map rendering, KPI calculator, and hotspot toggles. |
| `static/styles.css`, `static/script.js` | **Deleted** | Removed legacy dashboard files. |
| `dataset_fetcher.py` | **Renamed** | Renamed from `datasetfetcher.py` to follow PEP 8 snake_case module naming. |
| `train_model.py` | **Added** | Standalone offline training pipeline so web startup never triggers model training. |
| `requirements.txt` | **Modified** | Re-saved cleanly as UTF-8 without byte-order marks; pinned compatible dependencies. |
| `pytest.ini` | **Added** | Configured pytest test paths and warning filters. |
| `tests/test_application.py` | **Added** | Comprehensive 19-test automated regression suite. |
| `.gitignore` | **Updated** | Comprehensive rules for Python, virtualenv, IDEs, OS files, and dynamic data. |
| `README.md` | **Updated** | Updated architecture, project structure, requirements, and setup instructions. |

---

## 7. Remaining Limitations & Environmental Context

1. **Upstream USGS Dependency:** Live earthquake fetching requires outbound Internet connectivity to `https://earthquake.usgs.gov`. If network connectivity is lost, the application degrades gracefully: cached results are served within TTL, and clear HTTP 502/504 errors are returned rather than unhandled 500 crashes.
2. **Local Machine Compute:** Inference runs on CPU (`torch.device('cpu')`), which executes in under 2 ms per request without requiring CUDA hardware.

---

## 8. Instructions for Running Locally

```bash
# 1. Clone or navigate to the repository
cd Earthquake-Forecaster

# 2. Activate virtual environment (or use python directly)
.\.venv\Scripts\python.exe app.py

# 3. Open your browser
http://127.0.0.1:5000/

# 4. Run automated test suite
.\.venv\Scripts\pytest.exe -v
```
