"""
Comprehensive Automated Test Suite for Earthquake-Forecaster
Tests model loading, spatial geometry, antimeridian handling, API contracts,
error handling, and edge cases with mocked USGS services.
"""
import os
import json
import math
import pytest
from unittest.mock import patch, MagicMock

import torch
import numpy as np
import pandas as pd

import app
from app import app as flask_app, EarthquakeModel, EarthquakeNN, FEATURE_NAMES


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def model_mgr():
    return EarthquakeModel()


# ==============================================================================
# 1. Model Loading & Degradation Tests
# ==============================================================================
def test_model_and_scaler_load_successfully(model_mgr):
    """Verifies that the restored scaler and existing model weights load cleanly."""
    assert model_mgr.model is not None, "Neural network model should be loaded"
    assert model_mgr.scaler is not None, "StandardScaler should be loaded"
    assert isinstance(model_mgr.model, EarthquakeNN)
    assert model_mgr.scaler.n_features_in_ == len(FEATURE_NAMES)


def test_missing_model_files_behavior():
    """Verifies that missing model/scaler degrades gracefully without attempting training."""
    with patch('os.path.exists', return_value=False):
        mgr = EarthquakeModel()
        assert mgr.model is None
        assert mgr.scaler is None
        # Predict should return error dict
        pred = mgr.predict(35.0, -120.0)
        assert "error" in pred


def test_model_eval_mode(model_mgr):
    """Ensures model is set to eval mode (crucial for BatchNorm1d single-sample inference)."""
    if model_mgr.model is not None:
        assert not model_mgr.model.training, "Model must be in eval() mode for inference"


# ==============================================================================
# 2. Geospatial & Antimeridian Calculation Tests
# ==============================================================================
def test_antimeridian_wraparound_distance():
    """
    Verifies that points across the antimeridian (180°/-180°) calculate true ~0.2° distance
    instead of ~359.8° distance (DEF-07).
    """
    # Point at lon=179.9, lat=52.0
    # Segment from (-179.9, 51.9) to (-179.9, 52.1)
    dist = EarthquakeModel.point_to_segment_distance(179.9, 52.0, -179.9, 51.9, -179.9, 52.1)
    assert dist < 1.0, f"Distance across antimeridian should be ~0.2 deg, got {dist}"
    assert math.isclose(dist, 0.2, abs_tol=0.01)


def test_normal_segment_distance():
    """Verifies standard Cartesian distance calculation on non-wrapping segments."""
    # Point at (0, 1), segment from (0, 0) to (2, 0) -> closest point (0, 0), distance 1.0
    dist = EarthquakeModel.point_to_segment_distance(0.0, 1.0, 0.0, 0.0, 2.0, 0.0)
    assert math.isclose(dist, 1.0, abs_tol=1e-5)


def test_polar_latitude_density_clamping(model_mgr):
    """
    Verifies that querying live density at lat=90 or lat=-90 clamps to valid USGS bounds [-90, 90]
    preventing HTTP 400 errors from USGS (DEF-08).
    """
    with patch('requests.get') as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"features": []}
        mock_get.return_value = mock_resp

        # At North Pole
        count = model_mgr.get_live_local_density(90.0, 0.0, radius=1.0)
        assert count == 0
        called_url = mock_get.call_args[0][0]
        assert "maxlatitude=90.0" in called_url or "maxlatitude=90" in called_url
        assert "minlatitude=89.0" in called_url or "minlatitude=89" in called_url


# ==============================================================================
# 3. Model Inference Tests
# ==============================================================================
def test_predict_returns_valid_probabilities(model_mgr):
    """Verifies that model inference yields valid probabilities [0, 1] and risk categories."""
    if model_mgr.model is None or model_mgr.scaler is None:
        pytest.skip("Model artifacts not present")

    # San Andreas coordinates
    res = model_mgr.predict(lat=36.0, lng=-120.0, depth=10.0, live_density=5)
    assert "probability" in res
    assert 0.0 <= res["probability"] <= 1.0
    assert res["risk"] in ("low", "medium", "high")
    assert "metadata" in res
    assert "plate_distance" in res["metadata"]


# ==============================================================================
# 4. API Endpoint Contract Tests
# ==============================================================================
def test_index_route(client):
    """Verifies that the index route loads the modern dashboard index.html successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"SEISMIC" in response.data
    assert b"dashboard.css" in response.data
    assert b"dashboard.js" in response.data


def test_dashboard_v2_route(client):
    """Verifies that /dashboard-v2 also resolves to index.html."""
    response = client.get('/dashboard-v2')
    assert response.status_code == 200
    assert b"SEISMIC" in response.data
    assert b"dashboard.css" in response.data
    assert b"dashboard.js" in response.data


def test_predict_api_valid_input(client):
    """Verifies /api/predict with valid lat/lon."""
    response = client.get('/api/predict?lat=37.7749&lon=-122.4194')
    assert response.status_code == 200
    data = response.get_json()
    assert "probability" in data
    assert "risk" in data


def test_predict_api_missing_parameters(client):
    """Verifies /api/predict returns 400 Bad Request when coordinates are missing."""
    response = client.get('/api/predict?lat=37.7749')
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_api_invalid_non_numeric(client):
    """Verifies /api/predict returns 400 Bad Request on non-numeric input."""
    response = client.get('/api/predict?lat=invalid&lon=abc')
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_api_nan_inf_handling(client):
    """Verifies /api/predict returns 400 on NaN and Inf inputs."""
    response = client.get('/api/predict?lat=nan&lon=inf')
    assert response.status_code == 400


def test_predict_api_out_of_bounds(client):
    """Verifies /api/predict returns 400 on latitude > 90 or longitude > 180."""
    response = client.get('/api/predict?lat=95.0&lon=-120.0')
    assert response.status_code == 400


def test_earthquakes_api_input_validation(client):
    """Verifies /api/earthquakes validates query parameters cleanly."""
    # Invalid days
    res1 = client.get('/api/earthquakes?days=-5')
    assert res1.status_code == 400
    res2 = client.get('/api/earthquakes?days=abc')
    assert res2.status_code == 400

    # Invalid min_mag
    res3 = client.get('/api/earthquakes?min_mag=99')
    assert res3.status_code == 400
    res4 = client.get('/api/earthquakes?min_mag=abc')
    assert res4.status_code == 400


def test_earthquakes_api_populates_depth_and_handles_2d_coords(client):
    """
    Verifies that /api/earthquakes copies depth into feature properties (DEF-04)
    and handles both 2D and 3D coordinate arrays without crashing (DEF-06).
    """
    mock_usgs_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"mag": 4.5, "place": "Test 3D", "time": 1700000000000},
                "geometry": {"type": "Point", "coordinates": [-120.5, 36.5, 12.4]}
            },
            {
                "type": "Feature",
                "properties": {"mag": None, "place": "Test 2D null mag", "time": 1700000000000},
                "geometry": {"type": "Point", "coordinates": [-121.0, 37.0]}
            }
        ]
    }

    with patch('requests.get') as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_usgs_payload
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Clear cache first to force live processing
        app.USGS_CACHE.clear()

        response = client.get('/api/earthquakes?days=7&min_mag=2.5')
        assert response.status_code == 200
        data = response.get_json()
        features = data["features"]
        assert len(features) == 2

        # First feature (3D): depth should be populated in properties
        feat1 = features[0]
        assert feat1["properties"]["depth"] == 12.4
        assert "prediction" in feat1["properties"]

        # Second feature (2D): depth should fallback to 10.0 without throwing IndexError
        feat2 = features[1]
        assert feat2["properties"]["depth"] == 10.0
        assert "prediction" in feat2["properties"]


def test_hotspots_api(client):
    """Verifies /api/hotspots returns the 7 active tectonic zones with probabilities."""
    response = client.get('/api/hotspots')
    assert response.status_code == 200
    data = response.get_json()
    assert "hotspots" in data
    assert len(data["hotspots"]) == 7
    for spot in data["hotspots"]:
        assert "latitude" in spot
        assert "longitude" in spot
        assert "probability" in spot


def test_model_info_api(client):
    """Verifies /api/model-info outputs correct metadata schema."""
    response = client.get('/api/model-info')
    assert response.status_code == 200
    data = response.get_json()
    assert data["features"] == FEATURE_NAMES
    assert "device" in data
    assert data["status"] in ("loaded", "unloaded")


# ==============================================================================
# 5. Tectonic Plates File Casing & Parsing Tests
# ==============================================================================
def test_tectonic_plates_loading(model_mgr):
    """Verifies tectonic plates loaded all boundaries from the case-corrected file."""
    assert len(model_mgr.plate_lines) > 200, "Should load 241 tectonic boundaries"
