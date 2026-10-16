import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from flask import Flask, render_template, jsonify, request
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import json
import math
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import time
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'earthquake_model_v1.pth')
SCALER_FILE = os.path.join(BASE_DIR, 'scaler.pkl')
TECTONIC_DATA_PATHS = [
    os.path.join(BASE_DIR, 'data', 'tectonic_plates.geojson'),
    os.path.join(BASE_DIR, 'data', 'tectonic_plates.Geojson')
]

FEATURE_NAMES = [
    'latitude', 'longitude', 'depth',
    'local_density', 'plate_distance'
]

# Configure Device for GPU/MPS Acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
print(f"[DEVICE] Using compute device: {DEVICE}")

# Simple in-memory cache for USGS earthquake requests to respect rate limits
# Key: (days, min_mag) -> (timestamp, data_dict)
USGS_CACHE = {}
USGS_CACHE_TTL = 60  # seconds

# =======================
# Neural Network
# =======================
class EarthquakeNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size is 5
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# =======================
# Model Class
# =======================
class EarthquakeModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.plate_cache = {}
        self.MAX_CACHE_SIZE = 16384

        self.plate_lines = self.load_tectonic_data()
        self.load_model()

    # -----------------------
    # Load tectonic data
    # -----------------------
    def load_tectonic_data(self):
        target_path = None
        for path in TECTONIC_DATA_PATHS:
            if os.path.exists(path):
                target_path = path
                break

        if not target_path:
            print("[WARNING] Tectonic plates file not found. Plate distance will default to 0.")
            return []

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            lines = []
            for feature in data.get('features', []):
                geom = feature.get('geometry') or {}
                coords = geom.get('coordinates') or []
                geom_type = geom.get('type')

                if geom_type == 'MultiLineString':
                    for line in coords:
                        lines.append(line)
                elif geom_type == 'LineString':
                    lines.append(coords)

            print(f"[DATA] Loaded {len(lines)} tectonic boundaries from {os.path.basename(target_path)}")
            return lines

        except Exception as e:
            print(f"[ERROR] Failed to load tectonic data: {e}")
            return []

    # -----------------------
    # Distance calculation
    # -----------------------
    @staticmethod
    def point_to_segment_distance(px, py, x1, y1, x2, y2):
        """
        Calculates distance in degrees between point (px=lon, py=lat) and segment (x1, y1)-(x2, y2).
        Properly handles antimeridian longitude wraparound.
        """
        # Longitude difference wrapped to [-180, 180]
        dx = (x2 - x1 + 180.0) % 360.0 - 180.0
        dpx = (px - x1 + 180.0) % 360.0 - 180.0
        dy = y2 - y1
        dpy = py - y1

        l2 = dx * dx + dy * dy
        if l2 == 0:
            return (dpx * dpx + dpy * dpy) ** 0.5

        t = max(0.0, min(1.0, (dpx * dx + dpy * dy) / l2))

        proj_x = t * dx
        proj_y = t * dy

        return ((dpx - proj_x) ** 2 + (dpy - proj_y) ** 2) ** 0.5

    def compute_plate_distance_single(self, lat, lon):
        if not self.plate_lines:
            return 0.0

        lat = max(-90.0, min(90.0, float(lat)))
        lon = (float(lon) + 180.0) % 360.0 - 180.0

        key = (round(lat, 2), round(lon, 2))
        if key in self.plate_cache:
            return self.plate_cache[key]

        min_dist = float('inf')

        for line in self.plate_lines:
            for i in range(len(line) - 1):
                lon1, lat1 = line[i]
                lon2, lat2 = line[i + 1]

                dist = self.point_to_segment_distance(lon, lat, lon1, lat1, lon2, lat2)
                if dist < min_dist:
                    min_dist = dist

        # Keep cache bounded to prevent unbounded memory growth
        if len(self.plate_cache) >= self.MAX_CACHE_SIZE:
            for k in list(self.plate_cache.keys())[:1000]:
                del self.plate_cache[k]

        self.plate_cache[key] = min_dist
        return min_dist

    def compute_plate_distance(self, df):
        if df.empty or 'latitude' not in df or 'longitude' not in df:
            df['plate_distance'] = []
            return df
        df['plate_distance'] = df.apply(
            lambda r: self.compute_plate_distance_single(r['latitude'], r['longitude']),
            axis=1
        )
        return df

    # -----------------------
    # Feature engineering
    # -----------------------
    def compute_density(self, df, radius=1.0):
        if df.empty or 'latitude' not in df or 'longitude' not in df:
            df['local_density'] = []
            return df

        lats = df['latitude'].values
        lons = df['longitude'].values
        densities = []

        for lat, lon in zip(lats, lons):
            # Vectorized with antimeridian longitude wraparound
            dlon = np.abs((lons - lon + 180.0) % 360.0 - 180.0)
            dlat = np.abs(lats - lat)
            nearby = (dlat <= radius) & (dlon <= radius)
            densities.append(int(nearby.sum()))

        df['local_density'] = densities
        return df

    def get_live_local_density(self, lat, lng, radius=1.0, days=30):
        """Fetches live context for inference, safely clamped to geographic limits."""
        lat = max(-90.0, min(90.0, float(lat)))
        lng = (float(lng) + 180.0) % 360.0 - 180.0

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        min_lat = max(-90.0, lat - radius)
        max_lat = min(90.0, lat + radius)
        min_lon = max(-180.0, lng - radius)
        max_lon = min(180.0, lng + radius)

        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
            f"format=geojson&starttime={start_date.strftime('%Y-%m-%d')}"
            f"&endtime={end_date.strftime('%Y-%m-%d')}"
            f"&minlatitude={min_lat}&maxlatitude={max_lat}"
            f"&minlongitude={min_lon}&maxlongitude={max_lon}"
        )
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                features = response.json().get('features', [])
                return len(features)
        except Exception:
            pass
        return 0

    # -----------------------
    # Load model
    # -----------------------
    def load_model(self):
        """
        Deterministically loads model weights and scaler from disk.
        Does NOT trigger training during import or server startup.
        """
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            try:
                model = EarthquakeNN()
                # Secure load with weights_only=True
                model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True))
                model.to(DEVICE)
                model.eval()

                scaler = joblib.load(SCALER_FILE)
                self.model = model
                self.scaler = scaler

                print(f"[MODEL] Loaded pre-trained model and scaler successfully on {DEVICE}")
                return True
            except Exception as e:
                print(f"[ERROR] Load error: {e}")
                self.model = None
                self.scaler = None
                return False
        else:
            missing = []
            if not os.path.exists(MODEL_FILE):
                missing.append(os.path.basename(MODEL_FILE))
            if not os.path.exists(SCALER_FILE):
                missing.append(os.path.basename(SCALER_FILE))
            print(f"[WARNING] Missing model artifact(s): {', '.join(missing)}. Inference is disabled.")
            print("   Run `python train_model.py` to train and save model artifacts.")
            self.model = None
            self.scaler = None
            return False

    def predict(self, lat, lng, depth=10.0, live_density=None):
        if self.model is None or self.scaler is None:
            return {"error": "Model not loaded"}

        lat = max(-90.0, min(90.0, float(lat)))
        lng = (float(lng) + 180.0) % 360.0 - 180.0
        depth = max(0.0, min(700.0, float(depth)))

        if live_density is None:
            local_density = self.get_live_local_density(lat, lng)
        else:
            local_density = float(live_density)

        plate_distance = self.compute_plate_distance_single(lat, lng)

        x = [[lat, lng, depth, local_density, plate_distance]]
        x = self.scaler.transform(x)

        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            p = float(self.model(x).item())

        return {
            "probability": round(p, 4),
            "risk": "high" if p > 0.7 else "medium" if p > 0.3 else "low",
            "metadata": {
                "local_density_last_30_days": int(local_density),
                "plate_distance": round(plate_distance, 4)
            }
        }


# Initialize globally
model = EarthquakeModel()

# =======================
# FLASK ROUTES
# =======================
@app.route('/')
@app.route('/dashboard-v2')
def index():
    return render_template('index.html')

@app.route('/api/predict')
def predict_api():
    try:
        lat_raw = request.args.get('lat')
        lng_raw = request.args.get('lon') or request.args.get('lng')

        if lat_raw is None or lng_raw is None:
            return jsonify({"error": "Missing required query parameters: 'lat' and 'lon' (or 'lng')"}), 400

        lat = float(lat_raw)
        lng = float(lng_raw)

        if math.isnan(lat) or math.isinf(lat) or math.isnan(lng) or math.isinf(lng):
            return jsonify({"error": "Latitude and longitude must be valid finite numbers"}), 400

        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lng <= 180.0):
            return jsonify({"error": "Latitude must be in [-90, 90] and longitude in [-180, 180]"}), 400

        depth_raw = request.args.get('depth', '10')
        depth = float(depth_raw) if depth_raw else 10.0
        if math.isnan(depth) or math.isinf(depth) or depth < 0.0 or depth > 700.0:
            return jsonify({"error": "Depth must be a finite number between 0 and 700 km"}), 400

    except ValueError as ve:
        return jsonify({"error": f"Invalid input format: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    if model.model is None or model.scaler is None:
        return jsonify({
            "error": "AI model is not loaded on this server. Please check model artifacts.",
            "status": "degraded"
        }), 503

    return jsonify(model.predict(lat, lng, depth))

@app.route('/api/earthquakes')
def earthquakes():
    try:
        days_raw = request.args.get('days', '7')
        try:
            days = int(days_raw)
            if days < 1 or days > 365:
                return jsonify({"error": "Parameter 'days' must be an integer between 1 and 365"}), 400
        except ValueError:
            return jsonify({"error": "Parameter 'days' must be a valid integer"}), 400

        min_mag_raw = request.args.get('min_mag', '2.5')
        try:
            min_mag = float(min_mag_raw)
            if min_mag < -2.0 or min_mag > 10.0:
                return jsonify({"error": "Parameter 'min_mag' must be a float between -2.0 and 10.0"}), 400
        except ValueError:
            return jsonify({"error": "Parameter 'min_mag' must be a valid float"}), 400

        # Check Cache
        cache_key = (days, min_mag)
        now = time.time()
        if cache_key in USGS_CACHE:
            cached_time, cached_data = USGS_CACHE[cache_key]
            if now - cached_time < USGS_CACHE_TTL:
                return jsonify(cached_data)

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
            f"format=geojson&starttime={start_date.strftime('%Y-%m-%d')}"
            f"&endtime={end_date.strftime('%Y-%m-%d')}"
            f"&minmagnitude={min_mag}&limit=300"
        )
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        data = res.json()

        features = data.get('features', [])
        if features:
            coords_list = []
            valid_indices = []

            for i, f in enumerate(features):
                geom = f.get('geometry') or {}
                coords = geom.get('coordinates') or []
                props = f.get('properties') or {}

                lon = float(coords[0]) if len(coords) > 0 and coords[0] is not None else 0.0
                lat = float(coords[1]) if len(coords) > 1 and coords[1] is not None else 0.0
                depth = float(coords[2]) if len(coords) > 2 and coords[2] is not None else 10.0

                # Ensure depth is stored in props for frontend display
                props['depth'] = depth
                coords_list.append([lon, lat, depth])
                valid_indices.append(i)

            df = pd.DataFrame(coords_list, columns=['longitude', 'latitude', 'depth'])
            df = model.compute_density(df, radius=1.0)
            densities = df['local_density'].tolist()

            has_model = (model.model is not None and model.scaler is not None)

            for idx, feature_idx in enumerate(valid_indices):
                feature = features[feature_idx]
                props = feature['properties']
                coords = coords_list[idx]

                if has_model:
                    pred = model.predict(lat=coords[1], lng=coords[0], depth=coords[2], live_density=densities[idx])
                    props['prediction'] = pred.get('probability', 0.0)
                else:
                    props['prediction'] = None

        # Store in cache
        USGS_CACHE[cache_key] = (now, data)
        return jsonify(data)

    except requests.exceptions.Timeout:
        return jsonify({"error": "USGS API request timed out. Please try again."}), 504
    except requests.exceptions.RequestException as re:
        return jsonify({"error": f"Failed to fetch data from USGS: {re}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/hotspots')
def hotspots():
    zones = [
        (36.1, -120.2), # San Andreas
        (35.6, 140.1),  # Japan
        (-33.4, -70.6), # Chile
        (39.2, 40.0),   # Anatolian
        (-3.0, 100.0),  # Sumatra
        (41.9, 12.5),   # Italy
        (-41.2, 174.7)  # New Zealand
    ]

    has_model = (model.model is not None and model.scaler is not None)
    results = []

    for lat, lng in zones:
        if has_model:
            pred = model.predict(lat, lng, 10.0, live_density=5)
            prob = pred.get("probability", 0.0)
        else:
            prob = 0.5  # Neutral fallback

        results.append({
            "latitude": lat,
            "longitude": lng,
            "probability": prob
        })

    return jsonify({
        "status": "success" if has_model else "degraded",
        "hotspots": results
    })

@app.route('/api/model-info')
def model_info():
    is_loaded = (model.model is not None and model.scaler is not None)
    return jsonify({
        "features": FEATURE_NAMES,
        "model": "PyTorch Sequential NN",
        "device": str(DEVICE),
        "status": "loaded" if is_loaded else "unloaded",
        "version": "v1.1",
        "epochs": 60
    })

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    app.run(debug=debug, host=host, port=port)