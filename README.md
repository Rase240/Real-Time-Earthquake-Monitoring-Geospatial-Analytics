# 🌍 Earthquake Forecaster — Machine Learning Powered Seismic Risk Dashboard

A modern, data-driven earthquake visualization and risk assessment platform that queries real-time seismic activity from the USGS API, overlays global tectonic plate boundaries, and estimates earthquake risk using a PyTorch neural network.

---

## 🚀 Project Overview

The system provides:
* 📊 **Live Earthquake Feed**: Real-time seismic visualization on an interactive Leaflet dark-mode map.
* 🤖 **AI Risk Assessment**: Deep neural network evaluating location, depth, local density, and tectonic distance.
* 🗺️ **Tectonic Fault Overlay**: 241 global plate boundary segments with antimeridian-safe distance computation.
* ⚡ **High-Performance Caching**: In-memory caching for USGS queries to prevent upstream rate-limiting.
* 🧪 **Comprehensive Regression Tests**: Automated pytest suite verifying API contracts and spatial algorithms.

---

## 🏗️ Architecture & Tech Stack

### Frontend
* HTML5 / Vanilla CSS3 (modern glassmorphism, responsive sidebar & KPI cards)
* Interactive Maps via Leaflet 1.9.4 and ESRI World Dark Gray Canvas tiles
* DM Mono & Manrope typography

### Backend & Machine Learning
* Python (Flask web application)
* PyTorch (Multilayer Perceptron for seismic classification)
* Scikit-Learn (`StandardScaler`) & Joblib
* Pandas & NumPy for vectorized spatial processing
* Upstream Data: USGS FDSNWS Event Web Service

---

## 📂 Project Structure

```text
Earthquake-Forecaster/
├── app.py                     # Flask web server & prediction endpoints
├── train_model.py             # Dedicated offline training pipeline
├── earthquake_model_v1.pth    # Serialized PyTorch model weights
├── scaler.pkl                 # Feature scaler artifact
├── requirements.txt           # Python dependencies (UTF-8)
├── pytest.ini                 # Pytest configuration
├── data/
│   └── tectonic_plates.geojson # Global plate boundaries GeoJSON
├── dataset_fetcher.py         # Offline dataset gathering utility
├── static/
│   ├── dashboard.js           # Client map logic, KPI calculations, and UI interaction
│   └── dashboard.css          # Modern dark-mode dashboard styling
├── templates/
│   └── index.html             # Modern Seismic Intelligence dashboard template
└── tests/
    └── test_application.py    # Regression test suite (19 tests)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Rase240/Earthquake-Forecaster.git
cd Earthquake-Forecaster
```

### 2️⃣ Create Virtual Environment
```bash
# On Windows
python -m venv .venv
.\.venv\Scripts\activate

# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Regression Tests
```bash
pytest
```

### 5️⃣ Start the Application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

#### Optional Environment Variables
- `FLASK_PORT=5000` (specify custom port)
- `FLASK_HOST=127.0.0.1` (specify binding address)
- `FLASK_DEBUG=1` (enable debug mode for development)

---

## 🧠 Model Training (Optional)

The repository includes pre-trained model weights (`earthquake_model_v1.pth`) and scaler (`scaler.pkl`). To retrain the model on the latest 6 months of historical USGS earthquake data:

```bash
python train_model.py
```

---

## 📡 API Endpoints

| Endpoint | Method | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | None | Renders the main dashboard UI. |
| `/api/earthquakes` | `GET` | `days` (1-365), `min_mag` (-2.0-10.0) | Fetches and AI-grades earthquakes from USGS. |
| `/api/predict` | `GET` | `lat`, `lon` (or `lng`), `depth` (optional) | Returns AI risk assessment for given coordinates. |
| `/api/hotspots` | `GET` | None | Returns 7 active tectonic zones with risk scores. |
| `/api/model-info` | `GET` | None | Returns model architecture, features, and status. |
