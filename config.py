"""
Central configuration for TerraWatcher.

Every value can be overridden with an environment variable so the same code
runs on any machine without editing source files. No secrets are required:
the USGS FDSN event service is public and unauthenticated.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


# --- Server -----------------------------------------------------------------
HOST = os.environ.get("TERRAWATCHER_HOST", "127.0.0.1")
PORT = _env_int("TERRAWATCHER_PORT", 5000)
DEBUG = os.environ.get("TERRAWATCHER_DEBUG", "0").lower() in ("1", "true", "yes")
LOG_LEVEL = os.environ.get("TERRAWATCHER_LOG_LEVEL", "INFO").upper()

# --- Artifacts --------------------------------------------------------------
MODEL_FILE = os.path.join(BASE_DIR, "earthquake_model_v1.pth")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_META_FILE = os.path.join(BASE_DIR, "model_meta.json")
TECTONIC_FILE = os.path.join(BASE_DIR, "data", "tectonic_plates.geojson")

# --- Upstream: USGS FDSN event web service -----------------------------------
USGS_BASE_URL = os.environ.get(
    "TERRAWATCHER_USGS_URL", "https://earthquake.usgs.gov/fdsnws/event/1"
)
USGS_TIMEOUT = _env_float("TERRAWATCHER_USGS_TIMEOUT", 15)   # seconds per request
USGS_RETRIES = _env_int("TERRAWATCHER_USGS_RETRIES", 1)      # retries on 429/5xx/timeouts

# --- Event feed limits --------------------------------------------------------
MAX_DAYS = 30            # largest observation window the API accepts
MAX_EVENTS = 2000        # events requested from USGS per feed query
FEED_CACHE_TTL = _env_int("TERRAWATCHER_FEED_CACHE_TTL", 60)        # seconds
CONTEXT_CACHE_TTL = _env_int("TERRAWATCHER_CONTEXT_CACHE_TTL", 600)  # seconds
STALE_MAX_AGE = _env_int("TERRAWATCHER_STALE_MAX_AGE", 1800)        # serve stale copy up to this age
CACHE_MAX_ENTRIES = 32
