"""
Model definition, artifact loading and batch inference.

Artifacts (all produced by train_model.py):
  earthquake_model_v1.pth  PyTorch state_dict for EarthquakeNN
  scaler.pkl               sklearn StandardScaler fitted on the training split
  model_meta.json          feature names, label definition, base rate, metrics

Loading is all-or-nothing: if any artifact is missing or inconsistent with the
current feature schema, the bundle reports status "unavailable" together with
the reason, and inference is refused. No placeholder scores are produced.
"""
import json
import logging
import os

import joblib
import numpy as np
import torch
import torch.nn as nn

from features import FEATURE_NAMES

log = logging.getLogger("terrawatcher.model")


class EarthquakeNN(nn.Module):
    """Small fully-connected binary classifier: 5 inputs -> P(significant)."""

    def __init__(self, n_features=None):
        super().__init__()
        n_features = n_features or len(FEATURE_NAMES)
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ModelBundle:
    def __init__(self, model_path, scaler_path, meta_path):
        self.model = None
        self.scaler = None
        self.meta = {}
        self.load_error = None
        self._load(model_path, scaler_path, meta_path)

    @property
    def ready(self):
        return self.model is not None and self.scaler is not None

    @property
    def base_rate(self):
        return self.meta.get("label", {}).get("train_positive_rate")

    def _load(self, model_path, scaler_path, meta_path):
        missing = [os.path.basename(p) for p in (model_path, scaler_path, meta_path) if not os.path.exists(p)]
        if missing:
            self.load_error = f"Missing model artifact(s): {', '.join(missing)}. Run `python train_model.py`."
            log.warning(self.load_error)
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("feature_names") != FEATURE_NAMES:
                raise ValueError(
                    f"Feature schema mismatch: artifact has {meta.get('feature_names')}, code expects {FEATURE_NAMES}"
                )
            scaler = joblib.load(scaler_path)
            if getattr(scaler, "n_features_in_", None) != len(FEATURE_NAMES):
                raise ValueError("Scaler was fitted on a different number of features")
            model = EarthquakeNN()
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.eval()   # BatchNorm must use running statistics for inference
        except Exception as exc:  # noqa: BLE001 - any failure means "do not serve predictions"
            self.load_error = f"Model artifacts could not be loaded: {exc}"
            log.error(self.load_error)
            return
        self.model, self.scaler, self.meta = model, scaler, meta
        log.info("Model loaded (trained %s, %s features)", meta.get("trained_at"), len(FEATURE_NAMES))

    def predict_proba(self, X):
        """X: (n, 5) array in FEATURE_NAMES order -> (n,) probabilities."""
        if not self.ready:
            raise RuntimeError(self.load_error or "Model not loaded")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Expected shape (n, {len(FEATURE_NAMES)}), got {X.shape}")
        Xs = self.scaler.transform(X)
        with torch.no_grad():
            p = self.model(torch.tensor(Xs, dtype=torch.float32)).numpy().reshape(-1)
        if not np.all(np.isfinite(p)):
            raise ValueError("Model produced non-finite output")
        return np.clip(p, 0.0, 1.0)

    def relative_band(self, p):
        """Compare a probability with the training base rate (not an absolute 'risk')."""
        base = self.base_rate
        if not base:
            return None, None
        ratio = float(p) / base
        band = "above_base_rate" if ratio >= 1.25 else "below_base_rate" if ratio <= 0.8 else "near_base_rate"
        return round(ratio, 3), band

    def describe(self):
        info = {
            "status": "loaded" if self.ready else "unavailable",
            "load_error": self.load_error,
            "features": FEATURE_NAMES,
            "architecture": "MLP 5-32-64-32-1 (ReLU, BatchNorm1d, sigmoid output)",
        }
        if self.ready:
            for key in ("trained_at", "label", "data", "split", "training", "metrics", "baselines", "interpretation"):
                if key in self.meta:
                    info[key] = self.meta[key]
        return info
