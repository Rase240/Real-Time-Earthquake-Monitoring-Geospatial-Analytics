"""Model artifacts: loading, schema validation and inference behaviour."""
import json
import shutil

import numpy as np
import pytest

import config
from features import FEATURE_NAMES
from ml_model import EarthquakeNN, ModelBundle


@pytest.fixture(scope="module")
def bundle():
    b = ModelBundle(config.MODEL_FILE, config.SCALER_FILE, config.MODEL_META_FILE)
    if not b.ready:
        pytest.fail(f"Committed model artifacts must load: {b.load_error}")
    return b


def test_committed_artifacts_load_and_match_schema(bundle):
    assert isinstance(bundle.model, EarthquakeNN)
    assert not bundle.model.training, "BatchNorm requires eval() mode for single-sample inference"
    assert bundle.scaler.n_features_in_ == len(FEATURE_NAMES)
    assert bundle.meta["feature_names"] == FEATURE_NAMES


def test_metadata_documents_evaluation(bundle):
    meta = bundle.meta
    assert meta["split"]["method"].startswith("chronological")
    assert meta["split"]["train"]["to"] <= meta["split"]["test"]["from"]
    for key in ("roc_auc", "brier", "log_loss", "average_precision"):
        assert key in meta["metrics"]["test"]
    assert "Constant train base rate" in meta["baselines"]
    assert 0 < bundle.base_rate < 1


def test_predictions_are_probabilities_and_deterministic(bundle):
    X = np.array([[35.6, 140.1, 30.0, 6, 98.0], [-20.0, -175.0, 100.0, 0, 20.0], [0.0, 0.0, 10.0, 0, 1500.0]])
    p1, p2 = bundle.predict_proba(X), bundle.predict_proba(X)
    assert p1.shape == (3,)
    assert np.all((p1 >= 0) & (p1 <= 1))
    assert np.array_equal(p1, p2)


def test_single_row_matches_batch(bundle):
    X = np.array([[35.6, 140.1, 30.0, 6, 98.0], [-20.0, -175.0, 100.0, 0, 20.0]])
    batch = bundle.predict_proba(X)
    single = np.array([bundle.predict_proba(X[i:i + 1])[0] for i in range(2)])
    assert np.allclose(batch, single, atol=1e-6)


def test_wrong_shape_is_rejected(bundle):
    with pytest.raises(ValueError):
        bundle.predict_proba(np.zeros((1, 4)))


def test_relative_band(bundle):
    base = bundle.base_rate
    assert bundle.relative_band(base * 2)[1] == "above_base_rate"
    assert bundle.relative_band(base)[1] == "near_base_rate"
    assert bundle.relative_band(base * 0.5)[1] == "below_base_rate"


def test_missing_artifacts_disable_inference(tmp_path):
    b = ModelBundle(str(tmp_path / "m.pth"), str(tmp_path / "s.pkl"), str(tmp_path / "meta.json"))
    assert not b.ready and "Missing model artifact" in b.load_error
    assert b.describe()["status"] == "unavailable"
    with pytest.raises(RuntimeError):
        b.predict_proba(np.zeros((1, 5)))


def _copy_artifacts(tmp_path):
    paths = {}
    for name, src in (("m.pth", config.MODEL_FILE), ("s.pkl", config.SCALER_FILE), ("meta.json", config.MODEL_META_FILE)):
        paths[name] = str(tmp_path / name)
        shutil.copy(src, paths[name])
    return paths


def test_feature_schema_mismatch_is_refused(tmp_path):
    p = _copy_artifacts(tmp_path)
    meta = json.load(open(p["meta.json"], encoding="utf-8"))
    meta["feature_names"] = ["latitude", "longitude", "depth", "local_density", "plate_distance"]
    json.dump(meta, open(p["meta.json"], "w", encoding="utf-8"))
    b = ModelBundle(p["m.pth"], p["s.pkl"], p["meta.json"])
    assert not b.ready and "schema mismatch" in b.load_error


def test_corrupt_weights_are_refused(tmp_path):
    p = _copy_artifacts(tmp_path)
    open(p["m.pth"], "wb").write(b"not a torch file")
    b = ModelBundle(p["m.pth"], p["s.pkl"], p["meta.json"])
    assert not b.ready and "could not be loaded" in b.load_error
