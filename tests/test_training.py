"""
Training pipeline correctness on small synthetic catalogues.

The synthetic rows below are unit-test fixtures for checking code paths
(splitting, leakage, metrics); they are never used to train the served model.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import config
import features as F
import train_model as T

DAY = F.DAY_MS


@pytest.fixture(scope="module")
def segments():
    return F.load_plate_segments(config.TECTONIC_FILE)


def synthetic_catalog(n=600, days=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, days * DAY, n)) + 1_700_000_000_000
    return pd.DataFrame({
        "id": [f"e{i}" for i in range(n)], "time": t,
        "latitude": rng.uniform(-10, 10, n), "longitude": rng.uniform(150, 170, n),
        "depth_km": rng.uniform(5, 300, n), "mag": np.round(rng.uniform(4.5, 6.5, n), 1),
    })


def test_build_dataset_labels_warmup_and_columns(segments):
    cat = synthetic_catalog()
    df = T.build_dataset(cat, segments)
    assert set(F.FEATURE_NAMES) <= set(df.columns)
    assert df["time"].min() >= cat["time"].min() + F.DENSITY_WINDOW_DAYS * DAY   # warm-up excluded
    assert ((df["mag"] >= F.SIGNIFICANT_MAG) == (df["label"] == 1)).all()
    assert df["time"].is_monotonic_increasing


def test_density_feature_does_not_see_the_future(segments):
    """Appending later events must not change any earlier sample's features."""
    cat = synthetic_catalog()
    base = T.build_dataset(cat, segments)
    later = synthetic_catalog(seed=1)
    later["time"] = later["time"] + 400 * DAY
    later["id"] = "late" + later["id"]
    extended = T.build_dataset(pd.concat([cat, later]), segments)
    merged = base.merge(extended, on="id", suffixes=("", "_ext"))
    assert len(merged) == len(base)
    assert (merged["local_density_30d"] == merged["local_density_30d_ext"]).all()


def test_rows_missing_magnitude_or_depth_are_not_samples(segments):
    cat = synthetic_catalog()
    cat.loc[100:109, "mag"] = np.nan
    cat.loc[200:204, "depth_km"] = np.nan
    df = T.build_dataset(cat, segments)
    assert not df["mag"].isna().any() and not df["depth_km"].isna().any()


def test_chronological_split_has_no_overlap():
    df = pd.DataFrame({"time": np.random.default_rng(3).permutation(1000), "label": 0})
    tr, va, te = T.chronological_split(df)
    assert (len(tr), len(va), len(te)) == (700, 150, 150)
    assert tr["time"].max() < va["time"].min() and va["time"].max() < te["time"].min()


def test_grid_baseline_uses_train_history_only():
    train = pd.DataFrame({"latitude": [1.0] * 10, "longitude": [1.0] * 10, "label": [1] * 10})
    rate = T.grid_rate_baseline(train, base_rate=0.2)
    test = pd.DataFrame({"latitude": [1.0, 40.0], "longitude": [1.0, 40.0]})
    p = rate(test)
    assert p[0] == pytest.approx((10 + T.GRID_PRIOR * 0.2) / (10 + T.GRID_PRIOR))
    assert p[1] == pytest.approx(0.2)   # unseen cell falls back to the base rate


def test_metrics_against_known_values():
    y = np.array([0, 0, 1, 1])
    perfect = T.prob_metrics(y, np.array([0.0, 0.0, 1.0, 1.0]))
    assert perfect["roc_auc"] == 1.0 and perfect["brier"] < 1e-6
    const = T.prob_metrics(y, np.full(4, 0.5))
    assert const["roc_auc"] == 0.5 and const["brier"] == 0.25
    tm = T.threshold_metrics(y, np.array([0.1, 0.6, 0.7, 0.2]), 0.5)
    assert tm["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert tm["precision"] == 0.5 and tm["recall"] == 0.5


def test_training_is_reproducible_with_a_seed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=300) > 0).astype(int)
    m1, info1 = T.train_nn(X[:200], y[:200], X[200:], y[200:], seed=7, epochs=5, patience=5)
    m2, info2 = T.train_nn(X[:200], y[:200], X[200:], y[200:], seed=7, epochs=5, patience=5)
    assert info1["best_val_log_loss"] == info2["best_val_log_loss"]
    assert np.allclose(T.nn_proba(m1, X[200:]), T.nn_proba(m2, X[200:]))
    assert not m1.training
    assert isinstance(m1.net[0], torch.nn.Linear) and m1.net[0].in_features == len(F.FEATURE_NAMES)
