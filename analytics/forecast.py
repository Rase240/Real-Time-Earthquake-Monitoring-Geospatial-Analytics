"""
Seven-day regional count forecasting: candidate models, predictive distributions,
metrics and the served artifact.

Predictive distribution
-----------------------
Each model predicts a mean mu for the next-week count. Uncertainty is expressed
with a negative-binomial (NB2) predictive distribution

    Var(y) = mu + alpha * mu^2,

whose single dispersion parameter alpha is estimated by maximum likelihood on a
held-out period (never on the period being evaluated). alpha = 0 reduces to a
Poisson distribution. Central 80% and 95% intervals are NB quantiles; their
empirical coverage on the test period is reported, so the intervals are a
checked claim rather than an assumption.
"""
import json
import logging
import os

import joblib
import numpy as np
import sklearn
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from analytics import regions
from analytics.regional_features import FEATURE_NAMES

log = logging.getLogger("terrawatcher.forecast")

ARTIFACT_VERSION = 1
MIN_MU = 1e-6
INTERVALS = (0.8, 0.95)
# Feature column index used by the rolling-mean baselines.
_COL = {n: i for i, n in enumerate(FEATURE_NAMES)}


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def make_glm(l2):
    """Poisson GLM on log1p-transformed, standardised features (log link)."""
    return make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler(),
                         PoissonRegressor(alpha=l2, max_iter=1000))


def make_gbm(max_leaf_nodes, learning_rate, max_iter, seed):
    """Gradient-boosted trees with a Poisson loss. Early stopping is off so fits are deterministic."""
    return HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=max_leaf_nodes, learning_rate=learning_rate,
                                         max_iter=max_iter, min_samples_leaf=100, l2_regularization=1.0,
                                         early_stopping=False, random_state=seed)


class HistoricalMean:
    """Baseline 1: each region's mean weekly count over the model's training period."""

    def fit(self, X, y, region):
        self.global_mean = float(np.mean(y)) if len(y) else 0.0
        sums = np.bincount(region, weights=y, minlength=regions.N_REGIONS)
        counts = np.bincount(region, minlength=regions.N_REGIONS)
        self.means = np.where(counts > 0, sums / np.maximum(counts, 1), self.global_mean)
        return self

    def predict(self, X, region):
        return self.means[region]


class RollingMean:
    """Baselines 2/3: the trailing mean weekly count (4 or 52 weeks), read from the feature row."""

    def __init__(self, feature):
        self.feature = feature

    def fit(self, X, y, region):
        return self

    def predict(self, X, region):
        return X[:, _COL[self.feature]]


# -----------------------------------------------------------------------------
# Negative-binomial predictive distribution
# -----------------------------------------------------------------------------
def _nb_params(mu, alpha):
    mu = np.maximum(np.asarray(mu, float), MIN_MU)
    r = 1.0 / alpha
    return r, r / (r + mu)


def nb_logpmf(y, mu, alpha):
    if alpha < 1e-8:
        return stats.poisson.logpmf(y, np.maximum(mu, MIN_MU))
    r, p = _nb_params(mu, alpha)
    return stats.nbinom.logpmf(y, r, p)


def fit_dispersion(y, mu):
    """MLE of the NB2 dispersion alpha given observed counts and model means (alpha in [1e-8, 50])."""
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), MIN_MU)

    def nll(log_a):
        return -nb_logpmf(y, mu, float(np.exp(log_a))).sum()
    res = optimize.minimize_scalar(nll, bounds=(np.log(1e-8), np.log(50.0)), method="bounded",
                                   options={"xatol": 1e-4})
    alpha = float(np.exp(res.x))
    return 0.0 if alpha < 1e-6 else alpha


def interval(mu, alpha, level):
    """Central NB (or Poisson) prediction interval [lo, hi] of integer counts."""
    q_lo, q_hi = (1 - level) / 2, 1 - (1 - level) / 2
    mu = np.maximum(np.asarray(mu, float), MIN_MU)
    if alpha < 1e-8:
        return stats.poisson.ppf(q_lo, mu), stats.poisson.ppf(q_hi, mu)
    r, p = _nb_params(mu, alpha)
    return stats.nbinom.ppf(q_lo, r, p), stats.nbinom.ppf(q_hi, r, p)


def prob_at_least(k, mu, alpha):
    """P(Y >= k) under the predictive distribution."""
    mu = np.maximum(np.asarray(mu, float), MIN_MU)
    if alpha < 1e-8:
        return stats.poisson.sf(k - 1, mu)
    r, p = _nb_params(mu, alpha)
    return stats.nbinom.sf(k - 1, r, p)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def poisson_deviance(y, mu):
    """Mean Poisson deviance 2 * [y log(y/mu) - (y - mu)], with 0 log 0 = 0. Defined for y = 0."""
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), MIN_MU)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


def count_metrics(y, mu, alpha=None):
    """Point and distributional metrics. No percentage errors: they are undefined when y = 0."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    out = {"n": int(len(y)), "mean_observed": float(np.mean(y)), "mean_predicted": float(np.mean(mu)),
           "mae": float(np.mean(np.abs(y - mu))), "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
           "poisson_deviance": poisson_deviance(y, mu)}
    zero = y == 0
    if zero.any():
        out["mae_zero_weeks"] = float(np.mean(np.abs(y[zero] - mu[zero])))
    if (~zero).any():
        out["mae_nonzero_weeks"] = float(np.mean(np.abs(y[~zero] - mu[~zero])))
    if alpha is not None:
        out["alpha"] = float(alpha)
        out["nb_log_score"] = float(-np.mean(nb_logpmf(y, mu, alpha)))
        for level in INTERVALS:
            lo, hi = interval(mu, alpha, level)
            key = f"{int(level * 100)}"
            out[f"coverage_{key}"] = float(np.mean((y >= lo) & (y <= hi)))
            out[f"mean_width_{key}"] = float(np.mean(hi - lo))
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


# -----------------------------------------------------------------------------
# Served artifact
# -----------------------------------------------------------------------------
def artifact_paths(model_dir, threshold):
    tag = f"m{threshold:.1f}"
    return os.path.join(model_dir, f"forecast_{tag}.joblib"), os.path.join(model_dir, f"forecast_{tag}.json")


def save_artifact(model_dir, threshold, estimator, meta):
    est_path, meta_path = artifact_paths(model_dir, threshold)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(estimator, est_path, compress=3)
    meta = {**meta, "artifact_version": ARTIFACT_VERSION, "sklearn_version": sklearn.__version__,
            "feature_names": FEATURE_NAMES, "threshold": threshold}
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return est_path, meta_path


class ForecastModel:
    """
    One trained threshold. Loading is all-or-nothing: if the estimator or metadata is
    missing, unreadable or built for another feature schema / sklearn version, the
    model reports `ready = False` with the reason and refuses to predict.
    """

    def __init__(self, model_dir, threshold):
        self.threshold = threshold
        self.estimator, self.meta, self.load_error = None, {}, None
        est_path, meta_path = artifact_paths(model_dir, threshold)
        try:
            if not (os.path.exists(est_path) and os.path.exists(meta_path)):
                raise FileNotFoundError(f"Missing forecast artifact for M{threshold:.1f}")
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("artifact_version") != ARTIFACT_VERSION:
                raise ValueError("artifact version mismatch")
            if meta.get("feature_names") != FEATURE_NAMES:
                raise ValueError("feature schema mismatch")
            if meta.get("sklearn_version") != sklearn.__version__:
                raise ValueError(f"trained with scikit-learn {meta.get('sklearn_version')}, "
                                 f"running {sklearn.__version__}")
            if not meta.get("regions") or not np.isfinite(meta.get("alpha", np.nan)):
                raise ValueError("metadata lacks regions or dispersion")
            est = joblib.load(est_path)
            if not hasattr(est, "predict"):
                raise TypeError("artifact is not an estimator")
            self.estimator, self.meta = est, meta
        except Exception as exc:  # noqa: BLE001 - any failure disables the model, with the reason
            self.load_error = f"Forecast model M{threshold:.1f} unavailable: {exc}"
            log.warning(self.load_error)

    @property
    def ready(self):
        return self.estimator is not None

    @property
    def alpha(self):
        return float(self.meta["alpha"])

    @property
    def region_index(self):
        return [r["index"] for r in self.meta["regions"]]

    def predict(self, X):
        if not self.ready:
            raise RuntimeError(self.load_error or "forecast model not loaded")
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected (n, {len(FEATURE_NAMES)}) features, got {X.shape}")
        return np.maximum(self.estimator.predict(X), MIN_MU)
