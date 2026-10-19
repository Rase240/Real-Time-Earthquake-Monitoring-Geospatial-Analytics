"""
Seven-day regional count forecasting: candidate models, predictive distributions,
metrics and the served artifact.

Predictive distribution
-----------------------
Each model predicts a mean mu for the next-week count. Uncertainty is expressed
with a negative-binomial (NB2) predictive distribution

    Var(y) = mu + alpha(mu) * mu^2,    alpha(mu) = exp(a + b * log(mu)),

whose dispersion curve (a, b) is estimated by maximum likelihood on the
validation period (never on the period being evaluated). b = 0 is the single
constant alpha used before; on validation data low-mean cells are relatively
more overdispersed (b < 0), and the curve improved the out-of-fold NB log score
at both thresholds. alpha = 0 reduces to a Poisson distribution.

Intervals. Counts are integers and most predictive distributions put more than
10% of their mass on 0, so the equal-tailed interval [q(0.10), q(0.90)] keeps
far more than 80% of the mass (often 93-97%). The served "80%" and "95%"
intervals are therefore the SHORTEST integer intervals whose predictive
probability is at least the level; for discrete counts no interval has exactly
the nominal mass, and the model-implied mass of each interval is reported with
its empirical test coverage. A calibrated model has empirical coverage close to
the model-implied coverage, not to the nominal level.
"""
import json
import logging
import math
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

ARTIFACT_VERSION = 2            # 2: mean-dependent dispersion curve + shortest intervals
MIN_MU = 1e-6
MIN_ALPHA = 1e-8
LOG_ALPHA_RANGE = (math.log(MIN_ALPHA), math.log(50.0))
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
def alpha_of(mu, dispersion):
    """
    NB2 dispersion for each mean. `dispersion` is a constant alpha (float) or a curve
    {"a": a, "b": b} meaning alpha(mu) = exp(a + b log mu), clipped to [1e-8, 50].
    """
    mu = np.maximum(np.asarray(mu, float), MIN_MU)
    if isinstance(dispersion, dict):
        return np.exp(np.clip(dispersion["a"] + dispersion["b"] * np.log(mu), *LOG_ALPHA_RANGE))
    return np.full(mu.shape, float(dispersion))


def _dist(mu, dispersion):
    """Frozen predictive distribution (NB2, or Poisson where alpha < 1e-8)."""
    mu = np.maximum(np.asarray(mu, float), MIN_MU)
    a = alpha_of(mu, dispersion)
    if np.all(a < MIN_ALPHA):
        return stats.poisson(mu)
    r = 1.0 / np.maximum(a, MIN_ALPHA)
    return stats.nbinom(r, r / (r + mu))


def nb_logpmf(y, mu, dispersion):
    return _dist(mu, dispersion).logpmf(np.asarray(y, float))


def fit_dispersion(y, mu):
    """MLE of a constant NB2 dispersion alpha given observed counts and model means (alpha in [1e-8, 50])."""
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), MIN_MU)

    def nll(log_a):
        return -nb_logpmf(y, mu, float(np.exp(log_a))).sum()
    res = optimize.minimize_scalar(nll, bounds=LOG_ALPHA_RANGE, method="bounded", options={"xatol": 1e-4})
    alpha = float(np.exp(res.x))
    return 0.0 if alpha < 1e-6 else alpha


def fit_dispersion_curve(y, mu):
    """
    MLE of the dispersion curve alpha(mu) = exp(a + b log mu) given observed counts and model means.
    Starts from the constant-alpha MLE (b = 0), so the fit can only match or improve its likelihood.
    """
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), MIN_MU)
    a0 = math.log(max(fit_dispersion(y, mu), 1e-3))

    def nll(th):
        return -nb_logpmf(y, mu, {"a": th[0], "b": th[1]}).sum()
    res = optimize.minimize(nll, [a0, 0.0], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
    th = res.x if res.fun <= nll([a0, 0.0]) else np.array([a0, 0.0])
    return {"a": round(float(th[0]), 6), "b": round(float(th[1]), 6)}


def central_interval(mu, dispersion, level):
    """Equal-tailed interval [q((1-level)/2), q((1+level)/2)]; kept for comparison (it over-covers discrete counts)."""
    d = _dist(mu, dispersion)
    return d.ppf((1 - level) / 2), d.ppf(1 - (1 - level) / 2)


def interval(mu, dispersion, level):
    """
    Shortest integer interval [lo, hi] whose predictive probability is >= level (ties: larger probability).
    Every lower bound from 0 to the (1 - level) quantile is tried with its smallest valid upper bound.
    """
    mu = np.maximum(np.asarray(mu, float), MIN_MU)
    d = _dist(mu, dispersion)
    best_lo, best_hi = np.zeros(mu.shape), d.ppf(level)
    best_mass = d.cdf(best_hi)
    l_max = d.ppf(1 - level)
    for lo in range(1, int(np.max(l_max, initial=0)) + 1):
        below = d.cdf(lo - 1)
        hi = d.ppf(np.minimum(below + level, 1.0))
        hi = np.where(d.cdf(hi) - below < level - 1e-12, hi + 1, hi)        # guard against rounding in ppf
        mass = d.cdf(hi) - below
        width, best_width = hi - lo, best_hi - best_lo
        better = (lo <= l_max) & ((width < best_width) | ((width == best_width) & (mass > best_mass)))
        best_lo, best_hi = np.where(better, lo, best_lo), np.where(better, hi, best_hi)
        best_mass = np.where(better, mass, best_mass)
    return best_lo, best_hi


def interval_mass(lo, hi, mu, dispersion):
    """Predictive probability P(lo <= Y <= hi) (the model-implied coverage of an interval)."""
    d = _dist(mu, dispersion)
    lo = np.asarray(lo, float)
    return d.cdf(hi) - np.where(lo > 0, d.cdf(lo - 1), 0.0)


def prob_at_least(k, mu, dispersion):
    """P(Y >= k) under the predictive distribution."""
    return _dist(mu, dispersion).sf(k - 1)


def randomised_pit(y, mu, dispersion, seed=0):
    """Randomised probability integral transform for counts; uniform on [0, 1] when the forecast is calibrated."""
    d = _dist(mu, dispersion)
    y = np.asarray(y, float)
    lo = np.where(y > 0, d.cdf(y - 1), 0.0)
    return lo + np.random.default_rng(seed).uniform(size=len(y)) * (d.cdf(y) - lo)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def poisson_unit_deviance(y, mu):
    """Per-observation Poisson deviance 2 * [y log(y/mu) - (y - mu)], with 0 log 0 = 0. Defined for y = 0."""
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), MIN_MU)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (term - (y - mu))


def poisson_deviance(y, mu):
    """Mean Poisson deviance."""
    return float(np.mean(poisson_unit_deviance(y, mu)))


def count_metrics(y, mu, alpha=None, central=False):
    """
    Point and distributional metrics. No percentage errors: they are undefined when y = 0.
    `alpha` is a constant NB2 dispersion or a dispersion curve {"a", "b"} (see alpha_of).
    With a dispersion: NB log score (lower is better), and for each interval level the empirical coverage,
    the mean model-implied coverage (what the predictive distribution claims) and the mean width.
    `central=True` adds the same for the equal-tailed interval, for comparison.
    """
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    out = {"n": int(len(y)), "mean_observed": float(np.mean(y)), "mean_predicted": float(np.mean(mu)),
           "mae": float(np.mean(np.abs(y - mu))), "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
           "poisson_deviance": poisson_deviance(y, mu)}
    dispersion = alpha
    zero = y == 0
    if zero.any():
        out["mae_zero_weeks"] = float(np.mean(np.abs(y[zero] - mu[zero])))
    if (~zero).any():
        out["mae_nonzero_weeks"] = float(np.mean(np.abs(y[~zero] - mu[~zero])))
    if dispersion is not None:
        if isinstance(dispersion, dict):
            out["dispersion_curve"] = dispersion
        else:
            out["alpha"] = float(dispersion)
        out["nb_log_score"] = float(-np.mean(nb_logpmf(y, mu, dispersion)))
        kinds = [("", interval)] + ([("central_", central_interval)] if central else [])
        for prefix, fn in kinds:
            for level in INTERVALS:
                lo, hi = fn(mu, dispersion, level)
                key = f"{int(level * 100)}"
                out[f"{prefix}coverage_{key}"] = float(np.mean((y >= lo) & (y <= hi)))
                out[f"{prefix}model_implied_coverage_{key}"] = float(np.mean(interval_mass(lo, hi, mu, dispersion)))
                out[f"{prefix}mean_width_{key}"] = float(np.mean(hi - lo))
        if central:
            pit = randomised_pit(y, mu, dispersion)
            out["pit_deciles"] = np.round(np.histogram(pit, bins=np.linspace(0, 1, 11))[0] / len(y), 4).tolist()
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
            curve = meta.get("dispersion_curve")
            if (not meta.get("regions") or not isinstance(curve, dict)
                    or not all(np.isfinite(curve.get(k, np.nan)) for k in ("a", "b"))):
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
        """Constant-dispersion MLE on validation (reported for reference; intervals use `dispersion`)."""
        return float(self.meta["alpha"])

    @property
    def dispersion(self):
        """Dispersion curve {"a", "b"} used for every predictive distribution and interval."""
        return self.meta["dispersion_curve"]

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
