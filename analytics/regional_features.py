"""
Forecast-origin features for regional weekly counts (shared by training and serving).

Contract
--------
A forecast is issued at an origin time T for one region r and magnitude
threshold m. Its target is

    y(r, T) = number of catalogue events with magnitude >= m, epicentre in r,
              and time in [T, T + 7 days).

Every feature below is computed from the weekly count matrix columns that end
at or before T (week bins [T - 7k days, T - 7(k-1) days), k = 1..52), so a
feature can never contain an event at or after T. The training pipeline calls
`features_at` with a long matrix and many origin columns; the API calls it with
a 52-week matrix ending at the live origin. Both run this same function, so
there is a single definition of each feature (no train/serve skew).

Missing values: a week with no qualifying event is a genuine observed zero in
the catalogue (the API only builds a matrix after a complete, non-truncated
download; failures raise instead). There are no other missing values, so no
imputation is performed. `weeks_since_last` is capped at 52 when the region had
no qualifying event in the full 52-week window ("52 or more").
"""
import numpy as np

from analytics import regions

LOOKBACK_WEEKS = 52
LARGE_MAG = 6.0

FEATURES = [
    {"name": "lag1", "unit": "events", "definition": "Count in the week [T-7d, T)."},
    {"name": "lag2", "unit": "events", "definition": "Count in the week [T-14d, T-7d)."},
    {"name": "mean_4w", "unit": "events/week", "definition": "Mean weekly count over [T-28d, T)."},
    {"name": "mean_13w", "unit": "events/week", "definition": "Mean weekly count over [T-91d, T)."},
    {"name": "mean_52w", "unit": "events/week", "definition": "Mean weekly count over [T-364d, T)."},
    {"name": "std_13w", "unit": "events/week", "definition": "Population standard deviation of the 13 weekly counts before T."},
    {"name": "neighbor_mean_4w", "unit": "events/week",
     "definition": "Mean weekly count over [T-28d, T) summed over the up-to-8 adjacent cells (longitude wraps)."},
    {"name": "large_4w", "unit": "events",
     "definition": f"Number of M>={LARGE_MAG} events in the region in [T-28d, T) (independent of the threshold)."},
    {"name": "weeks_since_last", "unit": "weeks",
     "definition": "Whole weeks between T and the latest week before T with a qualifying event (0 = last week); "
                   "52 means none in the previous 52 weeks."},
]
FEATURE_NAMES = [f["name"] for f in FEATURES]
_ADJ = None


def _adjacency():
    global _ADJ
    if _ADJ is None:
        _ADJ = regions.adjacency()
    return _ADJ


def features_at(C, L, origin_cols, region_idx):
    """
    Feature matrix for every (origin column, region) pair.

    C            (N_REGIONS, W) weekly counts at the forecast threshold.
    L            (N_REGIONS, W) weekly counts of M >= LARGE_MAG.
    origin_cols  column indices k; the origin T is the START of column k, so only
                 columns < k are read. Requires k >= LOOKBACK_WEEKS.
    region_idx   regions to return.

    Returns X with shape (len(origin_cols) * len(region_idx), len(FEATURE_NAMES)),
    ordered origin-major (all regions for the first origin, then the next, ...).
    """
    C = np.asarray(C, dtype=np.int64)
    L = np.asarray(L, dtype=np.int64)
    ks = np.asarray(origin_cols, dtype=int)
    reg = np.asarray(region_idx, dtype=int)
    if ks.size and (ks.min() < LOOKBACK_WEEKS or ks.max() > C.shape[1]):
        raise ValueError(f"origin columns must lie in [{LOOKBACK_WEEKS}, {C.shape[1]}]")

    def csum(M):
        return np.concatenate([np.zeros((M.shape[0], 1), dtype=float), np.cumsum(M, axis=1, dtype=float)], axis=1)

    Cr = C[reg]
    S, S2 = csum(Cr), csum(Cr.astype(float) ** 2)
    N = csum((_adjacency()[reg] @ C).astype(float))
    Ls = csum(L[reg])

    def window_sum(cs, w):
        return cs[:, ks] - cs[:, ks - w]                  # (regions, origins)

    mean13 = window_sum(S, 13) / 13
    var13 = np.maximum(window_sum(S2, 13) / 13 - mean13 ** 2, 0.0)

    # Index of the latest non-empty column strictly before each origin.
    cols = np.arange(Cr.shape[1])
    last = np.where(Cr > 0, cols[None, :], -1)
    last = np.maximum.accumulate(last, axis=1)
    last_before = last[:, ks - 1]
    since = np.where(last_before < 0, LOOKBACK_WEEKS, (ks - 1)[None, :] - last_before)
    since = np.minimum(since, LOOKBACK_WEEKS)

    feats = [
        Cr[:, ks - 1], Cr[:, ks - 2],
        window_sum(S, 4) / 4, mean13, window_sum(S, 52) / 52, np.sqrt(var13),
        window_sum(N, 4) / 4, window_sum(Ls, 4), since,
    ]
    # (regions, origins) -> origin-major rows
    X = np.stack([np.asarray(f, float).T.reshape(-1) for f in feats], axis=1)
    if not np.all(np.isfinite(X)):
        raise ValueError("feature matrix contains non-finite values")
    return X


def targets_at(C, origin_cols, region_idx):
    """y(r, T): counts in column k (the week starting at the origin), origin-major like features_at."""
    ks = np.asarray(origin_cols, dtype=int)
    if ks.size and ks.max() >= C.shape[1]:
        raise ValueError("target week lies outside the count matrix")
    return np.asarray(C)[np.asarray(region_idx, dtype=int)][:, ks].T.reshape(-1).astype(float)


def live_matrices(times, mags, lats, lons, threshold, origin_ms):
    """52-week matrices ending exactly at origin_ms, for serving. Column 52 does not exist (no target)."""
    epoch = regions.week_epoch(origin_ms, LOOKBACK_WEEKS)
    C = regions.count_matrix(times, mags, lats, lons, threshold, epoch, LOOKBACK_WEEKS)
    L = regions.count_matrix(times, mags, lats, lons, LARGE_MAG, epoch, LOOKBACK_WEEKS)
    return C, L, epoch
