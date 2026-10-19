"""
Module C: elevated observed activity relative to a region's own recent history.

Operational definition
----------------------
For region r, magnitude threshold m, an observation window of k weeks ending at
time T, and a reference window of n weeks immediately before it:

  1. x_1..x_n  = weekly counts in the reference window, mean mu, variance s^2;
     y = the k-week total in the observation window.
  2. Weekly overdispersion alpha = max(alpha_recent, alpha_long), where
     alpha_recent = max(0, (s^2 - mu) / mu^2) (method of moments, NB2) and
     alpha_long is the maximum-likelihood NB2 dispersion of the region's
     2001-2015 weekly counts (models/anomaly_reference.json). One quiet reference
     year therefore cannot make the test overconfident: heavy aftershock tails are
     rare within 52 weeks. The MLE is used rather than the 2001-2015 moment
     estimate because one megathrust sequence dominates a variance (alpha up to
     ~400) but not a likelihood, so the moment floor left most active cells
     without power.
  3. Model: weekly counts are independent NB2(mu, alpha). The reference total
     X = x_1 + ... + x_n is then NB with shape n/alpha and the window total Y is
     NB with shape k/alpha, with a common unknown scale.
  4. One-sided p-value p = P(Y >= y | X + Y = N), N the observed total of both
     windows. Conditioning on N removes the unknown mean exactly: Y given N is
     BetaBinomial(N, k/alpha, n/alpha), or Binomial(N, k / (n + k)) when
     alpha = 0 (the classical conditional test for two Poisson counts). Plugging
     in the estimated mean instead is anti-conservative, because a quiet
     reference window underestimates mu; in the null simulation that alone
     produced most of the false alerts.
  5. Benjamini-Hochberg across every region tested in the same run gives q-values;
     a region is "elevated" when q <= fdr and observed >= MIN_OBSERVED.

`expected` = k * mu and `reference_sd` = sqrt(k * w + k^2 * w / n), with
w = mu + alpha * mu^2, describe the reference for display (the last term carries
the uncertainty of mu estimated from n weeks); the p-value comes from step 4.

Regions whose reference window holds fewer than `min_reference_events`
qualifying events are reported as "insufficient_reference", not as normal.

Only excess activity is tested. With weekly counts this small, a deficit is
rarely distinguishable from chance and is not reported.

Data gaps vs. zeros: a day with no M4.5+ event anywhere on Earth is essentially
impossible (the global rate is ~20 per day), so such days are treated as
possible feed gaps. When any are present, regions that are not flagged get the
status "data_incomplete" instead of "not_elevated".

An elevated region means the observed count is unusually high for that region's
recent history, often because of one large earthquake and its aftershocks. It is
not a prediction of a future large earthquake.
"""
import json

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from analytics import regions

DAY_MS = 86_400_000
MIN_OBSERVED = 3
DEFAULTS = {"window_weeks": 4, "reference_weeks": 52, "fdr": 0.05, "min_reference_events": 5}
WINDOW_WEEKS = (1, 2, 4)
REFERENCE_WEEKS = (26, 52)
FDR_LEVELS = (0.01, 0.05, 0.1)
LOG_ALPHA_BOUNDS = (-9.0, 8.0)   # MLE search range for log(alpha): 1e-4 .. ~3000


def moment_dispersion(counts):
    """NB2 method-of-moments dispersion of weekly counts along the last axis (0 when not overdispersed)."""
    x = np.asarray(counts, float)
    mu = x.mean(axis=-1)
    s2 = x.var(axis=-1, ddof=1) if x.shape[-1] > 1 else mu
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(mu > 0, (s2 - mu) / np.where(mu > 0, mu, 1.0) ** 2, 0.0)
    return np.maximum(a, 0.0)


def mle_dispersion(counts):
    """
    Maximum-likelihood NB2 dispersion of one series of weekly counts (the MLE of
    the mean is the sample mean). 0 when the sample is not overdispersed or a
    Poisson fits at least as well.
    """
    x = np.asarray(counts, float)
    mu = x.mean() if len(x) else 0.0
    if len(x) < 2 or mu <= 0 or x.var(ddof=1) <= mu:
        return 0.0

    def nll(log_a):
        r = np.exp(-log_a)
        return -np.sum(gammaln(x + r) - gammaln(r) + r * np.log(r / (r + mu)) + x * np.log(mu / (r + mu)))

    res = minimize_scalar(nll, bounds=LOG_ALPHA_BOUNDS, method="bounded")
    poisson_nll = -np.sum(x * np.log(mu) - mu - gammaln(x + 1))
    return float(np.exp(res.x)) if res.fun < poisson_nll else 0.0


def reference_moments(ref_counts, k, long_alpha=0.0):
    """(mean, variance, alpha) of the k-week reference total given reference counts (rows = regions)."""
    x = np.asarray(ref_counts, float)
    n = x.shape[-1]
    mu = x.mean(axis=-1)
    alpha = np.maximum(moment_dispersion(x), np.asarray(long_alpha, float))
    w = mu + alpha * mu ** 2
    return k * mu, k * w + k ** 2 * w / n, alpha


def conditional_p_value(observed, reference_total, k, n, alpha):
    """
    P(Y >= observed | X + Y = reference_total + observed) for a k-week window total Y
    and an n-week reference total X of independent NB2 weekly counts with dispersion alpha
    (vectorised). Beta-binomial, or binomial when alpha == 0; 1 when nothing was observed.
    """
    y, x, a = np.broadcast_arrays(np.asarray(observed, float), np.asarray(reference_total, float),
                                  np.asarray(alpha, float))
    total = x + y
    out = np.ones(y.shape)
    pois, nb = (y > 0) & (a <= 0), (y > 0) & (a > 0)
    out[pois] = stats.binom.sf(y[pois] - 1, total[pois], k / (n + k))
    r = 1.0 / a[nb]
    out[nb] = stats.betabinom.sf(y[nb] - 1, total[nb], k * r, n * r)
    return np.clip(out, 0.0, 1.0)


def load_reference(path):
    """
    Long-term dispersion per region from models/anomaly_reference.json, as
    {threshold: array(N_REGIONS)}. Returns ({}, reason) if missing or malformed;
    detection then falls back to the recent-window dispersion only.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        out = {}
        for thr, values in data["thresholds"].items():
            arr = np.asarray(values, float)
            if arr.shape != (regions.N_REGIONS,) or not np.all(np.isfinite(arr)) or (arr < 0).any():
                raise ValueError(f"bad dispersion table for M{thr}")
            out[float(thr)] = arr
        return out, data.get("period"), None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {}, None, f"Long-term dispersion unavailable ({type(exc).__name__}); using the reference window only."


def benjamini_hochberg(p):
    """BH-adjusted q-values (monotone, capped at 1)."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def gap_days(times_ms, start_ms, end_ms):
    """UTC days fully inside [start, end) with no event at all in the supplied (global) catalogue."""
    first = -(-int(start_ms) // DAY_MS)            # first whole day starting at or after start
    last = int(end_ms) // DAY_MS                    # days must end at or before end
    if last <= first:
        return []
    days = np.asarray(times_ms, np.int64) // DAY_MS
    present = set(np.unique(days[(days >= first) & (days < last)]).tolist())
    return [d * DAY_MS for d in range(first, last) if d not in present]


def detect(C, window_weeks=4, reference_weeks=52, fdr=0.05, min_reference_events=5, region_idx=None,
           gaps_in_window=0, gaps_in_reference=0, long_alpha=None):
    """
    Test every region. C has reference_weeks + window_weeks columns ending at the
    observation end (oldest first). Returns a list of per-region results sorted by q-value.
    """
    C = np.asarray(C)
    if C.shape[1] != reference_weeks + window_weeks:
        raise ValueError("count matrix width must equal reference_weeks + window_weeks")
    region_idx = np.arange(C.shape[0]) if region_idx is None else np.asarray(region_idx)
    ref_all, obs_all = C[region_idx, :reference_weeks], C[region_idx, reference_weeks:].sum(axis=1)
    keep = ref_all.sum(axis=1) >= min_reference_events
    la_all = np.zeros(len(region_idx)) if long_alpha is None else np.asarray(long_alpha, float)[region_idx]
    mean, var, alpha = reference_moments(ref_all[keep], window_weeks, la_all[keep])
    p = conditional_p_value(obs_all[keep], ref_all[keep].sum(axis=1), window_weeks, reference_weeks, alpha)
    results, tested = [], []
    j = 0
    for i, r in enumerate(region_idx):
        ref, obs = ref_all[i], int(obs_all[i])
        row = {"region": regions.region_id(r), "index": int(r), "observed": obs,
               "reference_total": int(ref.sum()), "reference_weekly_mean": round(float(ref.mean()), 4)}
        if not keep[i]:
            if obs == 0:
                continue                     # nothing observed and nothing to compare with: omit
            row.update(status="insufficient_reference", expected=None, p_value=None, q_value=None)
            results.append(row)
            continue
        m, v, a = float(mean[j]), float(var[j]), float(alpha[j])
        row.update(expected=round(m, 3), reference_sd=round(float(np.sqrt(v)), 3),
                   weekly_dispersion=round(a, 4), long_term_dispersion=round(float(la_all[i]), 4),
                   ratio=round(obs / m, 3) if m > 0 else None,
                   distribution="poisson" if a <= 0 else "negative_binomial",
                   p_value=float(p[j]))
        j += 1
        results.append(row)
        tested.append(row)
    q = benjamini_hochberg([t["p_value"] for t in tested])
    incomplete = (gaps_in_window + gaps_in_reference) > 0
    for row, qv in zip(tested, q, strict=True):
        row["q_value"] = float(qv)
        elevated = qv <= fdr and row["observed"] >= MIN_OBSERVED
        row["status"] = "elevated" if elevated else ("data_incomplete" if incomplete else "not_elevated")
        row["p_value"] = float(f"{row['p_value']:.4g}")
        row["q_value"] = float(f"{qv:.4g}")
    results.sort(key=lambda x: (x.get("q_value") is None, x.get("q_value") or 0, -x["observed"]))
    return results, len(tested)


def rolling_rule(C, window_weeks, reference_weeks, factor=2.0):
    """Baseline for evaluation: flag when observed >= factor x reference mean x k and >= MIN_OBSERVED."""
    ref = C[:, :reference_weeks].mean(axis=1) * window_weeks
    obs = C[:, reference_weeks:].sum(axis=1)
    return (obs >= np.maximum(factor * ref, MIN_OBSERVED)) & (C[:, :reference_weeks].sum(axis=1) > 0)
