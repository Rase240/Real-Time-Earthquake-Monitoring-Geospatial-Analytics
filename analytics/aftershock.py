"""
Module D: event-centred sequence analysis and modified Omori-Utsu fitting.

Event selection (deterministic)
-------------------------------
Given a reference event (time t0, epicentre P) and parameters:
  * an event is "post" when t0 < t <= min(t0 + post_days, data end), "pre" when
    t0 - pre_days <= t < t0;
  * its great-circle epicentral distance to P is <= radius_km;
  * its magnitude is >= min_mag (events without a magnitude are excluded);
  * optional depth limits apply to events with a reported depth; events
    without depth are excluded only when a depth limit is set;
  * the reference event itself (by id) is always excluded.
Nearby post-event earthquakes are not assumed to be caused by the reference
event; they are simply the events satisfying these criteria.

Decay model
-----------
    lambda(t) = mu + K / (t + c)^p        (events per day, t in days after t0)

mu is the background rate, fixed to the pre-event rate in the same circle and
magnitude range (0 when no pre-event window is requested). K, c and p are
estimated by maximum likelihood for an inhomogeneous Poisson process (Ogata
1983) on [t_start, t_end], where t_start excludes the first hours after the
reference event, when catalogues are known to miss many events.

The fit is only attempted with at least MIN_FIT_EVENTS events. It is reported
as reliable only if the optimiser converged, p and c are not at their search
bounds, the Hessian is positive definite (standard errors exist), the
transformed-time Kolmogorov-Smirnov test does not reject the model (p >= 0.05)
and the model beats a constant-rate Poisson model by AIC (difference >= 2).
Otherwise the observed data are shown with the reasons why no fit is shown.
The fitted curve is evaluated only inside the fitted interval: it is not
extrapolated beyond the observations.
"""
import math

import numpy as np
from scipy import optimize, stats

from analytics import geo
from analytics.catalog import events_to_frame

DAY_MS = 86_400_000
MIN_FIT_EVENTS = 20
P_BOUNDS = (0.2, 3.0)
C_BOUNDS = (1e-4, 5.0)          # days
DEFAULTS = {"radius_km": None, "pre_days": 30, "post_days": 60, "min_mag": 4.5, "fit_start_hours": 6.0}
LIMITS = {"radius_km": (5.0, 1000.0), "pre_days": (0, 365), "post_days": (1, 365), "fit_start_hours": (0.0, 72.0)}


def default_radius_km(mag):
    """Wells & Coppersmith (1994) subsurface rupture length (all slip types), clipped to [30, 1000] km."""
    if mag is None or not math.isfinite(mag):
        return 100.0
    return float(np.clip(round(10 ** (-2.44 + 0.59 * mag)), 30, 1000))


# -----------------------------------------------------------------------------
# Selection and descriptive statistics
# -----------------------------------------------------------------------------
def select(events, ref, radius_km, pre_days, post_days, min_mag, min_depth=None, max_depth=None, data_end_ms=None):
    df, quality = events_to_frame(events)
    t0 = int(ref["time"])
    end = t0 + int(post_days * DAY_MS)
    if data_end_ms is not None:
        end = min(end, int(data_end_ms))
    dist = geo.haversine_km(ref["latitude"], ref["longitude"], df["latitude"].to_numpy(), df["longitude"].to_numpy())
    keep = (dist <= radius_km) & (df["id"] != str(ref["id"])).to_numpy()
    keep &= np.isfinite(df["mag"].to_numpy(float)) & (df["mag"].to_numpy(float) >= min_mag - 1e-9)
    depth = df["depth_km"].to_numpy(float)
    if min_depth is not None:
        keep &= np.isfinite(depth) & (depth >= min_depth)
    if max_depth is not None:
        keep &= np.isfinite(depth) & (depth <= max_depth)
    t = df["time"].to_numpy(np.int64)
    pre = keep & (t >= t0 - int(pre_days * DAY_MS)) & (t < t0)
    post = keep & (t > t0) & (t <= end)
    out = df.assign(distance_km=dist, t_days=(t - t0) / DAY_MS,
                    bearing_deg=geo.bearing_deg(ref["latitude"], ref["longitude"], df["latitude"], df["longitude"]))
    return out[pre].reset_index(drop=True), out[post].reset_index(drop=True), (end - t0) / DAY_MS, quality


def _hist(values, edges):
    counts, _ = np.histogram(np.asarray(values, float), bins=edges)
    return [{"from": round(float(a), 3), "to": round(float(b), 3), "count": int(c)}
            for a, b, c in zip(edges[:-1], edges[1:], counts, strict=True)]


def describe(pre, post, observed_days, pre_days, radius_km, min_mag):
    """Binned statistics. Zero bins are observed zeros in the queried catalogue."""
    t = post["t_days"].to_numpy()
    daily_edges = np.arange(0, math.ceil(observed_days) + 1, 1.0)
    hourly_span = min(3.0, observed_days)
    hourly_edges = np.arange(0, math.ceil(hourly_span * 24) + 1) / 24.0
    log_edges = np.unique(np.concatenate([[0.0], np.logspace(-2, np.log10(max(observed_days, 0.02)), 16)]))
    log_bins = []
    for a, b in zip(log_edges[:-1], log_edges[1:], strict=True):
        n = int(((t > a) & (t <= b)).sum())
        log_bins.append({"from": round(float(a), 4), "to": round(float(b), 4), "count": n,
                         "rate_per_day": round(n / (b - a), 4)})
    mags = post["mag"].to_numpy(float)
    mag_lo = math.floor(min(min_mag, mags.min() if len(mags) else min_mag) * 2) / 2
    mag_hi = math.ceil(max(mags.max() if len(mags) else min_mag + 1, min_mag + 0.5) * 2) / 2 + 0.5
    depth = post["depth_km"].to_numpy(float)
    depth = depth[np.isfinite(depth)]
    pre_rate = len(pre) / pre_days if pre_days > 0 else None
    post_rate = len(post) / observed_days if observed_days > 0 else None
    comparison = None
    if pre_days > 0 and observed_days > 0:
        n = len(pre) + len(post)
        share = observed_days / (observed_days + pre_days)
        p = float(stats.binomtest(len(post), n, share, alternative="greater").pvalue) if n else 1.0
        comparison = {"pre_rate_per_day": round(pre_rate, 4), "post_rate_per_day": round(post_rate, 4),
                      "rate_ratio": round(post_rate / pre_rate, 3) if pre_rate > 0 else None,
                      "p_value_post_rate_higher": float(f"{p:.4g}"),
                      "test": "Conditional binomial test of equal Poisson rates (one-sided)"}
    return {
        "counts": {"pre": int(len(pre)), "post": int(len(post))},
        "observed_post_days": round(observed_days, 3),
        "daily": _hist(t, daily_edges) if len(daily_edges) > 1 else [],
        "hourly_first_days": _hist(t, hourly_edges) if len(hourly_edges) > 1 else [],
        "log_time": log_bins,
        "cumulative": [{"t_days": round(float(x), 5), "n": i + 1} for i, x in enumerate(np.sort(t))],
        "magnitude": _hist(mags, np.arange(mag_lo, mag_hi + 0.01, 0.5)),
        "depth_km": _hist(depth, np.array([0, 10, 20, 35, 50, 70, 100, 150, 300, 700])) if len(depth) else [],
        "distance_km": _hist(post["distance_km"], np.linspace(0, radius_km, 11)),
        "pre_comparison": comparison,
        "binning_note": ("Daily bins hide the rapid early decay; the logarithmic time bins (rate per day) show it. "
                         "Bin counts depend on the magnitude threshold and on catalogue completeness."),
    }


# -----------------------------------------------------------------------------
# Omori-Utsu MLE
# -----------------------------------------------------------------------------
def _integral(K, c, p, a, b):
    if abs(p - 1.0) < 1e-6:
        return K * (math.log(b + c) - math.log(a + c))
    return K * ((b + c) ** (1 - p) - (a + c) ** (1 - p)) / (1 - p)


def _nll(theta, t, mu, a, b):
    logK, logc, p = theta
    K, c = math.exp(logK), math.exp(logc)
    lam = mu + K * (t + c) ** (-p)
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        return 1e12
    return -(np.log(lam).sum() - mu * (b - a) - _integral(K, c, p, a, b))


def _hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def fit_omori(t_days, t_start, t_end, mu=0.0):
    """Fit lambda(t) = mu + K/(t+c)^p on [t_start, t_end]. Always returns a dict with `status`."""
    t = np.sort(np.asarray(t_days, float))
    t = t[(t >= t_start) & (t <= t_end)]
    n = len(t)
    base = {"n_events": int(n), "fit_window_days": [round(t_start, 4), round(t_end, 4)],
            "background_rate_per_day": round(mu, 5), "min_events_required": MIN_FIT_EVENTS}
    if n < MIN_FIT_EVENTS:
        return {**base, "status": "insufficient_data", "reliable": False,
                "reasons": [f"Only {n} events in the fit window; at least {MIN_FIT_EVENTS} are required."]}
    if t_end - t_start <= 0.1:
        return {**base, "status": "insufficient_data", "reliable": False, "reasons": ["Fit window shorter than 0.1 day."]}

    bounds = [(math.log(1e-6), math.log(1e6)), (math.log(C_BOUNDS[0]), math.log(C_BOUNDS[1])), P_BOUNDS]
    best = None
    for c0 in (0.01, 0.1, 1.0):
        for p0 in (0.8, 1.1, 1.5):
            g = _integral(1.0, c0, p0, t_start, t_end)
            k0 = max((n - mu * (t_end - t_start)) / g, 1e-3) if g > 0 else 1.0
            res = optimize.minimize(_nll, [math.log(k0), math.log(c0), p0], args=(t, mu, t_start, t_end),
                                    method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
    logK, logc, p = best.x
    reasons, notes = [], []
    c_fixed = False
    if math.exp(logc) <= C_BOUNDS[0] * 1.05 and t_start >= 10 * C_BOUNDS[0]:
        # c is shorter than anything the data can resolve (the fit starts at t_start >> c). Fix it at the bound
        # and estimate K and p only; this is a statement about resolution, not a failed fit.
        c_fixed = True
        logc = math.log(C_BOUNDS[0])
        res2 = optimize.minimize(lambda th: _nll([th[0], logc, th[1]], t, mu, t_start, t_end), [logK, p],
                                 method="L-BFGS-B", bounds=[bounds[0], bounds[2]])
        logK, p = res2.x
        best = res2
        notes.append(f"c is not resolvable when the fit starts {t_start * 24:g} h after the event; it was fixed at "
                     f"{C_BOUNDS[0]} d and only K and p were estimated.")
    K, c = math.exp(logK), math.exp(logc)
    if not best.success:
        reasons.append(f"Optimiser did not converge ({best.message}).")
    if p <= P_BOUNDS[0] * 1.01 or p >= P_BOUNDS[1] * 0.99:
        reasons.append(f"Decay exponent p = {p:.2f} is at its search bound {P_BOUNDS}.")
    if not c_fixed and (c <= C_BOUNDS[0] * 1.05 or c >= C_BOUNDS[1] * 0.95):
        reasons.append(f"Offset c = {c:.4g} d is at its search bound {C_BOUNDS}.")

    se = cov = None
    if c_fixed:
        theta, f = np.array([logK, p]), (lambda th: _nll([th[0], logc, th[1]], t, mu, t_start, t_end))
    else:
        theta, f = np.array([logK, logc, p]), (lambda th: _nll(th, t, mu, t_start, t_end))
    H = _hessian(f, theta)
    try:
        if np.all(np.isfinite(H)) and np.all(np.linalg.eigvalsh(H) > 0):
            V = np.linalg.inv(H)
            sd = np.sqrt(np.diag(V))
            se = {"log_K": round(float(sd[0]), 4), "p": round(float(sd[-1]), 4),
                  "log_c": None if c_fixed else round(float(sd[1]), 4)}
            cov = {"parameters": ["log_K", "p"] if c_fixed else ["log_K", "log_c", "p"],
                   "matrix": [[float(f"{v:.6g}") for v in row] for row in V]}
        else:
            reasons.append("Likelihood surface is flat or not curved at the optimum (Hessian not positive definite).")
    except np.linalg.LinAlgError:
        reasons.append("Hessian could not be inverted.")

    # Goodness of fit: transformed times Lambda(t_i) should be uniform on [0, Lambda(t_end)].
    tau = np.array([mu * (x - t_start) + _integral(K, c, p, t_start, x) for x in t])
    total = mu * (t_end - t_start) + _integral(K, c, p, t_start, t_end)
    ks = stats.kstest(tau / total, "uniform")
    if ks.pvalue < 0.05:
        reasons.append(f"Transformed-time KS test rejects the model (p = {ks.pvalue:.3g}).")
    ll = -float(best.fun)
    ll_const = n * math.log(n / (t_end - t_start)) - n
    k_params = 2 if c_fixed else 3
    aic, aic_const = 2 * k_params - 2 * ll, 2 * 1 - 2 * ll_const
    if aic_const - aic < 2:
        reasons.append("A constant-rate model is not worse by AIC (difference < 2), so decay is not supported.")

    ci = None
    if se is not None:
        ci = {"p": [round(p - 1.96 * se["p"], 3), round(p + 1.96 * se["p"], 3)],
              "c_days": None if c_fixed else [round(math.exp(logc - 1.96 * se["log_c"]), 5),
                                              round(math.exp(logc + 1.96 * se["log_c"]), 5)],
              "K": [round(math.exp(logK - 1.96 * se["log_K"]), 4), round(math.exp(logK + 1.96 * se["log_K"]), 4)]}
    reliable = not reasons
    return {**base, "status": "fitted" if reliable else "unreliable", "reliable": reliable, "reasons": reasons,
            "notes": notes, "c_fixed": c_fixed,
            "parameters": {"K": round(K, 5), "c_days": round(c, 6), "p": round(p, 4)},
            "standard_errors": se, "confidence_95": ci, "covariance": cov,
            "confidence_note": ("confidence_95 gives Wald 95% confidence intervals for the PARAMETERS K, c and p "
                                "(uncertainty of the fitted curve). They are not prediction intervals for future "
                                "event counts, which must also include Poisson variation and are wider."),
            "diagnostics": {"converged": bool(best.success), "log_likelihood": round(ll, 3),
                            "aic": round(aic, 3), "aic_constant_rate": round(aic_const, 3),
                            "delta_aic_vs_constant": round(aic_const - aic, 3),
                            "ks_statistic": round(float(ks.statistic), 4), "ks_p_value": float(f"{ks.pvalue:.4g}"),
                            "expected_events": round(total, 2), "observed_events": int(n)},
            "baseline": {"model": "constant rate (homogeneous Poisson)", "rate_per_day": round(n / (t_end - t_start), 4)}}


def expected_count(fit, a, b):
    """Expected number of events in [a, b] (days) under a fitted model."""
    pr = fit["parameters"]
    return fit["background_rate_per_day"] * (b - a) + _integral(pr["K"], pr["c_days"], pr["p"], a, b)


def rate_draws(fit, a, b, n_draws=4000, seed=0):
    """
    Expected counts in [a, b] days for parameter draws (log K, [log c], p) ~ N(MLE, inverse Hessian),
    background rate fixed. None when the fit has no covariance (no reliable parameter uncertainty).
    """
    pr, mu, cov = fit.get("parameters"), fit["background_rate_per_day"], fit.get("covariance")
    if not pr or not cov:
        return None
    names = cov["parameters"]
    centre = {"log_K": math.log(pr["K"]), "log_c": math.log(pr["c_days"]), "p": pr["p"]}
    draws = np.random.default_rng(seed).multivariate_normal([centre[n] for n in names], np.asarray(cov["matrix"]), n_draws)
    col = {n: draws[:, i] for i, n in enumerate(names)}
    lc = col.get("log_c", np.full(n_draws, centre["log_c"]))
    with np.errstate(all="ignore"):
        lam = np.array([mu * (b - a) + _integral(math.exp(k), math.exp(c), p, a, b)
                        for k, c, p in zip(col["log_K"], lc, col["p"], strict=True)])
    lam = lam[np.isfinite(lam) & (lam >= 0)]
    return lam if len(lam) >= n_draws // 2 else None


def predictive_count(fit, a, b, levels=(0.8, 0.95), n_draws=4000, seed=0, include_parameter_uncertainty=True,
                     dispersion=0.0):
    """
    Predictive distribution of the NUMBER of events in [a, b] days (an aggregate count, not individual
    aftershocks) under a fitted model, by simulation: expected counts from `rate_draws` (or the point
    estimate when include_parameter_uncertainty is False), then a count per draw that is Poisson
    (dispersion = 0) or negative binomial with Var = lam + dispersion * lam^2 (gamma-Poisson). The NB
    stage represents clustering and misspecification that the Poisson process ignores; its dispersion
    must be calibrated on OTHER sequences (see analytics.evaluation). Returns None without covariance.
    """
    if not fit.get("parameters"):
        return None
    rng = np.random.default_rng(seed + 1)
    if include_parameter_uncertainty:
        lam = rate_draws(fit, a, b, n_draws, seed)
        if lam is None:
            return None
    else:
        lam = np.full(n_draws, expected_count(fit, a, b))
    if dispersion > 0:
        lam = lam * rng.gamma(1.0 / dispersion, dispersion, len(lam))
    counts = rng.poisson(lam)
    out = {"window_days": [a, b], "expected": round(float(expected_count(fit, a, b)), 3),
           "median": float(np.median(counts)), "parameter_uncertainty": include_parameter_uncertainty,
           "count_dispersion": dispersion}
    for level in levels:
        q = (1 - level) / 2
        out[f"interval_{int(round(level * 100))}"] = [int(np.quantile(counts, q, method="lower")),
                                                     int(np.quantile(counts, 1 - q, method="higher"))]
    return out


def curve(fit, edges):
    """Model rate per day averaged over each bin, only for bins inside the fit window."""
    a0, b0 = fit["fit_window_days"]
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        lo, hi = max(a, a0), min(b, b0)
        if hi - lo <= 1e-9:
            continue
        out.append({"from": round(lo, 4), "to": round(hi, 4), "rate_per_day": round(expected_count(fit, lo, hi) / (hi - lo), 5)})
    return out


def analyse(events, ref, radius_km, pre_days, post_days, min_mag, fit_start_hours=6.0, min_depth=None,
            max_depth=None, data_end_ms=None, truncated=False):
    pre, post, observed_days, quality = select(events, ref, radius_km, pre_days, post_days, min_mag,
                                               min_depth, max_depth, data_end_ms)
    stats_ = describe(pre, post, observed_days, pre_days, radius_km, min_mag)
    mu = len(pre) / pre_days if pre_days > 0 else 0.0
    if truncated:
        fit = {"status": "incomplete_data", "reliable": False,
               "reasons": ["The USGS query reached its response limit, so the event list is incomplete."]}
    else:
        fit = fit_omori(post["t_days"].to_numpy(), fit_start_hours / 24.0, observed_days, mu)
    if fit.get("reliable"):
        edges = [b["from"] for b in stats_["log_time"]] + [stats_["log_time"][-1]["to"]]
        fit["curve_log_time"] = curve(fit, edges)
    members = [{"id": r.id, "time": int(r.time), "latitude": r.latitude, "longitude": r.longitude,
                "depth_km": None if not np.isfinite(r.depth_km) else r.depth_km, "mag": r.mag,
                "t_days": round(r.t_days, 5), "distance_km": round(r.distance_km, 1)}
               for r in post.itertuples()]
    return {"statistics": stats_, "fit": fit, "events": members, "pre_event_count": int(len(pre)),
            "data_quality": quality}
