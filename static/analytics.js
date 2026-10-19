/*
 * TerraWatcher ML view helpers: response validation, request supersession,
 * colour scales and plain-language explanations. No DOM access, so every
 * function is unit-tested in Node (tests/frontend/analytics.test.js).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TerraML = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
  const num = (v) => typeof v === 'number' && Number.isFinite(v);

  // ---------------------------------------------------------------------------
  // Response validation: the views render only payloads that pass these checks.
  // ---------------------------------------------------------------------------
  function fail(msg) { return { ok: false, error: `Unexpected response from the server: ${msg}.` }; }

  function validateSequences(b) {
    if (!isObj(b) || !Array.isArray(b.sequences) || !Array.isArray(b.events) || !isObj(b.summary)) return fail('sequence list missing');
    for (const s of b.sequences) {
      if (typeof s.id !== 'string' || !num(s.event_count) || !Array.isArray(s.member_ids) || !num(s.start) || !num(s.end)) return fail('malformed sequence');
    }
    return { ok: true };
  }

  function validateForecast(b) {
    if (!isObj(b) || !Array.isArray(b.regions) || typeof b.origin !== 'string' || !isObj(b.model)) return fail('forecast list missing');
    for (const r of b.regions) {
      if (typeof r.region !== 'string' || !num(r.expected) || r.expected < 0 || !isObj(r.bounds)
          || !Array.isArray(r.interval_80) || r.interval_80[0] > r.interval_80[1]) return fail('malformed regional forecast');
    }
    return { ok: true };
  }

  function validateAnomalies(b) {
    if (!isObj(b) || !Array.isArray(b.results) || !isObj(b.observation_window) || !isObj(b.reference_window)) return fail('anomaly results missing');
    const allowed = new Set(['elevated', 'not_elevated', 'insufficient_reference', 'data_incomplete']);
    for (const r of b.results) {
      if (typeof r.region !== 'string' || !allowed.has(r.status) || !num(r.observed)) return fail('malformed anomaly row');
    }
    return { ok: true };
  }

  function validateAftershock(b) {
    if (!isObj(b) || !isObj(b.reference_event) || !isObj(b.statistics) || !isObj(b.fit) || !Array.isArray(b.events)) return fail('aftershock analysis missing');
    if (typeof b.fit.status !== 'string') return fail('fit status missing');
    return { ok: true };
  }

  function validatePerformance(b) {
    if (!isObj(b) || !isObj(b.forecast)) return fail('evaluation summary missing');
    return { ok: true };
  }

  // ---------------------------------------------------------------------------
  // Superseded requests: only the newest request of a kind may update the view.
  // ---------------------------------------------------------------------------
  function createTracker() {
    const latest = new Map();
    return {
      start(kind) { const token = (latest.get(kind) || 0) + 1; latest.set(kind, token); return token; },
      isCurrent(kind, token) { return latest.get(kind) === token; },
    };
  }

  // ---------------------------------------------------------------------------
  // Colours and map specs
  // ---------------------------------------------------------------------------
  const SEQ_COLORS = ['#4fd1c5', '#ffb454', '#c084fc', '#6b8cff', '#f472b6', '#a3e635', '#fb923c', '#38bdf8'];
  const NOISE_COLOR = '#5b6d7a';

  function sequenceColor(index) { return SEQ_COLORS[((index % SEQ_COLORS.length) + SEQ_COLORS.length) % SEQ_COLORS.length]; }

  /** One style per event for the sequence layer; the selected sequence is emphasised and others dimmed. */
  function sequenceLayerSpec(events, sequences, selectedId) {
    const order = new Map((sequences || []).map((s, i) => [s.id, i]));
    return (events || []).map((e) => {
      const seq = e.sequence || null;
      const inSel = selectedId && seq === selectedId;
      const dim = selectedId && !inSel;
      return {
        id: e.id, latitude: e.latitude, longitude: e.longitude, sequence: seq,
        color: seq ? sequenceColor(order.get(seq) ?? 0) : NOISE_COLOR,
        radius: seq ? (inSel ? 6 : 4) : 2.5,
        opacity: dim ? 0.18 : seq ? 0.85 : 0.45,
        highlighted: !!inSel,
      };
    });
  }

  /** Sequential colour for an expected weekly count (log scale, 0.05 .. 20+). */
  function forecastColor(expected) {
    if (!num(expected)) return '#33444f';
    const stops = ['#16323b', '#1f5f5f', '#2f8f7f', '#7cc48d', '#e8d36a', '#ffb454', '#ff7a59'];
    const t = Math.max(0, Math.min(1, (Math.log10(Math.max(expected, 0.05)) + 1.3) / 2.6));
    return stops[Math.min(stops.length - 1, Math.floor(t * stops.length))];
  }

  function timeColor(fraction) {
    const stops = ['#f9f871', '#ffb454', '#ff7a59', '#c084fc', '#6b8cff'];
    const t = Math.max(0, Math.min(0.999, num(fraction) ? fraction : 0));
    return stops[Math.floor(t * stops.length)];
  }

  function boundsToLatLngs(b) { return [[b.lat_min, b.lon_min], [b.lat_max, b.lon_max]]; }

  // ---------------------------------------------------------------------------
  // Formatting and explanations
  // ---------------------------------------------------------------------------
  function formatP(p) {
    if (!num(p)) return '—';
    if (p < 1e-4) return p.toExponential(1);
    return p < 0.01 ? p.toFixed(4) : p.toFixed(3);
  }

  function formatDuration(hours) {
    if (!num(hours)) return '—';
    if (hours < 1) return `${Math.round(hours * 60)} min`;
    if (hours < 48) return `${hours.toFixed(1)} h`;
    return `${(hours / 24).toFixed(1)} d`;
  }

  const STATUS_LABELS = {
    elevated: 'Elevated observed activity',
    not_elevated: 'Within reference range',
    insufficient_reference: 'Too little reference data',
    data_incomplete: 'Data incomplete — not assessed',
  };
  function anomalyStatusLabel(status) { return STATUS_LABELS[status] || 'Unknown'; }

  /** Why a region is (or is not) flagged, from the fields the API returned. */
  function explainAnomaly(row, summary) {
    if (!row) return '';
    const w = `${summary.window_weeks}-week window`;
    if (row.status === 'insufficient_reference') {
      return `${row.observed} M${summary.threshold}+ events in the ${w}, but only ${row.reference_total} in the ${summary.reference_weeks}-week reference period, which is too few to define what is typical here.`;
    }
    const base = `${row.observed} M${summary.threshold}+ events were observed in the ${w}; the reference period predicts about ${Number(row.expected).toFixed(1)} (ratio ${num(row.ratio) ? row.ratio.toFixed(1) : '—'}×).`;
    const stat = ` One-sided p = ${formatP(row.p_value)}; after Benjamini–Hochberg correction across ${summary.regions_tested} regions, q = ${formatP(row.q_value)} (threshold ${summary.fdr}).`;
    if (row.status === 'elevated') return `${base}${stat} The count is unlikely under this region's own recent pattern, so it is flagged as elevated observed activity. This describes what happened; it is not a prediction.`;
    if (row.status === 'data_incomplete') return `${base}${stat} Possible gaps in the feed were detected, so a normal-looking count cannot be confirmed as normal.`;
    return `${base}${stat} This is within the range expected from the reference period.`;
  }

  /** One-paragraph summary of an Omori-Utsu fit result. */
  function explainFit(fit) {
    if (!fit) return '';
    if (fit.status === 'fitted') {
      const p = fit.parameters, ci = fit.confidence_95 || {};
      return `Decay exponent p = ${p.p.toFixed(2)}${ci.p ? ` (95% CI ${ci.p[0]}–${ci.p[1]})` : ''}. The fit converged, its parameters are not at search limits, the transformed-time KS test does not reject it (p = ${formatP(fit.diagnostics.ks_p_value)}), and it beats a constant-rate model by ΔAIC = ${fit.diagnostics.delta_aic_vs_constant.toFixed(1)}.`;
    }
    const reasons = (fit.reasons || []).join(' ');
    if (fit.status === 'insufficient_data') return `No decay model was fitted. ${reasons}`;
    if (fit.status === 'incomplete_data') return `No decay model was fitted. ${reasons}`;
    return `A fit was attempted but is not reliable, so no curve is shown. ${reasons}`;
  }

  /** Summary rows for comparing sequences analysed with possibly different parameters. */
  function compareRows(list) {
    return (list || []).map((a) => ({
      id: a.reference_event.id,
      label: `M${num(a.reference_event.mag) ? a.reference_event.mag.toFixed(1) : '?'} ${a.reference_event.place || a.reference_event.id}`,
      post: a.statistics.counts.post,
      days: a.statistics.observed_post_days,
      p: a.fit.status === 'fitted' ? a.fit.parameters.p : null,
      status: a.fit.status,
      params: `${a.parameters.radius_km} km · M${a.parameters.min_mag}+ · ${a.parameters.post_days} d`,
    }));
  }

  function sameDefinitions(list) {
    if (!list || list.length < 2) return true;
    const key = (a) => `${a.parameters.min_mag}|${a.parameters.post_days}|${a.parameters.fit_start_hours}|${a.parameters.min_depth}|${a.parameters.max_depth}`;
    return list.every((a) => key(a) === key(list[0]));
  }

  return {
    validateSequences, validateForecast, validateAnomalies, validateAftershock, validatePerformance,
    createTracker, sequenceColor, sequenceLayerSpec, forecastColor, timeColor, boundsToLatLngs,
    formatP, formatDuration, anomalyStatusLabel, explainAnomaly, explainFit, compareRows, sameDefinitions,
    SEQ_COLORS, NOISE_COLOR,
  };
});
