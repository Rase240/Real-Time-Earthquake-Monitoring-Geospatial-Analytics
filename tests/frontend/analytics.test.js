// Unit tests for static/analytics.js and static/charts.js. Run with: node --test "tests/frontend/*.test.js"
const test = require('node:test');
const assert = require('node:assert/strict');
const ML = require('../../static/analytics.js');
const CH = require('../../static/charts.js');

const seqPayload = () => ({
  summary: { events_considered: 3, sequence_count: 1, noise_events: 1 },
  sequences: [{ id: 'S-1', event_count: 2, member_ids: ['a', 'b'], start: 1, end: 2 }],
  events: [{ id: 'a', latitude: 1, longitude: 1, sequence: 'S-1' }, { id: 'b', latitude: 1, longitude: 1.1, sequence: 'S-1' },
    { id: 'c', latitude: 5, longitude: 5, sequence: null }],
});

// ------------------------------------------------------------------ validation of API responses
test('valid sequence payload passes; malformed ones are rejected with a message', () => {
  assert.equal(ML.validateSequences(seqPayload()).ok, true);
  for (const bad of [null, {}, { sequences: 'x', events: [], summary: {} }, { ...seqPayload(), sequences: [{ id: 3 }] }]) {
    const v = ML.validateSequences(bad);
    assert.equal(v.ok, false);
    assert.match(v.error, /Unexpected response/);
  }
});

test('forecast validation rejects negative or inverted intervals', () => {
  const ok = { origin: '2026-01-01T00:00:00Z', model: {}, regions: [{ region: 'R+00+000', expected: 1.2, bounds: {}, interval_80: [0, 3] }] };
  assert.equal(ML.validateForecast(ok).ok, true);
  assert.equal(ML.validateForecast({ ...ok, regions: [{ ...ok.regions[0], expected: -1 }] }).ok, false);
  assert.equal(ML.validateForecast({ ...ok, regions: [{ ...ok.regions[0], interval_80: [4, 2] }] }).ok, false);
  assert.equal(ML.validateForecast({ ...ok, regions: [{ ...ok.regions[0], expected: NaN }] }).ok, false);
});

test('anomaly validation only accepts documented statuses', () => {
  const ok = { observation_window: {}, reference_window: {}, results: [{ region: 'R', status: 'elevated', observed: 3 }] };
  assert.equal(ML.validateAnomalies(ok).ok, true);
  assert.equal(ML.validateAnomalies({ ...ok, results: [{ region: 'R', status: 'danger', observed: 3 }] }).ok, false);
});

test('aftershock and performance validation', () => {
  assert.equal(ML.validateAftershock({ reference_event: {}, statistics: {}, fit: { status: 'fitted' }, events: [] }).ok, true);
  assert.equal(ML.validateAftershock({ reference_event: {}, statistics: {}, fit: {}, events: [] }).ok, false);
  assert.equal(ML.validatePerformance({ forecast: {} }).ok, true);
  assert.equal(ML.validatePerformance([]).ok, false);
});

// ------------------------------------------------------------------ superseded requests
test('only the newest request of a kind is current', () => {
  const t = ML.createTracker();
  const first = t.start('fc');
  const second = t.start('fc');
  const other = t.start('an');
  assert.equal(t.isCurrent('fc', first), false);
  assert.equal(t.isCurrent('fc', second), true);
  assert.equal(t.isCurrent('an', other), true);
});

// ------------------------------------------------------------------ sequence layer
test('sequence layer highlights the selected sequence and dims the rest', () => {
  const p = seqPayload();
  const none = ML.sequenceLayerSpec(p.events, p.sequences, null);
  assert.equal(none.find((s) => s.id === 'c').color, ML.NOISE_COLOR);
  assert.ok(none.every((s) => !s.highlighted));
  const sel = ML.sequenceLayerSpec(p.events, p.sequences, 'S-1');
  assert.ok(sel.find((s) => s.id === 'a').highlighted);
  assert.ok(sel.find((s) => s.id === 'c').opacity < sel.find((s) => s.id === 'a').opacity);
  assert.equal(sel.find((s) => s.id === 'a').color, ML.sequenceColor(0));
});

test('colour scales are monotone or bounded', () => {
  assert.notEqual(ML.forecastColor(0.05), ML.forecastColor(20));
  assert.equal(ML.forecastColor(NaN), '#33444f');
  assert.equal(ML.timeColor(-1), ML.timeColor(0));
  assert.equal(ML.sequenceColor(8), ML.sequenceColor(0));
});

// ------------------------------------------------------------------ explanations
const summary = { window_weeks: 4, threshold: 4.5, reference_weeks: 52, regions_tested: 120, fdr: 0.05 };

test('anomaly explanation states observed, expected, test and correction without danger language', () => {
  const text = ML.explainAnomaly({ status: 'elevated', observed: 30, expected: 4.2, ratio: 7.1, p_value: 1e-8, q_value: 2e-6 }, summary);
  assert.match(text, /30 M4\.5\+ events/);
  assert.match(text, /about 4\.2/);
  assert.match(text, /Benjamini–Hochberg/);
  assert.match(text, /not a prediction/);
  assert.doesNotMatch(text, /danger|risk score/i);
  assert.match(ML.explainAnomaly({ status: 'insufficient_reference', observed: 3, reference_total: 1 }, summary), /too few/);
  assert.match(ML.explainAnomaly({ status: 'data_incomplete', observed: 0, expected: 1, ratio: 0, p_value: 1, q_value: 1 }, summary), /gaps/);
  assert.equal(ML.anomalyStatusLabel('elevated'), 'Elevated observed activity');
});

test('fit explanation is honest about failed and insufficient fits', () => {
  const fitted = { status: 'fitted', parameters: { p: 1.1 }, confidence_95: { p: [0.9, 1.3] }, diagnostics: { ks_p_value: 0.4, delta_aic_vs_constant: 12.3 } };
  assert.match(ML.explainFit(fitted), /p = 1\.10 \(95% CI 0\.9–1\.3\)/);
  assert.match(ML.explainFit({ status: 'insufficient_data', reasons: ['Only 5 events'] }), /No decay model was fitted\. Only 5 events/);
  assert.match(ML.explainFit({ status: 'unreliable', reasons: ['KS rejects'] }), /not reliable, so no curve is shown/);
});

test('sequence comparison flags inconsistent definitions', () => {
  const a = { reference_event: { id: 'x', mag: 7.1, place: 'P' }, statistics: { counts: { post: 10 }, observed_post_days: 30 },
    fit: { status: 'fitted', parameters: { p: 1.2 } }, parameters: { radius_km: 50, min_mag: 4.5, post_days: 30, fit_start_hours: 6 } };
  const b = { ...a, reference_event: { id: 'y', mag: 6.5 }, fit: { status: 'unreliable' } };
  assert.equal(ML.sameDefinitions([a, b]), true);
  assert.equal(ML.sameDefinitions([a, { ...b, parameters: { ...b.parameters, min_mag: 3 } }]), false);
  const rows = ML.compareRows([a, b]);
  assert.equal(rows[0].p, 1.2);
  assert.equal(rows[1].p, null);
});

test('formatting helpers', () => {
  assert.equal(ML.formatP(1e-9), '1.0e-9');
  assert.equal(ML.formatP(0.0123), '0.012');
  assert.equal(ML.formatP(null), '—');
  assert.equal(ML.formatDuration(0.5), '30 min');
  assert.equal(ML.formatDuration(72), '3.0 d');
});

// ------------------------------------------------------------------ charts
test('nice ticks cover the range with round steps', () => {
  assert.deepEqual(CH.niceTicks(0, 10, 5), [0, 2, 4, 6, 8, 10]);
  assert.deepEqual(CH.logTicks(0.01, 100), [0.01, 0.1, 1, 10, 100]);
  assert.deepEqual(CH.niceTicks(NaN, 1), []);
});

test('line chart is accessible SVG with labelled axes and escaped text', () => {
  const svg = CH.lineChart({ title: 'Counts <test>', x: { label: 'Week' }, y: { label: 'Events' },
    series: [{ name: 'Observed', kind: 'bar', data: [{ x: 1, y: 2 }, { x: 2, y: 0 }] }, { name: 'Forecast', kind: 'line', data: [{ x: 1, y: 1.5 }, { x: 2, y: 1 }] }],
    markers: [{ x: 3, y: 1, lo: 0, hi: 3, label: 'next', legend: 'Next week' }] });
  assert.match(svg, /role="img"/);
  assert.match(svg, /<title>Counts &lt;test&gt;<\/title>/);
  assert.match(svg, />Week</);
  assert.match(svg, />Events</);
  assert.equal((svg.match(/class="bar"/g) || []).length, 2);
  assert.match(svg, /class="errbar"/);
  assert.match(svg, /chart-legend/);
});

test('charts show an explicit empty state instead of an empty frame', () => {
  assert.match(CH.lineChart({ series: [], empty: 'Nothing here' }), /Nothing here/);
  assert.match(CH.barChart({ bins: [{ label: 'a', count: 0 }] }), /No values/);
});

test('log axes skip non-positive values', () => {
  const svg = CH.lineChart({ x: { type: 'log' }, y: { log: true }, series: [{ name: 'r', kind: 'dot', data: [{ x: 0, y: 1 }, { x: 0.1, y: 0 }, { x: 1, y: 10 }] }] });
  assert.equal((svg.match(/class="dot"/g) || []).length, 1);
});

test('histogram bins carry the labels and counts', () => {
  const bins = CH.histogramBins([{ from: 4, to: 4.5, count: 3 }, { from: 4.5, to: 5, count: 1 }]);
  assert.deepEqual(bins, [{ label: '4.0–4.5', count: 3 }, { label: '4.5–5.0', count: 1 }]);
  const svg = CH.barChart({ bins, xLabel: 'Magnitude', yLabel: 'Events' });
  assert.equal((svg.match(/class="bar"/g) || []).length, 2);
  assert.match(svg, /4\.0–4\.5: 3/);
});
