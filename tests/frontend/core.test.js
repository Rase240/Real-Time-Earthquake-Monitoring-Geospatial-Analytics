// Unit tests for static/core.js. Run with: node --test tests/frontend
const test = require('node:test');
const assert = require('node:assert/strict');
const C = require('../../static/core.js');

const ev = (over = {}) => ({ id: 'a', time: 1700000000000, latitude: 38.1, longitude: 142.5, depth_km: 24, mag: 5.1, place: 'Japan', url: 'https://earthquake.usgs.gov/earthquakes/eventpage/a', ...over });

test('normalizeEvents keeps valid events and coordinates as given', () => {
  const { events, rejected, malformed } = C.normalizeEvents({ events: [ev()] });
  assert.equal(malformed, false);
  assert.equal(rejected, 0);
  assert.deepEqual([events[0].latitude, events[0].longitude, events[0].depth, events[0].mag], [38.1, 142.5, 24, 5.1]);
});

test('normalizeEvents rejects missing or out-of-range coordinates instead of guessing', () => {
  const { events, rejected } = C.normalizeEvents({ events: [ev({ latitude: null }), ev({ longitude: 200 }), ev({ time: 'x' }), ev({ id: 'ok' })] });
  assert.equal(rejected, 3);
  assert.deepEqual(events.map((e) => e.id), ['ok']);
});

test('normalizeEvents tolerates missing optional fields', () => {
  const { events } = C.normalizeEvents({ events: [ev({ mag: null, depth_km: undefined, place: '', url: 'javascript:alert(1)' })] });
  const e = events[0];
  assert.equal(e.mag, null);
  assert.equal(e.depth, null);
  assert.equal(e.place, null);
  assert.equal(e.url, null, 'non-USGS URLs are dropped');
});

test('normalizeEvents flags malformed payloads', () => {
  for (const payload of [null, {}, { events: 'nope' }, []]) {
    assert.equal(C.normalizeEvents(payload).malformed, true);
  }
});

test('summarize computes statistics from returned events only', () => {
  const { events } = C.normalizeEvents({ events: [ev({ id: '1', mag: 2.1, depth_km: 5 }), ev({ id: '2', mag: 6.3, depth_km: 400, place: 'Deep one' }), ev({ id: '3', mag: null, depth_km: 100 })] });
  const s = C.summarize(events);
  assert.equal(s.count, 3);
  assert.equal(s.withMagnitude, 2);
  assert.equal(s.strongest.place, 'Deep one');
  assert.equal(s.medianDepth, 100);
  assert.deepEqual(s.depthBins.map((b) => b.count), [1, 1, 1]);
  assert.equal(s.magBins.find((b) => b.label === '6+').count, 1);
});

test('summarize handles an empty list', () => {
  const s = C.summarize([]);
  assert.equal(s.count, 0);
  assert.equal(s.strongest, null);
  assert.equal(s.medianDepth, null);
});

test('depth filter', () => {
  const { events } = C.normalizeEvents({ events: [ev({ id: 's', depth_km: 10 }), ev({ id: 'd', depth_km: 500 }), ev({ id: 'n', depth_km: null })] });
  assert.deepEqual(C.filterByDepth(events, 'deep').map((e) => e.id), ['d']);
  assert.equal(C.filterByDepth(events, 'all').length, 3);
});

test('formatting helpers', () => {
  assert.equal(C.formatUtc(0), '1970-01-01 00:00 UTC');
  assert.equal(C.formatUtc(NaN), 'Time unavailable');
  assert.equal(C.formatCoord(-33.45, -70.66), '33.450°S, 70.660°W');
  assert.equal(C.formatAgo(1000, 61000), '1 min ago');
  assert.equal(C.fmt(null), '—');
  assert.equal(C.escapeHtml('<b x="1">'), '&lt;b x=&quot;1&quot;&gt;');
});

test('errorMessage prefers the API error string', () => {
  assert.equal(C.errorMessage(400, { error: "Parameter 'days' must be between 1 and 30" }), "Parameter 'days' must be between 1 and 30");
  assert.match(C.errorMessage(502, null), /502/);
  assert.match(C.errorMessage(0), /could not be reached/);
});

test('magnitude styling is monotonic', () => {
  assert.ok(C.magRadius(6) > C.magRadius(3));
  assert.notEqual(C.magColor(6.5), C.magColor(2));
  assert.equal(C.magRadius(null), 3.5);
});
