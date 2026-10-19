/*
 * TerraWatcher core helpers.
 * Pure functions with no DOM or Leaflet access, so they can be unit-tested in
 * Node (`node --test tests/frontend`) and reused by dashboard.js in the browser.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TerraCore = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const DEPTH_CLASSES = [
    { key: 'shallow', label: 'Shallow (< 70 km)', min: -Infinity, max: 70 },
    { key: 'intermediate', label: 'Intermediate (70–300 km)', min: 70, max: 300 },
    { key: 'deep', label: 'Deep (> 300 km)', min: 300, max: Infinity },
  ];
  const MAG_BINS = [
    { label: '< 2', min: -Infinity, max: 2 },
    { label: '2–2.9', min: 2, max: 3 },
    { label: '3–3.9', min: 3, max: 4 },
    { label: '4–4.9', min: 4, max: 5 },
    { label: '5–5.9', min: 5, max: 6 },
    { label: '6+', min: 6, max: Infinity },
  ];

  const toNum = (v) => (v === null || v === undefined || v === '' ? null : Number.isFinite(Number(v)) ? Number(v) : null);

  function isValidCoord(lat, lon) {
    return Number.isFinite(lat) && Number.isFinite(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
  }

  /** Validate the /api/earthquakes payload. Records without usable coordinates are rejected, never guessed. */
  function normalizeEvents(payload) {
    const raw = payload && Array.isArray(payload.events) ? payload.events : null;
    if (!raw) return { events: [], rejected: 0, malformed: true };
    const events = [];
    let rejected = 0;
    raw.forEach((e, i) => {
      const lat = toNum(e && e.latitude);
      const lon = toNum(e && e.longitude);
      const time = toNum(e && e.time);
      if (!isValidCoord(lat, lon) || time === null) { rejected += 1; return; }
      events.push({
        id: e.id ? String(e.id) : `event-${i}`,
        time,
        latitude: lat,
        longitude: lon,
        depth: toNum(e.depth_km),
        mag: toNum(e.mag),
        magType: e.mag_type || null,
        place: typeof e.place === 'string' && e.place.trim() ? e.place : null,
        status: e.status || null,
        tsunami: e.tsunami === true,
        alert: e.alert || null,
        url: typeof e.url === 'string' && /^https:\/\/earthquake\.usgs\.gov\//.test(e.url) ? e.url : null,
      });
    });
    return { events, rejected, malformed: false };
  }

  function depthClass(depth) {
    if (!Number.isFinite(depth)) return null;
    return DEPTH_CLASSES.find((c) => depth >= c.min && depth < c.max).key;
  }

  function filterByDepth(events, key) {
    if (!key || key === 'all') return events;
    return events.filter((e) => depthClass(e.depth) === key);
  }

  function median(values) {
    if (!values.length) return null;
    const s = [...values].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  /** Statistics shown in the dashboard, computed only from the events actually returned. */
  function summarize(events) {
    const mags = events.map((e) => e.mag).filter(Number.isFinite);
    const depths = events.map((e) => e.depth).filter(Number.isFinite);
    let strongest = null;
    events.forEach((e) => { if (Number.isFinite(e.mag) && (!strongest || e.mag > strongest.mag)) strongest = e; });
    return {
      count: events.length,
      withMagnitude: mags.length,
      strongest,
      medianDepth: median(depths),
      tsunamiFlagged: events.filter((e) => e.tsunami).length,
      magBins: MAG_BINS.map((b) => ({ label: b.label, count: mags.filter((m) => m >= b.min && m < b.max).length })),
      depthBins: DEPTH_CLASSES.map((c) => ({ key: c.key, label: c.label, count: depths.filter((d) => d >= c.min && d < c.max).length })),
    };
  }

  function magColor(mag) {
    if (!Number.isFinite(mag)) return '#7d93a3';
    if (mag >= 6) return '#ff4d6d';
    if (mag >= 5) return '#ff7a59';
    if (mag >= 4) return '#ffb454';
    if (mag >= 3) return '#e8d36a';
    return '#4fd1c5';
  }

  function magRadius(mag) {
    if (!Number.isFinite(mag)) return 3.5;
    return Math.max(3, Math.min(16, 2.2 + Math.max(mag, 0) * 1.7));
  }

  function fmt(n, digits = 1, fallback = '—') {
    return Number.isFinite(n) ? n.toFixed(digits) : fallback;
  }

  function formatUtc(ms) {
    if (!Number.isFinite(ms)) return 'Time unavailable';
    const d = new Date(ms);
    return `${d.toISOString().slice(0, 16).replace('T', ' ')} UTC`;
  }

  function formatAgo(ms, now = Date.now()) {
    if (!Number.isFinite(ms)) return '—';
    const s = Math.max(0, Math.round((now - ms) / 1000));
    if (s < 60) return `${s}s ago`;
    const m = Math.round(s / 60);
    if (m < 60) return `${m} min ago`;
    const h = Math.round(m / 60);
    if (h < 48) return `${h} h ago`;
    return `${Math.round(h / 24)} d ago`;
  }

  function formatCoord(lat, lon) {
    if (!isValidCoord(lat, lon)) return '—';
    return `${Math.abs(lat).toFixed(3)}°${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(3)}°${lon >= 0 ? 'E' : 'W'}`;
  }

  function bandLabel(band) {
    return { above_base_rate: 'Above base rate', near_base_rate: 'Near base rate', below_base_rate: 'Below base rate' }[band] || 'Unclassified';
  }

  /** Error message from a JSON error body ({error, code}) or a generic fallback. */
  function errorMessage(status, body) {
    if (body && typeof body.error === 'string' && body.error) return body.error;
    if (status === 0) return 'The TerraWatcher server could not be reached.';
    return `Request failed with HTTP ${status}.`;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  return {
    DEPTH_CLASSES, MAG_BINS, isValidCoord, normalizeEvents, depthClass, filterByDepth, median, summarize,
    magColor, magRadius, fmt, formatUtc, formatAgo, formatCoord, bandLabel, errorMessage, escapeHtml,
  };
});
