/*
 * TerraWatcher dashboard: state, API calls, Leaflet map and rendering.
 * Pure helpers (validation, statistics, formatting) live in core.js.
 */
(() => {
  'use strict';
  const C = window.TerraCore;
  const $ = (id) => document.getElementById(id);
  const esc = C.escapeHtml;

  const REFRESH_MS = 180000;        // automatic refresh every 3 minutes (skipped while the tab is hidden)
  const REQUEST_TIMEOUT_MS = 25000;

  const state = {
    days: 1,
    minMag: 2.5,
    depth: 'all',
    allEvents: [],        // events returned by /api/earthquakes (validated)
    events: [],           // allEvents after the client-side depth filter
    feed: null,           // metadata of the last successful feed response
    hotspots: [],
    selectedId: null,
    layersOn: { events: true, clusters: true, plates: true },
    map: null,
    layers: {},
    markers: new Map(),
    feedController: null,
  };

  // ---------------------------------------------------------------------------
  // Networking
  // ---------------------------------------------------------------------------
  async function fetchJson(url, { signal } = {}) {
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), REQUEST_TIMEOUT_MS);
    const onAbort = () => timeout.abort();
    if (signal) signal.addEventListener('abort', onAbort);
    let response;
    try {
      response = await fetch(url, { headers: { Accept: 'application/json' }, signal: timeout.signal });
    } catch (err) {
      if (signal && signal.aborted) throw Object.assign(new Error('superseded'), { superseded: true });
      throw new Error(err.name === 'AbortError' ? 'The request timed out.' : C.errorMessage(0));
    } finally {
      clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', onAbort);
    }
    let body = null;
    try { body = await response.json(); } catch (_) { /* handled below */ }
    if (!response.ok) throw Object.assign(new Error(C.errorMessage(response.status, body)), { status: response.status, body });
    if (body === null) throw new Error('The server returned a response that is not valid JSON.');
    return body;
  }

  function toast(message, kind = 'info') {
    const node = document.createElement('div');
    node.className = `toast ${kind}`;
    node.textContent = message;
    $('toasts').appendChild(node);
    setTimeout(() => node.remove(), 5000);
  }

  // ---------------------------------------------------------------------------
  // Map
  // ---------------------------------------------------------------------------
  function initMap() {
    if (!window.L) {
      $('map').innerHTML = '<div class="state error">The map library (Leaflet) could not be loaded. Check your network connection; the event list still works.</div>';
      return;
    }
    if (state.map) return;  // guard against double initialisation
    const map = L.map('map', {
      preferCanvas: true,
      worldCopyJump: false,
      minZoom: 1,
      maxZoom: 12,
      zoomSnap: 0.5,
      maxBounds: [[-90, -200], [90, 200]],
      maxBoundsViscosity: 0.8,
      zoomControl: true,
    }).fitBounds([[-58, -170], [72, 190]]);
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 16,
      attribution: 'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
    }).addTo(map);

    state.layers.plates = L.layerGroup().addTo(map);
    state.layers.clusters = L.layerGroup().addTo(map);
    state.layers.events = L.layerGroup().addTo(map);
    state.map = map;

    map.on('moveend', () => {
      const c = map.getCenter();
      $('map-view').textContent = `centre ${C.formatCoord(c.lat, ((c.lng + 540) % 360) - 180)} · zoom ${map.getZoom()}`;
    });
    map.on('click', (e) => openPointAnalysis(e.latlng.lat, ((e.latlng.lng + 540) % 360) - 180));
    map.fire('moveend');
  }

  async function loadPlates() {
    if (!state.map) return;
    try {
      const geo = await fetchJson('/api/plates');
      L.geoJSON(geo, { interactive: false, style: { color: '#6b8cff', weight: 1, opacity: 0.45 } }).addTo(state.layers.plates);
    } catch (err) {
      toast(`Plate boundary layer unavailable: ${err.message}`, 'error');
    }
  }

  function renderMarkers() {
    if (!state.map) return;
    state.layers.events.clearLayers();
    state.markers.clear();
    // Draw small events first so large ones stay on top.
    const ordered = [...state.events].sort((a, b) => (a.mag ?? -9) - (b.mag ?? -9));
    ordered.forEach((e) => {
      const color = C.magColor(e.mag);
      const marker = L.circleMarker([e.latitude, e.longitude], {
        radius: C.magRadius(e.mag), color, weight: 1, fillColor: color, fillOpacity: 0.65, bubblingMouseEvents: false,
      });
      marker.bindTooltip(`M ${esc(C.fmt(e.mag))} · ${esc(e.place || 'Location unavailable')}`, { direction: 'top', offset: [0, -4] });
      marker.on('click', () => selectEvent(e.id, { pan: false }));
      marker.addTo(state.layers.events);
      state.markers.set(e.id, marker);
    });
    highlightSelected();
  }

  function renderClusters() {
    if (!state.map) return;
    state.layers.clusters.clearLayers();
    state.hotspots.forEach((h) => {
      const b = h.cell;
      if (!b) return;
      const rect = L.rectangle([[b.lat_min, b.lon_min], [b.lat_max, b.lon_max]], {
        color: '#c084fc', weight: 1, dashArray: '3 3', fillColor: '#c084fc', fillOpacity: 0.08, bubblingMouseEvents: false,
      });
      const score = Number.isFinite(h.model_probability) ? `${(h.model_probability * 100).toFixed(1)}% (conditional)` : 'unavailable';
      const ring = L.circleMarker([h.latitude, h.longitude], {
        radius: Math.min(22, 6 + Math.sqrt(h.event_count) * 1.6), color: '#c084fc', weight: 1.5, fill: false, opacity: 0.85,
        bubblingMouseEvents: false,
      });
      const popup = (
        `<div class="pop"><div class="pop-kicker">Activity cluster · observed</div>
         <strong>${h.event_count} events M4.5+ in 30 days</strong>
         <dl><dt>Cell</dt><dd>${b.lat_min}° to ${b.lat_max}°, ${b.lon_min}° to ${b.lon_max}°</dd>
         <dt>Largest</dt><dd>M ${esc(C.fmt(h.max_mag))}</dd>
         <dt>Latest</dt><dd>${esc(C.formatUtc(h.latest_time))}</dd>
         <dt>Median depth</dt><dd>${esc(C.fmt(h.median_depth_km, 0))} km</dd>
         <dt>Model severity score</dt><dd>${esc(score)}</dd></dl>
         <p class="pop-note">Recent observed activity is not a forecast of future earthquakes.</p></div>`);
      rect.bindPopup(popup);
      ring.bindPopup(popup);
      rect.addTo(state.layers.clusters);
      ring.addTo(state.layers.clusters);
    });
  }

  function applyLayerVisibility() {
    if (!state.map) return;
    Object.entries(state.layersOn).forEach(([name, on]) => {
      const layer = state.layers[name];
      if (!layer) return;
      if (on && !state.map.hasLayer(layer)) layer.addTo(state.map);
      if (!on && state.map.hasLayer(layer)) state.map.removeLayer(layer);
    });
  }

  function highlightSelected() {
    state.markers.forEach((m, id) => {
      const sel = id === state.selectedId;
      m.setStyle({ weight: sel ? 3 : 1, color: sel ? '#ffffff' : m.options.fillColor });
      if (sel) m.bringToFront();
    });
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------
  function setStatus(kind, text) {
    $('status-pill').dataset.state = kind;
    $('status-text').textContent = text;
  }

  function renderFreshness() {
    const f = state.feed;
    if (!f) return;
    const fetched = Date.parse(f.fetched_at);
    const cache = f.cache || {};
    const label = cache.stale ? 'Stale copy' : cache.hit ? 'Cached' : 'Fresh';
    $('freshness').textContent = `${label} · USGS data fetched ${C.formatAgo(fetched)}`;
    $('freshness').dataset.stale = cache.stale ? 'true' : 'false';
    $('source-detail').textContent = `Last fetch ${C.formatUtc(fetched)} (${label.toLowerCase()}). Window ${f.query.start.slice(0, 16).replace('T', ' ')} → ${f.query.end.slice(0, 16).replace('T', ' ')} UTC.`;
  }

  function renderMetrics() {
    const s = C.summarize(state.events);
    const f = state.feed;
    $('m-count').textContent = s.count.toLocaleString();
    let foot = `${state.days === 1 ? 'Past 24 h' : `Past ${state.days} days`} · M ${state.minMag}+`;
    if (state.depth !== 'all') foot += ` · ${state.depth}`;
    if (f && f.truncated) foot += ` · newest ${f.count.toLocaleString()} of ${(f.total_available ?? '?').toLocaleString()}`;
    $('m-count-foot').textContent = foot;
    $('m-strongest').textContent = s.strongest ? `M ${C.fmt(s.strongest.mag)}` : '—';
    $('m-strongest-foot').textContent = s.strongest ? (s.strongest.place || 'Location unavailable') : 'No magnitudes in view';
    $('m-depth').textContent = s.medianDepth === null ? '—' : `${C.fmt(s.medianDepth, 0)} km`;
    $('m-depth-foot').textContent = `${s.depthBins[0].count} shallow · ${s.depthBins[1].count} intermediate · ${s.depthBins[2].count} deep`;
    $('m-tsunami').textContent = s.tsunamiFlagged.toLocaleString();
    $('list-count').textContent = s.count.toLocaleString();
    renderBars($('mag-chart'), s.magBins, s.withMagnitude ? null : 'No magnitude values in view.');
    renderBars($('depth-chart'), s.depthBins, s.count ? null : 'No depth values in view.');
  }

  function renderBars(node, bins, emptyText) {
    if (emptyText) { node.innerHTML = `<p class="muted">${esc(emptyText)}</p>`; return; }
    const max = Math.max(1, ...bins.map((b) => b.count));
    node.innerHTML = bins.map((b) => `
      <div class="bar-row"><span class="bar-label">${esc(b.label)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(b.count / max) * 100}%"></span></span>
      <span class="bar-count">${b.count}</span></div>`).join('');
  }

  function renderList() {
    const list = $('event-list');
    if (!state.events.length) {
      list.innerHTML = `<div class="state">No earthquakes match the current filters.<br><span class="muted small">Try a longer window or lower magnitude. An empty result does not mean a region is safe.</span></div>`;
      return;
    }
    const frag = document.createDocumentFragment();
    [...state.events].sort((a, b) => b.time - a.time).slice(0, 200).forEach((e) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'event-row';
      row.dataset.id = e.id;
      row.setAttribute('aria-pressed', String(e.id === state.selectedId));
      row.innerHTML = `
        <span class="mag-badge" style="--c:${C.magColor(e.mag)}">${esc(C.fmt(e.mag))}</span>
        <span class="event-main"><span class="event-place">${esc(e.place || 'Location unavailable')}</span>
        <span class="event-meta">${esc(C.formatAgo(e.time))} · ${esc(C.formatUtc(e.time))}</span></span>
        <span class="event-depth">${e.depth === null ? '—' : `${esc(C.fmt(e.depth, Math.abs(e.depth) < 10 ? 1 : 0))} km`}</span>`;
      row.addEventListener('click', () => selectEvent(e.id, { pan: true }));
      frag.appendChild(row);
    });
    list.replaceChildren(frag);
    if (state.events.length > 200) {
      const more = document.createElement('p');
      more.className = 'muted small list-more';
      more.textContent = `Showing the newest 200 of ${state.events.length.toLocaleString()} events; all are on the map.`;
      list.appendChild(more);
    }
  }

  function renderDetail() {
    const e = state.events.find((x) => x.id === state.selectedId);
    const node = $('detail');
    if (!e) {
      node.innerHTML = '<p class="muted">Select an event on the map or in the list to see its details.</p>';
      return;
    }
    node.innerHTML = `
      <div class="detail-head">
        <span class="mag-badge big" style="--c:${C.magColor(e.mag)}">${esc(C.fmt(e.mag))}</span>
        <div><div class="kicker">Observed event${e.magType ? ` · <span class="as-is">${esc(e.magType)}</span>` : ''}</div>
        <h2>${esc(e.place || 'Location unavailable')}</h2></div>
        <button class="icon-btn" id="detail-close" aria-label="Clear selection">×</button>
      </div>
      <dl class="detail-grid">
        <dt>Time (UTC)</dt><dd>${esc(C.formatUtc(e.time))}</dd>
        <dt>Local time</dt><dd>${esc(new Date(e.time).toLocaleString())}</dd>
        <dt>Coordinates</dt><dd>${esc(C.formatCoord(e.latitude, e.longitude))}</dd>
        <dt>Depth</dt><dd>${e.depth === null ? 'Not reported' : `${esc(C.fmt(e.depth, 1))} km (${esc(C.depthClass(e.depth))})`}</dd>
        <dt>Review status</dt><dd>${esc(e.status || '—')}</dd>
        <dt>Tsunami flag</dt><dd>${e.tsunami ? 'Yes (see USGS)' : 'No'}</dd>
        ${e.alert ? `<dt>PAGER alert</dt><dd>${esc(e.alert)}</dd>` : ''}
      </dl>
      <div class="detail-actions">
        ${e.url ? `<a class="btn-small" href="${esc(e.url)}" target="_blank" rel="noopener noreferrer">USGS event page ↗</a>` : ''}
        <button class="btn-small" id="detail-score">Point analysis here</button>
      </div>`;
    $('detail-close').addEventListener('click', () => selectEvent(null));
    $('detail-score').addEventListener('click', () => openPointAnalysis(e.latitude, e.longitude, e.depth ?? 10));
  }

  function selectEvent(id, { pan = false } = {}) {
    state.selectedId = id;
    renderDetail();
    document.querySelectorAll('.event-row').forEach((r) => r.setAttribute('aria-pressed', String(r.dataset.id === id)));
    highlightSelected();
    const e = state.events.find((x) => x.id === id);
    if (e && pan && state.map) state.map.flyTo([e.latitude, e.longitude], Math.max(state.map.getZoom(), 5), { duration: 0.6 });
  }

  function renderModel(info) {
    const chip = $('model-chip');
    const body = $('model-body');
    if (!info || info.status !== 'loaded') {
      chip.textContent = 'Unavailable';
      chip.dataset.state = 'error';
      body.innerHTML = `<p class="muted">${esc(info?.load_error || 'Model metadata could not be retrieved.')}</p><p class="muted small">The event feed and map work without the model.</p>`;
      return;
    }
    chip.textContent = 'Loaded';
    chip.dataset.state = 'ok';
    const t = info.metrics?.test || {};
    const base = info.baselines?.['Constant train base rate'] || {};
    const lr = info.baselines?.['Logistic regression (same features)'] || {};
    body.innerHTML = `
      <p class="small">Estimates <strong>P(M ≥ ${esc(info.label.significant_mag)} | an M ≥ ${esc(info.label.catalog_min_mag)} event is recorded)</strong> from location, depth, recent local activity and plate-boundary distance.</p>
      <table class="mini-table"><thead><tr><th>Test period ${esc(info.split?.test?.from || '')} → ${esc(info.split?.test?.to || '')}</th><th>ROC-AUC</th><th>Brier</th></tr></thead>
      <tbody><tr><td>MLP (served)</td><td>${esc(C.fmt(t.roc_auc, 3))}</td><td>${esc(C.fmt(t.brier, 3))}</td></tr>
      <tr><td>Logistic regression</td><td>${esc(C.fmt(lr.roc_auc, 3))}</td><td>${esc(C.fmt(lr.brier, 3))}</td></tr>
      <tr><td>Constant base rate</td><td>${esc(C.fmt(base.roc_auc, 3))}</td><td>${esc(C.fmt(base.brier, 3))}</td></tr></tbody></table>
      <p class="muted small">Only marginally better than the base rate on unseen recent data. Treat scores as exploratory; this is not an earthquake forecast.</p>`;
  }

  // ---------------------------------------------------------------------------
  // Data loading
  // ---------------------------------------------------------------------------
  function applyDepthFilter() {
    state.events = C.filterByDepth(state.allEvents, state.depth);
    if (state.selectedId && !state.events.some((e) => e.id === state.selectedId)) state.selectedId = null;
    renderMetrics();
    renderList();
    renderDetail();
    renderMarkers();
  }

  async function loadFeed() {
    if (state.feedController) state.feedController.abort();   // cancel superseded request
    const controller = new AbortController();
    state.feedController = controller;
    setStatus('loading', 'Loading…');
    $('map-loading').hidden = false;
    const params = new URLSearchParams({ days: state.days, min_mag: state.minMag });
    try {
      const data = await fetchJson(`/api/earthquakes?${params}`, { signal: controller.signal });
      const { events, rejected, malformed } = C.normalizeEvents(data);
      if (malformed) throw new Error('The event feed response was malformed.');
      if (rejected) console.warn(`${rejected} events were rejected for missing coordinates or time.`);
      state.allEvents = events;
      state.feed = data;
      $('feed-banner').hidden = !data.cache?.stale;
      if (data.cache?.stale) $('feed-banner-text').textContent = data.cache.warning;
      setStatus(data.cache?.stale ? 'warn' : 'ok', data.cache?.stale ? 'Showing stale data' : 'Live');
      renderFreshness();
      applyDepthFilter();
    } catch (err) {
      if (err.superseded) return;
      setStatus('error', 'Feed unavailable');
      $('feed-banner').hidden = false;
      $('feed-banner-text').textContent = `Earthquake feed failed: ${err.message}${state.allEvents.length ? ' Previously loaded events are still shown.' : ''}`;
      if (!state.allEvents.length) {
        $('event-list').innerHTML = `<div class="state error">Could not load earthquakes.<br><span class="small">${esc(err.message)}</span></div>`;
        renderMetrics();
      }
    } finally {
      if (state.feedController === controller) {
        state.feedController = null;
        $('map-loading').hidden = true;
      }
    }
  }

  async function loadHotspots() {
    try {
      const data = await fetchJson('/api/hotspots');
      state.hotspots = Array.isArray(data.hotspots) ? data.hotspots : [];
    } catch (err) {
      state.hotspots = [];
      toast(`Activity clusters unavailable: ${err.message}`, 'error');
    }
    renderClusters();
  }

  async function loadModelInfo() {
    try { renderModel(await fetchJson('/api/model-info')); } catch (err) { renderModel(null); }
  }

  function refreshAll() {
    loadFeed();
    loadHotspots();
  }

  // ---------------------------------------------------------------------------
  // Point analysis
  // ---------------------------------------------------------------------------
  let lastFocus = null;

  function openPointAnalysis(lat, lon, depth = 10) {
    lastFocus = document.activeElement;
    $('pt-lat').value = lat.toFixed(4);
    $('pt-lon').value = lon.toFixed(4);
    $('pt-depth').value = Math.max(0, Math.min(700, depth)).toFixed(1);
    $('point-modal').hidden = false;
    $('pt-depth').focus();
    scorePoint();
  }

  function closePointAnalysis() {
    $('point-modal').hidden = true;
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  async function scorePoint() {
    const out = $('point-result');
    const params = new URLSearchParams({ lat: $('pt-lat').value, lon: $('pt-lon').value, depth: $('pt-depth').value });
    out.innerHTML = '<div class="state"><span class="spinner"></span>Scoring…</div>';
    try {
      const d = await fetchJson(`/api/predict?${params}`);
      const f = d.features;
      out.innerHTML = `
        <div class="score-head"><strong>${(d.probability * 100).toFixed(1)}%</strong>
        <span class="band" data-band="${esc(d.band)}">${esc(C.bandLabel(d.band))} · ${esc(C.fmt(d.relative_to_base_rate, 2))}× of ${(d.base_rate * 100).toFixed(1)}%</span></div>
        <p class="small">${esc(d.interpretation)}</p>
        <dl class="detail-grid">
          <dt>Coordinates</dt><dd>${esc(C.formatCoord(f.latitude, f.longitude))}</dd>
          <dt>Depth</dt><dd>${esc(C.fmt(f.depth_km, 1))} km</dd>
          <dt>M4.5+ events within ±1°, last 30 d</dt><dd>${esc(f.local_density_30d)}</dd>
          <dt>Distance to plate boundary</dt><dd>≈ ${esc(C.fmt(f.plate_distance_km, 0))} km</dd>
          <dt>Model test ROC-AUC</dt><dd>${esc(C.fmt(d.model?.test_roc_auc, 3))} (0.5 = chance)</dd>
        </dl>
        ${d.context?.stale ? '<p class="warn small">Recent-activity feature uses a stale USGS copy.</p>' : ''}`;
    } catch (err) {
      out.innerHTML = `<div class="state error">No score available.<br><span class="small">${esc(err.message)}</span></div>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------------
  function bind() {
    document.querySelectorAll('[data-days]').forEach((btn) => btn.addEventListener('click', () => {
      document.querySelectorAll('[data-days]').forEach((b) => b.setAttribute('aria-checked', String(b === btn)));
      state.days = Number(btn.dataset.days);
      loadFeed();
    }));
    $('min-mag').addEventListener('change', (e) => { state.minMag = Number(e.target.value); loadFeed(); });
    $('depth-filter').addEventListener('change', (e) => { state.depth = e.target.value; applyDepthFilter(); });
    document.querySelectorAll('[data-layer]').forEach((btn) => btn.addEventListener('click', () => {
      const name = btn.dataset.layer;
      state.layersOn[name] = !state.layersOn[name];
      btn.setAttribute('aria-pressed', String(state.layersOn[name]));
      applyLayerVisibility();
    }));
    $('refresh-btn').addEventListener('click', refreshAll);
    $('feed-retry').addEventListener('click', loadFeed);
    $('point-form').addEventListener('submit', (e) => { e.preventDefault(); scorePoint(); });
    document.querySelectorAll('[data-close]').forEach((el) => el.addEventListener('click', closePointAnalysis));
    document.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape') return;
      if (!$('point-modal').hidden) closePointAnalysis();
      else if ($('controls').classList.contains('open')) $('controls-toggle').click();
    });
    $('controls-toggle').addEventListener('click', () => {
      const open = $('controls').classList.toggle('open');
      $('controls-toggle').setAttribute('aria-expanded', String(open));
    });
    setInterval(() => { if (!document.hidden) refreshAll(); }, REFRESH_MS);
    setInterval(renderFreshness, 30000);
  }

  document.addEventListener('DOMContentLoaded', () => {
    initMap();
    bind();
    loadPlates();
    loadModelInfo();
    refreshAll();
  });
})();
