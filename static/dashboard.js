(() => {
  'use strict';
  const API = { earthquakes: '/api/earthquakes', hotspots: '/api/hotspots', modelInfo: '/api/model-info', predict: '/api/predict' };
  const state = { days: 1, minMag: 2.5, events: [], hotspots: [], modelInfo: null, map: null, eventLayer: null, hotspotLayer: null, selectedId: null, lastSuccess: null, requestToken: 0, showHotspots: true };
  const $ = (id) => document.getElementById(id);
  const fmt = (n, digits = 1) => Number.isFinite(Number(n)) ? Number(n).toFixed(digits) : 'N/A';
  const safeText = (value, fallback = 'Unknown') => value === null || value === undefined || value === '' ? fallback : String(value);
  const dateLabel = (raw) => { if (raw === null || raw === undefined) return 'Time unavailable'; const d = new Date(Number(raw) > 1e12 ? Number(raw) : (typeof raw === 'number' ? Number(raw) * 1000 : raw)); return Number.isNaN(d.getTime()) ? 'Time unavailable' : d.toLocaleString(undefined, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }); };
  const timeAgo = (d) => { if (!d) return 'Not yet refreshed'; const mins = Math.max(0, Math.floor((Date.now() - d.getTime()) / 60000)); return mins < 1 ? 'Just now' : mins < 60 ? `${mins} min ago` : mins < 1440 ? `${Math.floor(mins / 60)} hr ago` : `${Math.floor(mins / 1440)} d ago`; };
  const eventId = (feature, index) => feature?.id ?? feature?.properties?.code ?? feature?.properties?.id ?? `${feature?.geometry?.coordinates?.join(',') ?? 'event'}-${feature?.properties?.time ?? index}`;
  const eventCoords = (feature) => { const coords = feature?.geometry?.coordinates; if (Array.isArray(coords) && coords.length >= 2) return { lon: Number(coords[0]), lat: Number(coords[1]), depth: Number.isFinite(Number(coords[2])) ? Number(coords[2]) : null }; const p = feature?.properties ?? feature ?? {}; const lat = Number(p.latitude ?? p.lat); const lon = Number(p.longitude ?? p.lon ?? p.lng); return { lat, lon, depth: Number.isFinite(Number(p.depth)) ? Number(p.depth) : null }; };
  const eventProps = (feature) => feature?.properties ?? feature ?? {};
  const getFeatures = (payload) => { if (Array.isArray(payload)) return payload; if (Array.isArray(payload?.features)) return payload.features; if (Array.isArray(payload?.data?.features)) return payload.data.features; if (Array.isArray(payload?.earthquakes)) return payload.earthquakes; if (Array.isArray(payload?.data)) return payload.data; return []; };
  const getHotspots = (payload) => { if (Array.isArray(payload)) return payload; if (Array.isArray(payload?.features)) return payload.features; if (Array.isArray(payload?.hotspots)) return payload.hotspots; if (Array.isArray(payload?.data?.hotspots)) return payload.data.hotspots; if (Array.isArray(payload?.data)) return payload.data; return []; };
  const isValidCoord = (lat, lon) => Number.isFinite(lat) && Number.isFinite(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
  const magColor = (mag) => !Number.isFinite(mag) ? '#6e8c9b' : mag >= 5 ? '#ff646b' : mag >= 3 ? '#ffb35e' : '#54dce5';
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  async function fetchJson(url) {
    const response = await fetch(url, { headers: { Accept: 'application/json' }, cache: 'no-store' });
    let body;
    try { body = await response.json(); } catch (_) { throw new Error(`The endpoint returned invalid JSON (${response.status}).`); }
    if (!response.ok || body?.error) throw new Error(body?.error || `Request failed (${response.status}).`);
    return body;
  }

  function toast(message, kind = 'info') {
    const node = document.createElement('div'); node.className = `toast${kind === 'error' ? ' error' : ''}`; node.textContent = message; $('toast-region').appendChild(node); window.setTimeout(() => node.remove(), 4200);
  }

  function setConnection(kind, message) {
    const pill = $('connection-pill'); pill.classList.toggle('is-live', kind === 'live'); pill.classList.toggle('is-error', kind === 'error'); $('connection-text').textContent = message;
    $('source-dot').classList.toggle('is-live', kind === 'live'); $('source-dot').classList.toggle('is-error', kind === 'error');
    $('source-meta').textContent = kind === 'live' ? `Last response ${timeAgo(state.lastSuccess)}` : kind === 'error' ? 'Feed request failed' : 'Waiting for first response';
  }

  function initMap() {
    if (!window.L) { $('seismic-map').innerHTML = '<div class="empty-state"><strong>Map library unavailable</strong><span>Leaflet could not load. Check network access to the map library CDN.</span></div>'; return; }
    state.map = L.map('seismic-map', {
      zoomControl: false,
      worldCopyJump: false,
      preferCanvas: true,
      minZoom: 2,
      maxZoom: 16,
      maxBounds: [[-85, -180], [85, 180]],
      maxBoundsViscosity: 1.0
    }).setView([18, 4], 2);

    const urlParams = new URLSearchParams(window.location.search);
    const cartoKey = urlParams.get('carto_key') || window.CARTO_API_KEY;

    if (cartoKey) {
      L.tileLayer(`https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?api_key=${encodeURIComponent(cartoKey)}`, {
        subdomains: 'abcd',
        maxZoom: 19,
        updateWhenIdle: true,
        updateWhenZooming: false,
        keepBuffer: 2,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
      }).addTo(state.map);
    } else {
      // High-performance single-layer dark canvas: zero watermarks, no duplicate requests, buttery smooth
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        minZoom: 2,
        maxZoom: 16,
        updateWhenIdle: true,
        updateWhenZooming: false,
        keepBuffer: 2,
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
      }).addTo(state.map);
    }
    state.eventLayer = L.layerGroup().addTo(state.map); state.hotspotLayer = L.layerGroup().addTo(state.map);
    state.map.on('moveend zoomend', () => { const c = state.map.getCenter(); $('map-center').textContent = `${c.lat.toFixed(1)}° / ${c.lng.toFixed(1)}° · ${state.map.getZoom().toFixed(1)}×`; });
    state.map.on('click', (e) => requestPrediction(e.latlng.lat, e.latlng.lng));
    window.setTimeout(() => state.map.invalidateSize(), 180);
  }

  function renderMetrics() {
    const events = state.events; const mags = events.map(f => Number(eventProps(f).mag)).filter(Number.isFinite); const max = mags.length ? Math.max(...mags) : null;
    $('metric-events').textContent = events.length.toLocaleString(); $('metric-events-foot').textContent = `${state.days === 1 ? 'Past 24 hours' : `Past ${state.days} days`} · M ${fmt(state.minMag, 1)}+`;
    $('metric-strongest').innerHTML = max === null ? '— <small>MAG</small>' : `${fmt(max)} <small>MAG</small>`;
    const strongest = max === null ? null : events.find(f => Number(eventProps(f).mag) === max); $('metric-strongest-foot').textContent = strongest ? safeText(eventProps(strongest).place, 'Location unavailable') : 'No magnitude values returned';
    $('metric-hotspots').textContent = state.hotspots.length.toLocaleString(); $('metric-hotspots-foot').textContent = state.hotspots.length ? 'Returned by /api/hotspots' : 'No hotspots returned';
    $('metric-recency').textContent = timeAgo(state.lastSuccess); $('metric-updated').textContent = state.lastSuccess ? state.lastSuccess.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'Not yet refreshed';
    $('recency-track-fill').style.width = state.lastSuccess ? `${Math.max(15, 100 - Math.min(85, (Date.now() - state.lastSuccess.getTime()) / 6000))}%` : '0%';
    $('event-count-chip').textContent = events.length.toLocaleString(); $('events-footer-text').textContent = `${events.length.toLocaleString()} records in current response`;
  }

  function renderEvents() {
    const list = $('events-list'); list.replaceChildren();
    if (!state.events.length) { list.innerHTML = '<div class="empty-state"><strong>No earthquake records returned</strong><span>Try a longer observation window or a lower minimum magnitude. Empty results are not an indication that the region is safe.</span></div>'; return; }
    const sorted = [...state.events].sort((a, b) => Number(eventProps(b).time ?? 0) - Number(eventProps(a).time ?? 0)).slice(0, 70);
    sorted.forEach((feature, index) => {
      const p = eventProps(feature); const { lat, lon, depth } = eventCoords(feature); const mag = Number(p.mag); const hasMag = Number.isFinite(mag); const id = String(eventId(feature, index));
      const row = document.createElement('button'); row.type = 'button'; row.className = `event-row${state.selectedId === id ? ' selected' : ''}`; row.dataset.eventId = id;
      const badgeClass = hasMag && mag >= 5 ? 'mag-high' : hasMag && mag >= 3 ? 'mag-mid' : '';
      const place = safeText(p.place ?? p.title ?? p.location, 'Location unavailable');
      row.innerHTML = `<span class="magnitude-badge ${badgeClass}">${hasMag ? escapeHtml(mag.toFixed(1)) : 'N/A'}</span><span><span class="event-location" title="${escapeHtml(place)}">${escapeHtml(place)}</span><span class="event-time">${escapeHtml(dateLabel(p.time))}</span></span><span class="event-depth">${depth === null ? '—' : `${escapeHtml(fmt(depth, 0))} km`}</span>`;
      row.addEventListener('click', () => focusEvent(feature, id)); list.appendChild(row);
    });
  }

  function renderMagnitudeChart() {
    const groups = [{ label: '< 2.0', min: -Infinity, max: 2 }, { label: '2.0–2.9', min: 2, max: 3 }, { label: '3.0–3.9', min: 3, max: 4 }, { label: '4.0–4.9', min: 4, max: 5 }, { label: '5.0–5.9', min: 5, max: 6 }, { label: '6.0+', min: 6, max: Infinity }];
    const mags = state.events.map(f => Number(eventProps(f).mag)).filter(Number.isFinite); const counts = groups.map(g => mags.filter(m => m >= g.min && m < g.max).length); const max = Math.max(1, ...counts); const node = $('magnitude-chart'); node.replaceChildren();
    if (!mags.length) { node.innerHTML = '<div class="chart-placeholder">No magnitude values are available to chart.</div>'; return; }
    groups.forEach((g, i) => { const row = document.createElement('div'); row.className = 'mag-bar-row'; row.innerHTML = `<span class="mag-bar-label">${g.label}</span><span class="mag-bar-track"><span class="mag-bar-fill" style="display:block;width:${counts[i] / max * 100}%"></span></span><span class="mag-bar-count">${counts[i]}</span>`; node.appendChild(row); });
  }

  function renderModelInfo() {
    const info = state.modelInfo; const status = info ? (info.status ?? info.model_status ?? info.loaded ?? info.model_loaded ?? info.is_loaded ?? info.available) : null;
    const ready = status === true || String(status).toLowerCase() === 'loaded' || String(status).toLowerCase() === 'ready' || String(status).toLowerCase() === 'true' || String(status).toLowerCase() === 'available';
    const chip = $('model-status-chip'); chip.classList.toggle('is-ready', ready); chip.innerHTML = `<i></i> ${info ? (ready ? 'AVAILABLE' : 'RESPONSE RECEIVED') : 'UNAVAILABLE'}`;
    $('model-detail').textContent = info ? safeText(info.model_name ?? info.name ?? info.status ?? info.model_status ?? (ready ? 'Available' : 'Metadata returned'), 'Metadata returned') : 'Endpoint unavailable';
    $('insight-hotspot-value').innerHTML = `${state.hotspots.length.toLocaleString()} <span>locations</span>`;
    $('insight-description').textContent = state.hotspots.length ? 'Hotspots are exploratory model output, not a guarantee of future earthquakes.' : 'No hotspots were returned. This can indicate unavailable model output, not a lack of seismic risk.';
    $('last-success').textContent = state.lastSuccess ? state.lastSuccess.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—';
  }

  function renderMap() {
    if (!state.map || !state.eventLayer || !state.hotspotLayer) return;
    state.eventLayer.clearLayers(); state.hotspotLayer.clearLayers();
    state.events.forEach((feature, index) => {
      const p = eventProps(feature); const { lat, lon, depth } = eventCoords(feature); if (!isValidCoord(lat, lon)) return;
      const mag = Number(p.mag); const hasMag = Number.isFinite(mag); const color = magColor(hasMag ? mag : NaN); const radius = hasMag ? Math.max(4, Math.min(13, 3.7 + mag * 1.45)) : 4.5; const id = String(eventId(feature, index));
      const marker = L.circleMarker([lat, lon], { radius, color, weight: 1.4, opacity: .95, fillColor: color, fillOpacity: hasMag ? .72 : .45, bubblingMouseEvents: false });
      const place = safeText(p.place ?? p.title ?? p.location, 'Location unavailable'); const time = dateLabel(p.time); const magText = hasMag ? mag.toFixed(1) : 'N/A';
      marker.bindPopup(`<div class="popup-title">${escapeHtml(place)}</div><div class="popup-mag">M ${escapeHtml(magText)}</div><div class="popup-meta">Depth: ${depth === null ? 'N/A' : `${escapeHtml(fmt(depth, 1))} km`}<br>${escapeHtml(time)}</div>`);
      marker.on('click', () => { state.selectedId = id; renderEvents(); }); marker.addTo(state.eventLayer);
    });
    state.hotspots.forEach((hotspot, index) => {
      const p = hotspot?.properties ?? hotspot ?? {}; let lat, lon;
      if (hotspot?.geometry?.coordinates) { lon = Number(hotspot.geometry.coordinates[0]); lat = Number(hotspot.geometry.coordinates[1]); } else { lat = Number(p.lat ?? p.latitude); lon = Number(p.lon ?? p.lng ?? p.longitude); }
      if (!isValidCoord(lat, lon)) return;
      const radius = Math.max(8, Math.min(18, 10 + Number(p.probability ?? p.score ?? 0) * 5));
      L.circleMarker([lat, lon], { radius, color: '#ff9c60', weight: 1.5, fillColor: '#ff744f', fillOpacity: .12, opacity: .85 }).bindTooltip(`Model hotspot ${index + 1}<br>${lat.toFixed(2)}, ${lon.toFixed(2)}`).addTo(state.hotspotLayer);
    });
    if (!state.showHotspots) state.map.removeLayer(state.hotspotLayer); else if (!state.map.hasLayer(state.hotspotLayer)) state.hotspotLayer.addTo(state.map);
  }

  function focusEvent(feature, id) {
    state.selectedId = id; renderEvents(); const { lat, lon } = eventCoords(feature);
    if (state.map && isValidCoord(lat, lon)) { state.map.setView([lat, lon], Math.max(state.map.getZoom(), 5), { animate: true }); state.eventLayer.eachLayer(layer => { if (layer.getLatLng && Math.abs(layer.getLatLng().lat - lat) < 1e-5 && Math.abs(layer.getLatLng().lng - lon) < 1e-5) layer.openPopup(); }); }
  }

  async function requestPrediction(lat, lon) {
    $('prediction-modal').classList.add('is-open'); $('prediction-modal').setAttribute('aria-hidden', 'false'); $('prediction-coordinates').textContent = `${lat.toFixed(4)}°, ${lon.toFixed(4)}°`; $('prediction-content').innerHTML = '<span class="spinner"></span><span>Requesting model estimate…</span>';
    try {
      const data = await fetchJson(`${API.predict}?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&depth=10`);
      const risk = safeText(data.risk ?? data.prediction ?? data.label ?? data.status, 'Estimate returned'); const prob = data.probability ?? data.probability_score ?? data.score;
      const detail = Object.entries(data).filter(([k, v]) => !['risk', 'prediction', 'label', 'status'].includes(k) && (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean')).slice(0, 6).map(([k, v]) => `<div class="insight-detail-row"><span>${escapeHtml(k.replace(/_/g, ' '))}</span><strong>${escapeHtml(typeof v === 'number' ? fmt(v, 3) : String(v))}</strong></div>`).join('');
      $('prediction-content').innerHTML = `<div class="prediction-result"><strong>${escapeHtml(risk)}</strong>${prob !== undefined ? `<p>Reported score: ${escapeHtml(fmt(Number(prob), 3))}</p>` : ''}<p>Result returned by the existing model API for this coordinate. It is not a time-bound earthquake forecast or an official alert.</p>${detail}</div>`;
    } catch (err) { $('prediction-content').innerHTML = `<div class="prediction-result"><strong>Estimate unavailable</strong><p>${escapeHtml(err.message || 'The prediction endpoint could not be reached.')}</p><p>The map and observed earthquake feed remain available independently of this model request.</p></div>`; }
  }

  async function refreshData() {
    const token = ++state.requestToken; $('events-list').innerHTML = '<div class="loading-state"><span class="spinner"></span><span>Refreshing seismic records…</span></div>'; setConnection('connecting', 'CONNECTING');
    const query = new URLSearchParams({ days: String(state.days), min_mag: String(state.minMag) });
    const results = await Promise.allSettled([fetchJson(`${API.earthquakes}?${query.toString()}`), fetchJson(API.hotspots), fetchJson(API.modelInfo)]);
    if (token !== state.requestToken) return;
    const quakeResult = results[0];
    if (quakeResult.status === 'fulfilled') {
      state.events = getFeatures(quakeResult.value); state.lastSuccess = new Date(); setConnection('live', 'DATA CONNECTED');
      if (state.events.length === 0 && (quakeResult.value?.error || quakeResult.value?.status === 'error')) setConnection('error', 'FEED ERROR');
    } else {
      state.events = []; setConnection('error', 'FEED ERROR'); toast(`Earthquake feed: ${quakeResult.reason?.message || 'request failed'}`, 'error');
    }
    if (results[1].status === 'fulfilled') state.hotspots = getHotspots(results[1].value); else { state.hotspots = []; console.warn('Hotspot endpoint unavailable:', results[1].reason); }
    if (results[2].status === 'fulfilled') state.modelInfo = results[2].value; else { state.modelInfo = null; console.warn('Model metadata endpoint unavailable:', results[2].reason); }
    renderMetrics(); renderEvents(); renderMagnitudeChart(); renderMap(); renderModelInfo();
    if (results[1].status === 'rejected' && results[2].status === 'rejected') $('model-detail').textContent = 'Model endpoints unavailable';
  }

  function updateClock() { const now = new Date(); $('clock').textContent = now.toLocaleTimeString([], { hour12: false }); $('date-label').textContent = now.toLocaleDateString([], { year: 'numeric', month: 'short', day: '2-digit' }); if (state.lastSuccess) { $('metric-recency').textContent = timeAgo(state.lastSuccess); $('source-meta').textContent = `Last response ${timeAgo(state.lastSuccess)}`; } }
  function closePrediction() { $('prediction-modal').classList.remove('is-open'); $('prediction-modal').setAttribute('aria-hidden', 'true'); }

  function bindEvents() {
    document.querySelectorAll('.time-button').forEach(btn => btn.addEventListener('click', () => { document.querySelectorAll('.time-button').forEach(b => b.classList.toggle('active', b === btn)); state.days = Number(btn.dataset.days) || 1; refreshData(); }));
    $('min-magnitude').addEventListener('change', (e) => { state.minMag = Number(e.target.value) || 0; refreshData(); });
    $('refresh-button').addEventListener('click', refreshData); $('load-more').addEventListener('click', refreshData);
    $('toggle-hotspots').addEventListener('click', () => { state.showHotspots = !state.showHotspots; $('toggle-hotspots').setAttribute('aria-pressed', String(state.showHotspots)); $('hotspot-state').textContent = state.showHotspots ? 'ON' : 'OFF'; if (state.map && state.hotspotLayer) { if (state.showHotspots) state.hotspotLayer.addTo(state.map); else state.map.removeLayer(state.hotspotLayer); } });
    $('clear-selection').addEventListener('click', () => { state.selectedId = null; renderEvents(); });
    document.querySelectorAll('[data-close-prediction]').forEach(el => el.addEventListener('click', closePrediction)); document.addEventListener('keydown', e => { if (e.key === 'Escape') closePrediction(); });
    $('mobile-menu').addEventListener('click', () => $('sidebar').classList.toggle('is-open'));
    document.querySelectorAll('.main-nav a').forEach(a => a.addEventListener('click', () => $('sidebar').classList.remove('is-open')));
  }

  document.addEventListener('DOMContentLoaded', () => { initMap(); bindEvents(); updateClock(); window.setInterval(updateClock, 1000); refreshData(); window.setInterval(refreshData, 180000); });
})();
