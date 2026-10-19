/*
 * TerraWatcher analytical views: Sequence Explorer, Activity Forecasts, Anomaly
 * Detection, Aftershock Analysis and Model Performance.
 *
 * Every number shown comes from an API response (validated by analytics.js);
 * when a request fails or a module is unavailable the view says so and offers a
 * retry. Pure helpers live in analytics.js and charts.js.
 */
(() => {
  'use strict';
  const C = window.TerraCore;
  const CH = window.TerraCharts;
  const ML = window.TerraML;
  const A = window.TerraApp;
  const $ = (id) => document.getElementById(id);
  const esc = C.escapeHtml;
  const DAY = 86400000;

  const VIEWS = {
    live: { title: 'Live Monitor' },
    sequences: { title: 'Sequence Explorer', layer: 'Detected sequences', hint: 'Click a coloured event to select its sequence',
      subtitle: 'Space-time DBSCAN groups of the events in the current feed window. A detected sequence is a statistical grouping, not proof that its events are related.' },
    forecast: { title: 'Activity Forecasts', layer: 'Expected counts', hint: 'Click a cell to see its forecast',
      subtitle: 'Expected number of qualifying earthquakes per 10° cell over the next 7 days. An aggregate statistical forecast — not a prediction of any individual earthquake.' },
    anomaly: { title: 'Anomaly Detection', layer: 'Flagged cells', hint: 'Click a cell to inspect its activity',
      subtitle: 'Cells whose recent observed count is unusually high relative to their own reference period. Elevated activity is not a prediction of a large earthquake.' },
    aftershock: { title: 'Aftershock Analysis', layer: 'Selected sequence', hint: 'Events after the reference event, coloured by time',
      subtitle: 'Event-centred statistics and, only when the data support it, a modified Omori–Utsu decay fit.' },
    performance: { title: 'Model Performance' },
  };
  const ORDER = Object.keys(VIEWS);
  const tracker = ML.createTracker();
  const st = { view: 'live', loaded: {}, seq: null, seqSel: null, fc: null, fcSel: null, fcDetail: null,
    an: null, anSel: null, anDetail: null, as: null, compare: [], perf: null };
  let layer = null;

  // ---------------------------------------------------------------------------
  // Shared helpers
  // ---------------------------------------------------------------------------
  function stateBox(kind, text, retry) {
    const spinner = kind === 'loading' ? '<span class="spinner"></span>' : '';
    const btn = retry ? `<button class="btn-small" data-retry="${esc(retry)}">Retry</button>` : '';
    return `<div class="state ${kind === 'error' ? 'error' : ''}">${spinner}<span>${esc(text)}</span>${btn}</div>`;
  }

  function setViewStatus(text, kind = 'info') {
    const node = $('view-status');
    node.textContent = text || '';
    node.dataset.kind = kind;
  }

  async function request(kind, url) {
    const token = tracker.start(kind);
    try {
      const data = await A.fetchJson(url);
      return tracker.isCurrent(kind, token) ? data : null;     // a newer request of this kind superseded it
    } catch (err) {
      if (!tracker.isCurrent(kind, token)) return null;
      throw err;
    }
  }

  function dl(rows) {
    return `<dl class="detail-grid">${rows.filter(Boolean).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl>`;
  }

  function table(head, rows, cls = 'mini-table') {
    return `<div class="table-wrap"><table class="${cls}"><thead><tr>${head.map((h) => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${r.map((c, i) => (i === 0 ? `<th scope="row">${c}</th>` : `<td>${c}</td>`)).join('')}</tr>`).join('')}</tbody></table></div>`;
  }

  const f = (v, d = 3) => esc(C.fmt(v, d));
  const isoDay = (s) => (typeof s === 'string' ? s.slice(0, 10) : '—');
  const isoTime = (s) => (typeof s === 'string' ? `${s.slice(0, 16).replace('T', ' ')} UTC` : '—');
  const card = (title, tag, body, cls = '') => `<article class="panel chart-card ${cls}"><header><h2>${esc(title)}</h2>${tag ? `<span class="tag">${esc(tag)}</span>` : ''}</header>${body}</article>`;

  function fitMap(latlngs) {
    if (!A.state.map || !latlngs.length) return;
    A.state.map.fitBounds(L.latLngBounds(latlngs).pad(0.3), { maxZoom: 7 });
  }

  const WORLD = [[-58, -170], [72, 190]];
  function worldView() { if (A.state.map) A.state.map.fitBounds(WORLD); }

  function regionAt(list, lat, lon, key = 'region') {
    return (list || []).find((r) => lat >= r.bounds.lat_min && lat < r.bounds.lat_max && lon >= r.bounds.lon_min && lon < r.bounds.lon_max)?.[key] || null;
  }

  // ---------------------------------------------------------------------------
  // View switching (ARIA tabs)
  // ---------------------------------------------------------------------------
  function setView(view, { focus = false } = {}) {
    if (!VIEWS[view]) view = 'live';
    st.view = view;
    A.state.view = view;
    document.querySelectorAll('[role="tab"][data-view]').forEach((t) => {
      const on = t.dataset.view === view;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
    const ws = $('workspace');
    ws.dataset.activeView = view;
    ws.setAttribute('aria-labelledby', `tab-${view}`);
    document.querySelectorAll('[data-pane]').forEach((el) => { el.hidden = !el.dataset.pane.split(' ').includes(view); });
    const v = VIEWS[view];
    $('view-title').textContent = v.title;
    $('view-subtitle').textContent = v.subtitle || '';
    $('module-layer-name').textContent = v.layer || 'Module layer';
    $('map-hint-text').textContent = v.hint || 'Click the map for a point analysis';
    setViewStatus('');
    if (layer) layer.clearLayers();
    A.applyLayerVisibility();
    if (A.state.map) setTimeout(() => A.state.map.invalidateSize(), 0);
    if (history.replaceState) history.replaceState(null, '', view === 'live' ? location.pathname : `#${view}`);
    redraw();
    if (!st.loaded[view]) load(view);
  }

  function load(view) {
    st.loaded[view] = true;
    ({ sequences: loadSequences, forecast: loadForecast, anomaly: loadAnomalies, performance: loadPerformance,
      aftershock: renderAftershockIdle }[view] || (() => {}))();
  }

  function redraw() {
    if (st.view === 'sequences' && st.seq) drawSequenceLayer();
    if (st.view === 'forecast' && st.fc) drawForecastLayer();
    if (st.view === 'anomaly' && st.an) drawAnomalyLayer();
    if (st.view === 'aftershock' && st.as) drawAftershockLayer();
  }

  // ===========================================================================
  // Sequence Explorer
  // ===========================================================================
  async function loadSequences() {
    const form = $('seq-form');
    if (!form.checkValidity()) { form.reportValidity(); return; }
    const { days, minMag } = A.feedParams();
    const params = new URLSearchParams({ days, min_mag: minMag, eps_km: $('seq-eps-km').value,
      eps_hours: $('seq-eps-hours').value, min_samples: $('seq-min-samples').value });
    $('seq-panel').innerHTML = stateBox('loading', 'Detecting sequences…');
    if (!st.seq) $('seq-inspector').innerHTML = stateBox('loading', 'Detecting sequences…');
    setViewStatus('Running DBSCAN…');
    try {
      const data = await request('seq', `/api/sequences?${params}`);
      if (!data) return;
      const v = ML.validateSequences(data);
      if (!v.ok) throw new Error(v.error);
      st.seq = data;
      if (!data.sequences.some((s) => s.id === st.seqSel)) st.seqSel = null;
      setViewStatus(`${data.summary.sequence_count} sequences · ${data.summary.noise_events} unassigned events${data.cache?.stale ? ' · stale data' : ''}`,
        data.cache?.stale ? 'warn' : 'info');
      renderSequences();
    } catch (err) {
      setViewStatus('Sequence detection failed', 'error');
      $('seq-panel').innerHTML = stateBox('error', `Sequences unavailable: ${err.message}`, 'seq');
      $('seq-inspector').innerHTML = stateBox('error', 'No sequence results.', 'seq');
      if (layer && st.view === 'sequences') layer.clearLayers();
      st.seq = null;
    }
  }

  function renderSequences() {
    if (st.view === 'sequences') drawSequenceLayer();
    renderSequenceInspector();
    renderSequencePanel();
  }

  function drawSequenceLayer() {
    if (!layer) return;
    layer.clearLayers();
    const spec = ML.sequenceLayerSpec(st.seq.events, st.seq.sequences, st.seqSel);
    spec.sort((a, b) => (a.sequence ? 1 : 0) - (b.sequence ? 1 : 0) || (a.highlighted ? 1 : 0) - (b.highlighted ? 1 : 0));
    spec.forEach((s) => {
      const m = L.circleMarker([s.latitude, s.longitude], {
        radius: s.radius, color: s.highlighted ? '#ffffff' : s.color, weight: s.highlighted ? 1.5 : 0.5,
        fillColor: s.color, fillOpacity: s.opacity, opacity: Math.min(1, s.opacity + 0.1), bubblingMouseEvents: false,
      });
      m.bindTooltip(s.sequence ? `Sequence ${esc(s.sequence)}` : 'Unassigned (noise)', { direction: 'top' });
      if (s.sequence) m.on('click', () => selectSequence(s.sequence));
      m.addTo(layer);
    });
  }

  function selectSequence(id, { zoom = false } = {}) {
    st.seqSel = id;
    renderSequences();
    if (zoom) {
      const seq = st.seq.sequences.find((s) => s.id === id);
      const ids = new Set(seq.member_ids);
      fitMap(st.seq.events.filter((e) => ids.has(e.id)).map((e) => [e.latitude, e.longitude]));
    }
  }

  function renderSequenceInspector() {
    const d = st.seq, s = d.summary;
    const head = `<section class="detail"><div class="kicker">Detection summary · ${esc(d.query.days)} d · M${esc(d.query.min_mag)}+</div>
      ${dl([['Events considered', s.events_considered], ['Detected sequences', s.sequence_count], ['Events in sequences', s.clustered_events],
        ['Unassigned (noise)', s.noise_events], ['Ambiguous border events', s.ambiguous_border_events],
        ['Invalid / duplicate removed', `${s.dropped_invalid} / ${s.duplicates_removed}`]])}
      ${d.truncated ? `<p class="warn small">The feed returned the newest ${esc(d.query.limit)} of ${esc(d.total_available ?? '?')} events; older events are not clustered.</p>` : ''}</section>`;
    const sel = d.sequences.find((x) => x.id === st.seqSel);
    const detail = sel ? sequenceDetail(sel) : '';
    const list = d.sequences.length ? d.sequences.slice(0, 60).map((x, i) => `
      <button type="button" class="event-row seq-row" data-seq="${esc(x.id)}" aria-pressed="${x.id === st.seqSel}">
        <span class="mag-badge" style="--c:${ML.sequenceColor(i)}">${x.event_count}</span>
        <span class="event-main"><span class="event-place">${esc(x.id)} · max M ${f(x.max_mag, 1)}</span>
        <span class="event-meta">${esc(C.formatUtc(x.start))} · ${esc(ML.formatDuration(x.duration_hours))}</span></span>
        <span class="event-depth">${esc(x.event_count)} ev</span></button>`).join('')
      : '<div class="state">No sequences with these parameters. Try a larger ε or fewer events per core.</div>';
    $('seq-inspector').innerHTML = `${head}${detail}<header class="list-head"><h2>Sequences <span class="count">${d.sequences.length}</span></h2><span class="muted small">Largest first</span></header><div class="event-list">${list}</div>`;
    $('seq-inspector').querySelectorAll('[data-seq]').forEach((b) => b.addEventListener('click', () => selectSequence(b.dataset.seq, { zoom: true })));
    if (sel) {
      $('seq-clear').addEventListener('click', () => selectSequence(null));
      $('seq-zoom').addEventListener('click', () => selectSequence(sel.id, { zoom: true }));
      const as = $('seq-aftershock');
      if (as) as.addEventListener('click', () => openAftershock(sel.largest_event_id, {
        radius: Math.min(1000, Math.max(30, Math.ceil(sel.extent.max_distance_from_centroid_km + d.parameters.eps_km))) }));
      $('seq-inspector').querySelectorAll('[data-member]').forEach((b) => b.addEventListener('click', () => {
        const e = d.events.find((x) => x.id === b.dataset.member);
        if (e && A.state.map) A.state.map.flyTo([e.latitude, e.longitude], Math.max(A.state.map.getZoom(), 6), { duration: 0.5 });
      }));
    }
  }

  function sequenceDetail(s) {
    const ev = new Map(st.seq.events.map((e) => [e.id, e]));
    const members = s.member_ids.slice(0, 100).map((id) => ev.get(id)).filter(Boolean);
    const ext = s.extent;
    return `<section class="detail">
      <div class="detail-head"><div><div class="kicker">Detected sequence</div><h2>${esc(s.id)}</h2></div>
      <button class="icon-btn" id="seq-clear" aria-label="Clear sequence selection">×</button></div>
      ${dl([['Events', `${s.event_count} (${s.core_events} core)`], ['Start', esc(C.formatUtc(s.start))], ['End', esc(C.formatUtc(s.end))],
        ['Duration', esc(ML.formatDuration(s.duration_hours))], ['Largest', s.max_mag === null ? 'No magnitudes' : `M ${f(s.max_mag, 1)}`],
        ['Magnitude median (IQR)', `${f(s.magnitude.median, 2)} (${f(s.magnitude.q25, 2)}–${f(s.magnitude.q75, 2)})`],
        ['Depth median', s.depth_km.n ? `${f(s.depth_km.median, 0)} km (${f(s.depth_km.min, 0)}–${f(s.depth_km.max, 0)})` : 'Not reported'],
        ['Centroid', esc(C.formatCoord(s.centroid.latitude, s.centroid.longitude))],
        ['Extent', `${f(ext.lat_min, 2)}…${f(ext.lat_max, 2)}°, ${f(ext.lon_west, 2)}…${f(ext.lon_east, 2)}°${ext.crosses_antimeridian ? ' (crosses 180°)' : ''}`],
        ['Max distance from centroid', `${f(ext.max_distance_from_centroid_km, 0)} km`],
        ['Median nearest member', `${f(s.linking.median_nearest_member_km, 1)} km`], ['Median gap between events', esc(ML.formatDuration(s.linking.median_gap_hours))],
        s.ambiguous_border_events ? ['Ambiguous border events', s.ambiguous_border_events] : null])}
      <p class="small muted">${esc(st.seq.explanation)}</p>
      <div class="detail-actions"><button class="btn-small" id="seq-zoom">Zoom to sequence</button>
      ${s.largest_event_id ? '<button class="btn-small" id="seq-aftershock">Analyse activity after its largest event</button>' : ''}</div>
      <h3 class="sub">Member events${s.event_count > 100 ? ' (first 100)' : ''}</h3>
      <div class="member-list">${members.map((e) => `<button type="button" class="member" data-member="${esc(e.id)}">
        <span>M ${f(e.mag, 1)}</span><span>${esc(C.formatUtc(e.time))}</span><span class="muted">${esc(e.place || e.id)}</span></button>`).join('')}</div>
    </section>`;
  }

  function renderSequencePanel() {
    const d = st.seq;
    const sel = d.sequences.find((x) => x.id === st.seqSel);
    if (!sel) {
      const top = d.sequences.slice(0, 20).map((s) => ({ label: s.id.slice(2, 7), count: s.event_count }));
      $('seq-panel').innerHTML = `<div class="grid-2">
        ${card('Sequence sizes', 'Largest 20', CH.barChart({ bins: top, xLabel: 'Sequence (id prefix)', yLabel: 'Events', color: '#4fd1c5', title: 'Number of events per detected sequence', empty: 'No sequences detected.' }))}
        ${card('Method and limitations', 'DBSCAN', `<p class="small">${esc(d.explanation)}</p><ul class="small muted tight">${d.limitations.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
          <p class="small muted">Parameters: ε = ${esc(d.parameters.eps_km)} km, ${esc(d.parameters.eps_hours)} h, min. ${esc(d.parameters.min_samples)} events per core. Window ${esc(isoTime(d.query.start))} → ${esc(isoTime(d.query.end))}.</p>`)}
      </div>`;
      return;
    }
    const ev = new Map(d.events.map((e) => [e.id, e]));
    const members = sel.member_ids.map((id) => ev.get(id)).filter(Boolean).sort((a, b) => a.time - b.time);
    const scatter = members.filter((e) => Number.isFinite(e.mag)).map((e) => ({ x: e.time, y: e.mag, title: `M ${e.mag} · ${C.formatUtc(e.time)}` }));
    const cum = members.map((e, i) => ({ x: e.time, y: i + 1 }));
    $('seq-panel').innerHTML = `<div class="grid-2">
      ${card('Magnitude over time', sel.id, CH.lineChart({ title: 'Magnitude of member events over time', x: { type: 'time', label: 'Time (UTC)' }, y: { label: 'Magnitude', min: Math.floor(Math.min(...scatter.map((p) => p.y), 9)) - 0.5 }, series: [{ name: 'Member event', kind: 'dot', color: '#ffb454', data: scatter }], empty: 'No magnitudes reported.' }))}
      ${card('Cumulative events', sel.id, CH.lineChart({ title: 'Cumulative number of member events', x: { type: 'time', label: 'Time (UTC)' }, y: { label: 'Events' }, series: [{ name: 'Cumulative count', kind: 'step', color: '#4fd1c5', data: cum }] }))}
      ${card('Magnitude distribution', 'Members', CH.barChart({ bins: CH.histogramBins(sel.magnitude.histogram, 0), xLabel: 'Magnitude', yLabel: 'Events', color: '#ffb454', title: 'Magnitude histogram of the sequence' }))}
      ${card('Depth distribution', 'Members', CH.barChart({ bins: CH.histogramBins(sel.depth_km.histogram, 0), xLabel: 'Depth (km)', yLabel: 'Events', color: '#6b8cff', title: 'Depth histogram of the sequence', empty: 'No depths reported.' }))}
    </div>`;
  }

  // ===========================================================================
  // Activity Forecasts
  // ===========================================================================
  async function loadForecast() {
    const thr = $('fc-threshold').value;
    $('fc-panel').innerHTML = stateBox('loading', 'Computing forecasts from the latest USGS history (first load can take ~10 s)…');
    $('fc-inspector').innerHTML = stateBox('loading', 'Loading regional forecasts…');
    setViewStatus('Loading…');
    try {
      const data = await request('fc', `/api/forecast?threshold=${encodeURIComponent(thr)}`);
      if (!data) return;
      const v = ML.validateForecast(data);
      if (!v.ok) throw new Error(v.error);
      if (!st.fc) worldView();
      st.fc = data;
      if (!data.regions.some((r) => r.region === st.fcSel)) { st.fcSel = null; st.fcDetail = null; }
      const dw = data.data_window;
      setViewStatus(`Origin ${isoTime(data.origin)} · ${data.regions.length} cells${dw.stale ? ' · stale history' : ''}`, dw.stale ? 'warn' : 'info');
      if (st.view === 'forecast') drawForecastLayer();
      renderForecastInspector();
      if (st.fcSel) loadForecastDetail(st.fcSel); else renderForecastPanel();
    } catch (err) {
      st.fc = null;
      if (layer && st.view === 'forecast') layer.clearLayers();
      setViewStatus('Forecast unavailable', 'error');
      const msg = err.status === 503 ? `Forecast unavailable: ${err.message}` : `Forecast request failed: ${err.message}`;
      $('fc-panel').innerHTML = stateBox('error', `${msg} Live monitoring is unaffected.`, 'fc');
      $('fc-inspector').innerHTML = stateBox('error', 'No forecast was computed.', 'fc');
    }
  }

  function drawForecastLayer() {
    if (!layer) return;
    layer.clearLayers();
    st.fc.regions.forEach((r) => {
      const sel = r.region === st.fcSel;
      const rect = L.rectangle(ML.boundsToLatLngs(r.bounds), {
        color: sel ? '#ffffff' : ML.forecastColor(r.expected), weight: sel ? 2 : 0.6, fillColor: ML.forecastColor(r.expected),
        fillOpacity: sel ? 0.6 : 0.42, bubblingMouseEvents: false,
      });
      rect.bindTooltip(`${esc(r.label)}: ${C.fmt(r.expected, 2)} expected (80%: ${r.interval_80[0]}–${r.interval_80[1]})`, { sticky: true });
      rect.on('click', () => selectForecastRegion(r.region));
      rect.addTo(layer);
    });
  }

  function selectForecastRegion(id) {
    st.fcSel = id;
    if (st.view === 'forecast') drawForecastLayer();
    renderForecastInspector();
    loadForecastDetail(id);
  }

  async function loadForecastDetail(id) {
    $('fc-panel').innerHTML = stateBox('loading', 'Loading region history…');
    try {
      const data = await request('fcd', `/api/forecast/${encodeURIComponent(id)}?threshold=${encodeURIComponent($('fc-threshold').value)}`);
      if (!data) return;
      if (!data.region || !data.observed_weekly) throw new Error('Unexpected response from the server: region detail missing.');
      st.fcDetail = data;
      renderForecastInspector();
      renderForecastPanel();
    } catch (err) {
      $('fc-panel').innerHTML = stateBox('error', `Region detail unavailable: ${err.message}`, 'fcd');
    }
  }

  function metricsTable(m, names) {
    return table(['Model (final test period)', 'MAE', 'RMSE', 'Poisson dev.', '80% cov.', '95% cov.'],
      names.filter((n) => m[n]).map((n) => [esc(n), f(m[n].mae), f(m[n].rmse), f(m[n].poisson_deviance), f(m[n].coverage_80), f(m[n].coverage_95)]));
  }

  function renderForecastInspector() {
    const d = st.fc;
    const sel = d.regions.find((r) => r.region === st.fcSel);
    const top = [...d.regions].sort((a, b) => b.expected - a.expected).slice(0, 40);
    let detail = '';
    if (sel) {
      const det = st.fcDetail && st.fcDetail.region.region === sel.region ? st.fcDetail : null;
      const b = sel.baselines;
      detail = `<section class="detail"><div class="detail-head"><div><div class="kicker">Seven-day forecast · M${esc(d.threshold)}+</div><h2>${esc(sel.label)}</h2></div>
        <button class="icon-btn" id="fc-clear" aria-label="Clear region selection">×</button></div>
        <div class="score-head"><strong>${f(sel.expected, 2)}</strong><span class="band">expected events, ${esc(isoDay(d.valid_from))} → ${esc(isoDay(d.valid_to))}</span></div>
        ${dl([['80% interval', `${sel.interval_80[0]}–${sel.interval_80[1]} events`], ['95% interval', `${sel.interval_95[0]}–${sel.interval_95[1]} events`],
          ['P(at least one)', f(sel.prob_at_least_one, 3)], ['Observed last week', sel.last_week_observed]])}
        <h3 class="sub">Baselines for the same week</h3>
        ${table(['Method', 'Expected'], [['Model', f(sel.expected, 2)], ['2001–2015 mean', f(b.historical_mean_2001_2015, 2)],
          ['Last 4 weeks mean', f(b.rolling_mean_4w, 2)], ['Last 52 weeks mean', f(b.rolling_mean_52w, 2)]])}
        ${det && det.region_test_metrics ? `<h3 class="sub">This cell on the final test period</h3>${dl([
          ['Test weeks', det.region_test_metrics.test.n], ['MAE (model)', f(det.region_test_metrics.test.mae)],
          ['Poisson deviance (model)', f(det.region_test_metrics.test.poisson_deviance)],
          ['Poisson deviance (historical mean)', f(det.region_test_metrics.baseline_historical_mean.poisson_deviance)],
          ['80% interval coverage', f(det.region_test_metrics.test.coverage_80)]])}` : ''}
        ${det ? `<h3 class="sub">Inputs at the forecast origin</h3>${table(['Feature', 'Value', 'Unit'], det.features.map((x) =>
          [`<abbr title="${esc(x.definition)}">${esc(x.name)}</abbr>`, f(x.value, 2), esc(x.unit)]))}` : ''}
      </section>`;
    }
    $('fc-inspector').innerHTML = `${detail}<header class="list-head"><h2>Highest expected counts</h2><span class="muted small">${d.regions.length} cells</span></header>
      <div class="event-list">${top.map((r) => `<button type="button" class="event-row" data-region="${esc(r.region)}" aria-pressed="${r.region === st.fcSel}">
        <span class="mag-badge" style="--c:${ML.forecastColor(r.expected)}">${f(r.expected, 1)}</span>
        <span class="event-main"><span class="event-place">${esc(r.label)}</span><span class="event-meta">80%: ${r.interval_80[0]}–${r.interval_80[1]} · last week ${r.last_week_observed}</span></span>
        <span class="event-depth">${esc(r.region)}</span></button>`).join('')}</div>`;
    $('fc-inspector').querySelectorAll('[data-region]').forEach((b) => b.addEventListener('click', () => {
      selectForecastRegion(b.dataset.region);
      const r = d.regions.find((x) => x.region === b.dataset.region);
      fitMap(ML.boundsToLatLngs(r.bounds));
    }));
    if (sel) $('fc-clear').addEventListener('click', () => { st.fcSel = null; st.fcDetail = null; drawForecastLayer(); renderForecastInspector(); renderForecastPanel(); });
  }

  function renderForecastPanel() {
    const d = st.fc;
    const m = d.model;
    const names = Object.keys(m.test_metrics);
    const dw = d.data_window;
    const modelCard = card('Model and evaluation', `Selected: ${m.name}`, `
      <p class="small">${esc(d.target)}</p>
      ${metricsTable(m.test_metrics, names)}
      <p class="small muted">Final test period ${esc(m.test_period)}, walk-forward with yearly refits; not used for any model choice. Intervals: ${esc(m.interval_method)}, α = ${f(m.alpha)}. Served model trained on ${esc(m.training_window.from)} → ${esc(m.training_window.to)} (trained ${esc(isoTime(m.trained_at))}).</p>
      <p class="small muted">Features use USGS data ${esc(isoTime(dw.used_from))} → ${esc(isoTime(dw.used_to))}; latest event ${esc(isoTime(dw.latest_event))}; fetched ${esc(isoTime(dw.fetched_at))}${dw.stale ? ' <span class="warn">(stale copy)</span>' : ''}.</p>
      <p class="small">${esc(d.interpretation)}</p>`);
    const det = st.fcDetail && st.fcDetail.region.region === st.fcSel ? st.fcDetail : null;
    if (!det) {
      const top = [...d.regions].sort((a, b) => b.expected - a.expected).slice(0, 15);
      $('fc-panel').innerHTML = `<div class="grid-2">${card('Highest expected counts', 'Next 7 days', CH.barChart({
        bins: top.map((r) => ({ label: r.region.slice(1), count: Number(r.expected.toFixed(2)) })), xLabel: 'Cell (lower-left corner)', yLabel: 'Expected events', color: '#2f8f7f', title: 'Expected events in the next 7 days, top 15 cells' }))}${modelCard}</div>`;
      return;
    }
    const obs = det.observed_weekly;
    const weekMs = obs.week_starts.map((w) => Date.parse(`${w}T00:00:00Z`));
    const observed = obs.counts.map((c, i) => ({ x: weekMs[i] + 3.5 * DAY, y: c }));
    const back = det.backtest || {};
    const backMs = (back.week_starts || []).map((w) => Date.parse(`${w}T00:00:00Z`) + 3.5 * DAY);
    const firstObs = weekMs[0];
    const backPts = (back.forecast || []).map((v, i) => ({ x: backMs[i], y: v })).filter((p) => p.x >= firstObs - 26 * 7 * DAY);
    const basePts = (back.baseline_historical_mean || []).map((v, i) => ({ x: backMs[i], y: v })).filter((p) => p.x >= firstObs - 26 * 7 * DAY);
    const backObs = (back.observed || []).map((v, i) => ({ x: backMs[i], y: v })).filter((p) => p.x >= firstObs - 26 * 7 * DAY && p.x < firstObs);
    const origin = Date.parse(d.origin) + 3.5 * DAY;
    const r = det.region;
    const chart = CH.lineChart({
      title: `Weekly M${d.threshold}+ counts and forecasts for ${r.label}`, height: 300, width: 900,
      x: { type: 'time', label: 'Week (UTC)' }, y: { label: `Events per week (M${d.threshold}+)` },
      series: [
        { name: 'Observed weekly count', kind: 'bar', color: '#3b5566', data: backObs.concat(observed) },
        { name: 'Out-of-sample forecast (walk-forward test)', kind: 'line', color: '#4fd1c5', data: backPts },
        { name: 'Baseline: historical mean', kind: 'line', color: '#8ea2b0', dash: true, data: basePts },
      ],
      markers: [
        { x: origin, y: r.expected, lo: r.interval_95[0], hi: r.interval_95[1], color: '#ffb454', label: '95% interval' },
        { x: origin, y: r.expected, lo: r.interval_80[0], hi: r.interval_80[1], color: '#ff7a59', label: `Next 7 days: ${r.expected} (80% ${r.interval_80[0]}–${r.interval_80[1]})`, legend: 'Next-week forecast with 80% / 95% interval' },
      ],
    });
    $('fc-panel').innerHTML = `${card('Observed counts vs forecasts', r.region, `${chart}
      <p class="small muted">Bars are observed counts (catalogue data). The teal line is what the evaluation run forecast for those weeks without seeing them; it ends where the evaluation data end. The orange point is the forecast for ${esc(isoDay(d.valid_from))} → ${esc(isoDay(d.valid_to))}.</p>`, 'wide')}
      <div class="grid-2">${modelCard}${card('How to read this', 'Limits', `<ul class="small muted tight">
        <li>Counts are for the whole 10° cell; nothing is said about where in the cell, when in the week, or how large any event will be.</li>
        <li>Most skill comes from persistence: recent counts predict next week's count. A sequence that has not started cannot be anticipated.</li>
        <li>Intervals are negative-binomial quantiles; on the test period they covered more weeks than their nominal level (they are conservative).</li></ul>`)}</div>`;
  }

  // ===========================================================================
  // Anomaly Detection
  // ===========================================================================
  function anomalyQuery() {
    return new URLSearchParams({ threshold: $('an-threshold').value, window_weeks: $('an-window').value,
      reference_weeks: $('an-reference').value, fdr: $('an-fdr').value });
  }

  async function loadAnomalies() {
    $('an-panel').innerHTML = stateBox('loading', 'Testing every cell against its reference period…');
    $('an-inspector').innerHTML = stateBox('loading', 'Loading…');
    setViewStatus('Loading…');
    try {
      const data = await request('an', `/api/anomalies?${anomalyQuery()}`);
      if (!data) return;
      const v = ML.validateAnomalies(data);
      if (!v.ok) throw new Error(v.error);
      if (!st.an) worldView();
      st.an = data;
      if (!data.results.some((r) => r.region === st.anSel)) { st.anSel = null; st.anDetail = null; }
      const gaps = data.data_quality.possible_gap_days_in_window.length + data.data_quality.possible_gap_days_in_reference.length;
      setViewStatus(`${data.regions_elevated} of ${data.regions_tested} tested cells elevated${gaps ? ` · ${gaps} possible feed-gap days` : ''}${data.data_window.stale ? ' · stale history' : ''}`,
        gaps || data.data_window.stale ? 'warn' : 'info');
      if (st.view === 'anomaly') drawAnomalyLayer();
      renderAnomalyInspector();
      if (st.anSel) loadAnomalyDetail(st.anSel); else renderAnomalyPanel();
    } catch (err) {
      st.an = null;
      if (layer && st.view === 'anomaly') layer.clearLayers();
      setViewStatus('Anomaly test unavailable', 'error');
      $('an-panel').innerHTML = stateBox('error', `Anomaly detection unavailable: ${err.message}`, 'an');
      $('an-inspector').innerHTML = stateBox('error', 'No results. Missing data is never treated as zero activity.', 'an');
    }
  }

  function drawAnomalyLayer() {
    if (!layer) return;
    layer.clearLayers();
    st.an.results.forEach((r) => {
      const sel = r.region === st.anSel;
      if (r.status === 'not_elevated' && !sel) return;
      const elevated = r.status === 'elevated';
      const rect = L.rectangle(ML.boundsToLatLngs(r.bounds), {
        color: sel ? '#ffffff' : elevated ? '#fbbf24' : '#8ea2b0', weight: sel ? 2.5 : elevated ? 2 : 1,
        dashArray: elevated ? null : '4 4', fillColor: elevated ? '#fbbf24' : '#8ea2b0', fillOpacity: elevated ? 0.28 : 0.05,
        bubblingMouseEvents: false,
      });
      rect.bindTooltip(`${esc(r.label)}: ${esc(ML.anomalyStatusLabel(r.status))} (${r.observed} observed${r.expected !== null ? `, ~${C.fmt(r.expected, 1)} expected` : ''})`, { sticky: true });
      rect.on('click', () => selectAnomalyRegion(r.region));
      rect.addTo(layer);
    });
  }

  function selectAnomalyRegion(id) {
    st.anSel = id;
    st.anDetail = null;
    if (st.view === 'anomaly') drawAnomalyLayer();
    renderAnomalyInspector();
    loadAnomalyDetail(id);
  }

  async function loadAnomalyDetail(id) {
    $('an-panel').innerHTML = stateBox('loading', 'Loading weekly counts and contributing events…');
    try {
      const data = await request('and', `/api/anomalies/${encodeURIComponent(id)}?${anomalyQuery()}`);
      if (!data) return;
      if (!data.region || !data.weekly) throw new Error('Unexpected response from the server: region detail missing.');
      st.anDetail = data;
      renderAnomalyInspector();
      renderAnomalyPanel();
    } catch (err) {
      $('an-panel').innerHTML = stateBox('error', `Region detail unavailable: ${err.message}`, 'and');
    }
  }

  function renderAnomalyInspector() {
    const d = st.an;
    const sel = d.results.find((r) => r.region === st.anSel);
    const dq = d.data_quality;
    const head = `<section class="detail"><div class="kicker">Test definition · M${esc(d.threshold)}+</div>
      ${dl([['Observation window', `${esc(isoTime(d.observation_window.from))} → ${esc(isoTime(d.observation_window.to))}`],
        ['Reference window', `${esc(isoTime(d.reference_window.from))} → ${esc(isoTime(d.reference_window.to))}`],
        ['Cells tested', d.regions_tested], ['Flag rule', `BH q ≤ ${esc(d.fdr)} and ≥ 3 events`],
        ['Possible feed gaps', dq.possible_gap_days_in_window.length + dq.possible_gap_days_in_reference.length ? `<span class="warn">${esc([...dq.possible_gap_days_in_window, ...dq.possible_gap_days_in_reference].join(', '))}</span>` : 'None detected'],
        ['Long-term dispersion', esc(dq.long_term_dispersion)]])}</section>`;
    let detail = '';
    if (sel) {
      const ev = st.anDetail && st.anDetail.region.region === sel.region ? st.anDetail.contributing_events : null;
      detail = `<section class="detail"><div class="detail-head"><div><div class="kicker">${esc(ML.anomalyStatusLabel(sel.status))}</div><h2>${esc(sel.label)}</h2></div>
        <button class="icon-btn" id="an-clear" aria-label="Clear region selection">×</button></div>
        <p class="small">${esc(ML.explainAnomaly(sel, d))}</p>
        ${dl([['Observed', sel.observed], ['Expected (reference)', sel.expected === null ? '—' : f(sel.expected, 2)],
          ['Reference total / weekly mean', `${sel.reference_total} / ${f(sel.reference_weekly_mean, 2)}`],
          ['Reference SD of window total', f(sel.reference_sd, 2)], ['Weekly dispersion used', f(sel.weekly_dispersion, 3)],
          ['p-value (one-sided)', esc(ML.formatP(sel.p_value))], ['q-value (BH)', esc(ML.formatP(sel.q_value))],
          sel.largest_event ? ['Largest event in window', `M ${f(sel.largest_event.mag, 1)} · ${esc(sel.largest_event.place || sel.largest_event.id)}`] : null])}
        ${sel.largest_event ? `<div class="detail-actions"><button class="btn-small" id="an-aftershock">Analyse activity after the largest event</button></div>` : ''}
        ${ev ? `<h3 class="sub">Contributing events (${ev.length})</h3><div class="member-list">${ev.slice(0, 80).map((e) => `<div class="member static"><span>M ${f(e.mag, 1)}</span><span>${esc(C.formatUtc(e.time))}</span><span class="muted">${esc(e.place || e.id)}</span></div>`).join('')}</div>` : ''}
      </section>`;
    }
    const rows = [...d.results].sort((a, b) => (a.status === 'elevated' ? 0 : 1) - (b.status === 'elevated' ? 0 : 1) || (a.q_value ?? 2) - (b.q_value ?? 2)).slice(0, 40);
    $('an-inspector').innerHTML = `${head}${detail}<header class="list-head"><h2>Cells <span class="count">${d.results.length}</span></h2><span class="muted small">Elevated first</span></header>
      <div class="event-list">${rows.length ? rows.map((r) => `<button type="button" class="event-row" data-region="${esc(r.region)}" aria-pressed="${r.region === st.anSel}">
        <span class="status-dot" data-status="${esc(r.status)}" aria-hidden="true"></span>
        <span class="event-main"><span class="event-place">${esc(r.label)}</span><span class="event-meta">${esc(ML.anomalyStatusLabel(r.status))} · ${r.observed} obs${r.expected !== null ? ` / ${C.fmt(r.expected, 1)} exp` : ''}</span></span>
        <span class="event-depth">q ${esc(ML.formatP(r.q_value))}</span></button>`).join('') : '<div class="state">No cell had qualifying events.</div>'}</div>`;
    $('an-inspector').querySelectorAll('[data-region]').forEach((b) => b.addEventListener('click', () => {
      selectAnomalyRegion(b.dataset.region);
      fitMap(ML.boundsToLatLngs(d.results.find((x) => x.region === b.dataset.region).bounds));
    }));
    if (sel) {
      $('an-clear').addEventListener('click', () => { st.anSel = null; st.anDetail = null; drawAnomalyLayer(); renderAnomalyInspector(); renderAnomalyPanel(); });
      const btn = $('an-aftershock');
      if (btn) btn.addEventListener('click', () => openAftershock(sel.largest_event.id));
    }
  }

  function renderAnomalyPanel() {
    const d = st.an;
    const method = card('Method', 'Negative binomial + BH', `<p class="small">${esc(d.method)}</p><p class="small">${esc(d.interpretation)}</p>
      <ul class="small muted tight"><li>Each cell is compared only with its own history, because catalogue completeness differs between regions.</li>
      <li>Days with no M4.5+ event anywhere are treated as possible feed gaps; affected results are marked incomplete instead of normal.</li>
      <li>Only excess activity is tested; quiet periods are not flagged.</li></ul>`);
    const det = st.anDetail && st.anDetail.region.region === st.anSel ? st.anDetail : null;
    if (!det) {
      const flagged = d.results.filter((r) => r.status === 'elevated');
      $('an-panel').innerHTML = `<div class="grid-2">${card('Elevated cells', `${flagged.length} flagged`, flagged.length
        ? table(['Cell', 'Observed', 'Expected', 'q'], flagged.map((r) => [esc(r.label), r.observed, f(r.expected, 1), esc(ML.formatP(r.q_value))]))
        : '<p class="small muted">No cell is elevated under the current test definition. This does not mean activity is low or that any region is safe.</p>')}${method}</div>`;
      return;
    }
    const w = det.weekly;
    const ends = w.week_ends.map((s) => Date.parse(s));
    const refMean = det.region.reference_weekly_mean;
    const bars = w.counts.map((c, i) => ({ x: ends[i] - 3.5 * DAY, y: c, color: i >= w.reference_weeks ? '#fbbf24' : '#3b5566' }));
    const chart = CH.lineChart({
      title: `Weekly M${d.threshold}+ counts in ${det.region.label}: reference weeks and observation window`, height: 260, width: 900,
      x: { type: 'time', label: 'Week ending (UTC)' }, y: { label: `Events per week (M${d.threshold}+)` },
      series: [{ name: 'Weekly count (amber = observation window)', kind: 'bar', color: '#3b5566', data: bars },
        { name: 'Reference weekly mean', kind: 'line', color: '#8ea2b0', dash: true, data: [{ x: ends[0] - 7 * DAY, y: refMean }, { x: ends[ends.length - 1], y: refMean }] }],
    });
    $('an-panel').innerHTML = `${card('Observed activity vs reference', det.region.region, chart, 'wide')}<div class="grid-2">${method}${card('Why this result', ML.anomalyStatusLabel(det.region.status), `<p class="small">${esc(ML.explainAnomaly(det.region, d))}</p>`)}</div>`;
  }

  // ===========================================================================
  // Aftershock Analysis
  // ===========================================================================
  function renderAftershockIdle() {
    if (st.as) return;
    $('as-panel').innerHTML = stateBox('info', 'Choose a reference earthquake: enter a USGS event id, or use “Analyse activity after this event” in Live Monitor, Sequence Explorer or Anomaly Detection.');
    $('as-inspector').innerHTML = '<div class="state">No reference event selected.</div>';
  }

  function openAftershock(eventId, { radius = null } = {}) {
    if (!eventId) return;
    $('as-event').value = eventId;
    $('as-radius').value = radius === null ? '' : String(radius);
    st.loaded.aftershock = true;
    setView('aftershock');
    runAftershock();
  }

  async function runAftershock() {
    const form = $('as-form');
    if (!form.checkValidity()) { form.reportValidity(); return; }
    const params = new URLSearchParams({ event_id: $('as-event').value.trim(), pre_days: $('as-pre').value, post_days: $('as-post').value,
      min_mag: $('as-minmag').value, fit_start_hours: $('as-fit-start').value });
    [['radius_km', 'as-radius'], ['min_depth', 'as-min-depth'], ['max_depth', 'as-max-depth']].forEach(([k, id]) => { if ($(id).value !== '') params.set(k, $(id).value); });
    $('as-panel').innerHTML = stateBox('loading', 'Querying USGS around the reference event and fitting…');
    $('as-inspector').innerHTML = stateBox('loading', 'Loading…');
    setViewStatus('Loading…');
    try {
      const data = await request('as', `/api/aftershocks?${params}`);
      if (!data) return;
      const v = ML.validateAftershock(data);
      if (!v.ok) throw new Error(v.error);
      st.as = data;
      setViewStatus(`${data.statistics.counts.post} events after · fit: ${data.fit.status.replace('_', ' ')}${data.data_window.truncated ? ' · truncated' : ''}`,
        data.fit.status === 'fitted' ? 'info' : 'warn');
      if (st.view === 'aftershock') drawAftershockLayer(true);
      renderAftershock();
    } catch (err) {
      setViewStatus('Analysis failed', 'error');
      $('as-panel').innerHTML = stateBox('error', `Aftershock analysis unavailable: ${err.message}`, 'as');
      $('as-inspector').innerHTML = stateBox('error', 'No analysis was computed.', 'as');
    }
  }

  function drawAftershockLayer(zoom = false) {
    if (!layer) return;
    layer.clearLayers();
    const d = st.as, ref = d.reference_event;
    const circle = L.circle([ref.latitude, ref.longitude], { radius: d.parameters.radius_km * 1000, color: '#c084fc', weight: 1, dashArray: '4 4', fill: false, interactive: false });
    circle.addTo(layer);
    const span = Math.max(d.statistics.observed_post_days, 1e-3);
    d.events.forEach((e) => {
      const c = ML.timeColor(e.t_days / span);
      L.circleMarker([e.latitude, e.longitude], { radius: C.magRadius(e.mag) * 0.8, color: c, weight: 0.5, fillColor: c, fillOpacity: 0.75, bubblingMouseEvents: false })
        .bindTooltip(`M ${esc(C.fmt(e.mag))} · ${esc(C.fmt(e.t_days, 2))} d after · ${esc(C.fmt(e.distance_km, 0))} km`, { direction: 'top' }).addTo(layer);
    });
    L.circleMarker([ref.latitude, ref.longitude], { radius: 10, color: '#ffffff', weight: 2, fillColor: '#ff4d6d', fillOpacity: 0.9 })
      .bindTooltip(`Reference: M ${esc(C.fmt(ref.mag))} ${esc(ref.place || ref.id)}`, { direction: 'top' }).addTo(layer);
    if (zoom) A.state.map.fitBounds(circle.getBounds().pad(0.2), { maxZoom: 9 });
  }

  function renderAftershock() {
    const d = st.as, ref = d.reference_event, s = d.statistics, fit = d.fit, pc = s.pre_comparison;
    const fitRows = fit.parameters ? [['p (decay exponent)', `${f(fit.parameters.p, 3)}${fit.confidence_95 ? ` [${esc(fit.confidence_95.p.join(' – '))}]` : ''}`],
      ['c (days)', fit.c_fixed ? `${esc(fit.parameters.c_days)} (fixed: not resolvable)` : `${esc(fit.parameters.c_days)}${fit.confidence_95?.c_days ? ` [${esc(fit.confidence_95.c_days.join(' – '))}]` : ''}`],
      ['K (events/day)', f(fit.parameters.K, 3)], ['Background μ (fixed)', `${f(fit.background_rate_per_day, 4)} /day`],
      ['Converged', fit.diagnostics.converged ? 'Yes' : 'No'], ['KS p (transformed times)', esc(ML.formatP(fit.diagnostics.ks_p_value))],
      ['ΔAIC vs constant rate', f(fit.diagnostics.delta_aic_vs_constant, 1)],
      ['Expected / observed in fit window', `${f(fit.diagnostics.expected_events, 1)} / ${fit.diagnostics.observed_events}`]] : [];
    const inCompare = st.compare.some((a) => a.reference_event.id === ref.id && a.parameters.radius_km === d.parameters.radius_km);
    $('as-inspector').innerHTML = `<section class="detail">
      <div class="detail-head"><span class="mag-badge big" style="--c:${C.magColor(ref.mag)}">${f(ref.mag, 1)}</span>
      <div><div class="kicker">Reference event</div><h2>${esc(ref.place || ref.id)}</h2></div></div>
      ${dl([['Time (UTC)', esc(C.formatUtc(ref.time))], ['Coordinates', esc(C.formatCoord(ref.latitude, ref.longitude))], ['Depth', ref.depth_km === null ? 'Not reported' : `${f(ref.depth_km, 1)} km`],
        ['Radius', `${f(d.parameters.radius_km, 0)} km${d.parameters.radius_default ? ' (rupture-length rule)' : ''}`],
        ['Events after / before', `${s.counts.post} / ${s.counts.pre}`], ['Days observed after', f(s.observed_post_days, 2)],
        pc ? ['Rate after vs before', `${f(pc.post_rate_per_day, 3)} vs ${f(pc.pre_rate_per_day, 3)} /day${pc.rate_ratio ? ` (×${f(pc.rate_ratio, 1)})` : ''}`] : null,
        pc ? ['p (rate after > before)', esc(ML.formatP(pc.p_value_post_rate_higher))] : null])}
      <div class="detail-actions">${ref.url ? `<a class="btn-small" href="${esc(ref.url)}" target="_blank" rel="noopener noreferrer">USGS event page ↗</a>` : ''}
        <button class="btn-small" id="as-compare" ${inCompare || st.compare.length >= 4 ? 'disabled' : ''}>${inCompare ? 'In comparison' : 'Add to comparison'}</button></div>
    </section>
    <section class="detail"><div class="kicker">Omori–Utsu fit · ${esc(fit.status.replace('_', ' '))}</div>
      <p class="small">${esc(ML.explainFit(fit))}</p>
      ${(fit.notes || []).map((n) => `<p class="small muted">${esc(n)}</p>`).join('')}
      ${fitRows.length ? dl(fitRows) : ''}
      <p class="small muted">${esc(d.model_note)}</p></section>
    <section class="detail"><div class="kicker">Selection rules</div><ul class="small tight">${d.selection_rules.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>
      <p class="small muted">Window ${esc(isoTime(d.data_window.from))} → ${esc(isoTime(d.data_window.to))}. ${esc(d.data_window.note)}</p>
      ${d.data_window.truncated ? '<p class="warn small">The USGS query hit its response limit; the event list is incomplete and no fit was attempted.</p>' : ''}</section>
    ${st.compare.length ? `<section class="detail"><div class="kicker">Comparison (${st.compare.length}/4)</div>${table(['Sequence', 'Events', 'p', ''], ML.compareRows(st.compare).map((r, i) =>
      [esc(r.label), r.post, r.p === null ? esc(r.status.replace('_', ' ')) : f(r.p, 2), `<button class="btn-small" data-uncompare="${i}" aria-label="Remove ${esc(r.label)} from comparison">Remove</button>`]))}</section>` : ''}`;
    $('as-compare').addEventListener('click', () => { st.compare.push(d); renderAftershock(); });
    $('as-inspector').querySelectorAll('[data-uncompare]').forEach((b) => b.addEventListener('click', () => { st.compare.splice(Number(b.dataset.uncompare), 1); renderAftershock(); }));

    const mid = (b) => (b.from > 0 ? Math.sqrt(b.from * b.to) : b.to / 2);
    const ratePts = s.log_time.filter((b) => b.count > 0).map((b) => ({ x: mid(b), y: b.rate_per_day, title: `${b.count} events, ${C.fmt(b.from, 3)}–${C.fmt(b.to, 3)} d` }));
    const curvePts = (fit.curve_log_time || []).map((b) => ({ x: mid(b), y: b.rate_per_day }));
    const rate = CH.lineChart({ title: 'Event rate versus time after the reference event (log–log)', height: 240,
      x: { type: 'log', label: 'Days after reference event (log scale)' }, y: { log: true, label: 'Events per day (log scale)' },
      series: [{ name: 'Observed rate (log-time bins)', kind: 'dot', color: '#ffb454', data: ratePts },
        { name: 'Fitted Omori–Utsu rate (fit window only)', kind: 'line', color: '#4fd1c5', data: curvePts }],
      empty: 'No events after the reference event with these criteria.' });
    const cum = CH.lineChart({ title: 'Cumulative number of events after the reference event', height: 240, x: { label: 'Days after reference event' }, y: { label: 'Cumulative events' },
      series: [{ name: 'Cumulative count', kind: 'step', color: '#4fd1c5', data: [{ x: 0, y: 0 }, ...s.cumulative.map((p) => ({ x: p.t_days, y: p.n })), { x: s.observed_post_days, y: s.counts.post }] }] });
    const daily = CH.lineChart({ title: 'Events per day after the reference event', height: 220, width: 900, x: { label: 'Days after reference event' }, y: { label: 'Events per day' },
      series: [{ name: 'Daily count', kind: 'bar', color: '#6b8cff', data: s.daily.map((b) => ({ x: (b.from + b.to) / 2, y: b.count })) }] });
    let comparison = '';
    if (st.compare.length >= 2) {
      const series = st.compare.map((a, i) => ({ name: ML.compareRows([a])[0].label, kind: 'step', color: ML.sequenceColor(i),
        data: [{ x: 0, y: 0 }, ...a.statistics.cumulative.map((p) => ({ x: p.t_days, y: p.n }))] }));
      comparison = card('Sequence comparison', 'Cumulative events', `${ML.sameDefinitions(st.compare) ? '' : '<p class="warn small">These analyses use different magnitude, depth or window settings; counts are not directly comparable.</p>'}
        ${CH.lineChart({ title: 'Cumulative events after each reference event', height: 260, width: 900, x: { label: 'Days after reference event' }, y: { label: 'Cumulative events' }, series })}
        ${table(['Sequence', 'Settings', 'Events', 'Days', 'p'], ML.compareRows(st.compare).map((r) => [esc(r.label), esc(r.params), r.post, f(r.days, 1), r.p === null ? '—' : f(r.p, 2)]))}`, 'wide');
    }
    $('as-panel').innerHTML = `<div class="grid-2">${card('Decay', fit.status === 'fitted' ? 'Fitted' : 'Observed only', rate)}${card('Cumulative count', `${s.counts.post} events`, cum)}</div>
      ${card('Daily counts', 'Zero = no catalogue event', `${daily}<p class="small muted">${esc(s.binning_note)}</p>`, 'wide')}
      <div class="grid-3">${card('Magnitude', 'Events after', CH.barChart({ bins: CH.histogramBins(s.magnitude, 1), xLabel: 'Magnitude', yLabel: 'Events', color: '#ffb454', title: 'Magnitude histogram after the reference event' }))}
      ${card('Depth', 'Events after', CH.barChart({ bins: CH.histogramBins(s.depth_km, 0), xLabel: 'Depth (km)', yLabel: 'Events', color: '#6b8cff', title: 'Depth histogram after the reference event', empty: 'No depths reported.' }))}
      ${card('Distance', 'From epicentre', CH.barChart({ bins: CH.histogramBins(s.distance_km, 0), xLabel: 'Distance (km)', yLabel: 'Events', color: '#c084fc', title: 'Epicentral distance histogram' }))}</div>
      ${comparison}`;
  }

  // ===========================================================================
  // Model Performance
  // ===========================================================================
  async function loadPerformance() {
    $('perf-panel').innerHTML = stateBox('loading', 'Loading evaluation results…');
    try {
      const data = await request('perf', '/api/model-performance');
      if (!data) return;
      const v = ML.validatePerformance(data);
      if (!v.ok) throw new Error(v.error);
      st.perf = data;
      renderPerformance();
    } catch (err) {
      $('perf-panel').innerHTML = stateBox('error', `Evaluation results unavailable: ${err.message}`, 'perf');
    }
  }

  function forecastPerf(thr, m) {
    if (m.status !== 'loaded') return card(`Forecast M${thr}+`, 'Unavailable', `<p class="small warn">${esc(m.error)}</p>`);
    const sel = m.selected_model;
    const p = m.periods;
    const valRows = Object.entries(m.validation_metrics).map(([n, v]) => [esc(n) + (n === sel ? ' ✓' : ''), f(v.mae), f(v.rmse), f(v.poisson_deviance), f(v.nb_log_score)]);
    const years = Object.keys(m.test_by_year);
    const yearSeries = [sel, 'historical_mean', 'rolling_mean_4w'].map((n, i) => ({ name: n, kind: 'line', color: ['#4fd1c5', '#8ea2b0', '#ffb454'][i], dash: i > 0,
      data: years.map((y) => ({ x: Number(y), y: m.test_by_year[y][n]?.poisson_deviance })) }));
    const worse = m.test_by_region.filter((r) => r.poisson_deviance > r.baseline_deviance).length;
    const busiest = [...m.test_by_region].sort((a, b) => b.mean_observed - a.mean_observed).slice(0, 10);
    return `<h2 class="perf-h">Seven-day regional forecast · M${esc(thr)}+</h2>
      <p class="small muted">${esc(m.target)} Trained ${esc(isoTime(m.trained_at))}; ${m.regions} cells (${esc(m.eligibility)}). ${esc(m.selection_rule)}</p>
      <div class="grid-2">
        ${card('Validation — used for model selection', `${p.validation.from} → ${p.validation.to}`, table(['Model', 'MAE', 'RMSE', 'Poisson dev.', 'NB log score'], valRows) + `<p class="small muted">Models fit on ${esc(p.train.from)} → ${esc(p.train.to)} (${m.samples.train} samples).</p>`)}
        ${card('Final test — held out', `${p.test.from} → ${p.test.to}`, metricsTable(m.test_metrics, Object.keys(m.test_metrics)) +
          `<p class="small muted">${esc(p.test.method)}; ${m.samples.test} region-weeks. Deviance skill of ${esc(sel)} (95% interval from resampling whole weeks): ${Object.entries(m.deviance_skill_vs).map(([k, v]) => { const ci = m.deviance_skill_ci95_week_bootstrap?.[k]; return `${esc(k)} ${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%${ci ? ` [${(ci[0] * 100).toFixed(1)}, ${(ci[1] * 100).toFixed(1)}]` : ''}`; }).join(', ')}.</p>`)}
        ${card('Test deviance by year', 'Lower is better', CH.lineChart({ title: 'Mean Poisson deviance by test year', x: { label: 'Year' }, y: { label: 'Mean Poisson deviance' }, series: yearSeries }))}
        ${card('Predictive reliability (test)', m.dispersion_curve ? `α(μ) = exp(${C.fmt(m.dispersion_curve.a, 2)} ${m.dispersion_curve.b < 0 ? '−' : '+'} ${C.fmt(Math.abs(m.dispersion_curve.b), 2)} log μ)` : `α = ${C.fmt(m.alpha, 3)}`,
          table(['Predicted', 'n', 'Mean pred.', 'Mean obs.', '80% cov.', 'Model-implied', 'Width'],
          m.test_reliability.map((r) => [`${r.predicted_from}–${r.predicted_to ?? '∞'}`, r.n, f(r.mean_predicted), f(r.mean_observed), f(r.coverage_80), f(r.model_implied_coverage_80), f(r.mean_width_80, 2)])) +
          `<p class="small muted">${esc(m.interval_method || '')}. Counts are integers, so no interval holds exactly 80%: “model-implied” is the probability the forecast itself gives its interval (at least 0.80). Calibration means observed coverage is close to model-implied, not to 0.80.</p>`)}
        ${card('Busiest cells on the test period', `${worse} of ${m.test_by_region.length} cells worse than historical mean`, table(['Cell', 'Mean obs.', 'MAE', 'Dev. model', 'Dev. hist. mean'],
          busiest.map((r) => [esc(r.label), f(r.mean_observed, 2), f(r.mae), f(r.poisson_deviance), f(r.baseline_deviance)])))}
        ${card('Limitations', '', `<ul class="small muted tight">${m.limitations.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>`)}
      </div>`;
  }

  function renderPerformance() {
    const d = st.perf;
    const parts = [`<p class="small muted perf-note">${esc(d.note)}</p>`];
    Object.entries(d.forecast).forEach(([thr, m]) => parts.push(forecastPerf(thr, m)));

    const missing = (title, cmd) => `<h2 class="perf-h">${esc(title)}</h2><p class="small warn">Not evaluated yet: run <code>python ml_pipeline.py ${esc(cmd)}</code>. No metrics are shown for it.</p>`;
    const sq = d.sequences;
    if (!sq) parts.push(missing('Sequence clustering', 'sequences'));
    if (sq) {
      const rows = Object.entries(sq.scenarios).map(([n, s]) => [esc(n.replaceAll('_', ' ')), f(s.proposed.ari_mean), f(s.proposed.ari_min), f(s.naive_degrees.ari_mean), f(s.space_only.ari_mean), `${s.proposed.true_sequences} / ${esc(s.proposed.found_sequences.join(','))}`]);
      const live = st.seq ? `<p class="small">Latest run in Sequence Explorer: ${st.seq.summary.sequence_count} sequences, ${st.seq.summary.clustered_events} clustered and ${st.seq.summary.noise_events} unassigned events (ε ${st.seq.parameters.eps_km} km / ${st.seq.parameters.eps_hours} h, min ${st.seq.parameters.min_samples}). No ground truth exists for real events, so no accuracy is claimed for them.</p>` : '<p class="small muted">Open Sequence Explorer to see statistics for the current feed.</p>';
      parts.push(`<h2 class="perf-h">Sequence clustering · DBSCAN (synthetic labelled tests)</h2><div class="grid-2">
        ${card('Synthetic scenarios', `ε ${sq.parameters.eps_km} km, ${sq.parameters.eps_hours} h, min ${sq.parameters.min_samples}`, table(['Scenario', 'ARI', 'ARI min', 'ARI naive°', 'ARI space-only', 'true / found'], rows) +
          `<p class="small muted">${esc(sq.data)}. Seeds ${esc(sq.seeds.join(', '))}. “Naive°” is Euclidean distance on raw degrees; “space-only” ignores time.</p>`)}
        ${card('Overlap and sensitivity', 'ARI', table(['Overlapping sequences: ε km', 'ARI'], sq.overlapping_sequences_vs_eps_km.map((r) => [r.eps_km, f(r.ari_mean)])) +
          `<p class="small">${esc(sq.metrics_note)}</p>${live}`)}
        ${sq.method_comparison ? card('Alternative methods', 'mean ARI over scenarios', table(['Method', 'Mean ARI', 'Largest drop vs default'],
          Object.entries(sq.method_comparison.mean_ari_over_scenarios).map(([k, v]) => [esc(k), f(v), f(sq.method_comparison.largest_ari_drop_vs_default[k])])) +
          `<p class="small">Decision: ${esc(sq.method_comparison.decision)}.</p><p class="small muted">${esc(sq.method_comparison.selection_rule)}${sq.method_comparison.real_catalogue_split_check ? ' ' + esc(sq.method_comparison.real_catalogue_split_check.note) : ''}</p>` +
          (sq.substructure_check_candidate ? `<p class="small">Rejected candidate, a “possibly merged” check: it flagged ${sq.substructure_check_candidate.total.flagged_merged} of ${sq.substructure_check_candidate.total.merged_sequences} synthetic merges and ${sq.substructure_check_candidate.total.flagged_single} of ${sq.substructure_check_candidate.total.single_sequences} synthetic single sequences${sq.substructure_check_candidate.real_m7_mainshock_sequences ? `, but also ${sq.substructure_check_candidate.real_m7_mainshock_sequences.flagged} of ${sq.substructure_check_candidate.real_m7_mainshock_sequences.checked} real M7+ aftershock sequences` : ''}, so it is not shown.</p>` : '')) : ''}</div>`);
    }

    const an = d.anomaly;
    if (!an) parts.push(missing('Anomaly detection', 'anomaly'));
    if (an) {
      const c = an.configuration, ns = an.null_simulation, md = an.missing_data;
      parts.push(`<h2 class="perf-h">Anomaly detection · negative binomial + Benjamini–Hochberg</h2><div class="grid-2">
        ${card('False alerts (null simulation)', `${ns.simulated_runs} runs`, dl([['False-alert rate per cell test', f(ns.false_alert_rate_per_region_test, 4)],
          ['Runs with ≥ 1 false alert', `${f(ns.share_of_runs_with_any_false_alert, 3)}${ns.share_of_runs_with_any_false_alert_wilson95 ? ` (95% CI ${f(ns.share_of_runs_with_any_false_alert_wilson95[0], 3)}–${f(ns.share_of_runs_with_any_false_alert_wilson95[1], 3)})` : ''}`], ['Unadjusted p ≤ 0.05 rate', f(ns['unadjusted_p_below_0.05_rate'], 4)],
          ['Rolling-rule baseline false-alert rate', f(ns.baseline_rolling_rule_false_alert_rate, 4)]]) + `<p class="small muted">${esc(ns.construction)}</p>`)}
        ${card('Synthetic injected excess', `${c.window_weeks}-week window`, table(['Injected', 'Detected (BH)', 'p ≤ 0.05', 'Rolling rule'],
          Object.entries(an.synthetic_injection).map(([k, v]) => [esc(k), f(v.detection_rate), f(v['unadjusted_p_below_0.05_rate']), f(v.baseline_rolling_rule_detection_rate)])) +
          `<p class="small muted">${esc(an.injection_construction)} Synthetic results do not prove effectiveness on real geological phenomena.</p>`)}
        ${an.false_alert_diagnosis ? card('Why false alerts remain', `${an.false_alert_diagnosis.runs_per_null} runs per null`, table(['Null', ...Object.keys(Object.values(an.false_alert_diagnosis.nulls)[0]).map(esc)],
          Object.entries(an.false_alert_diagnosis.nulls).map(([k, v]) => [esc(k.replaceAll('_', ' ')), ...Object.values(v).map((x) => f(x.runs_with_false_alert, 3))])) +
          `<p class="small muted">Share of runs with at least one false alert. ${esc(an.false_alert_diagnosis.interpretation)}</p>`) : ''}
        ${card('Held-out real data', `${an.held_out_period.from} → ${an.held_out_period.to}`, table(['Threshold / window', 'Runs', 'Flags', 'Rate', 'Runs flagged', 'Rolling rule'],
          Object.entries(an.held_out_real_behaviour).map(([k, v]) => [esc(k), v.runs, v.flags, f(v.flag_rate_per_region_test, 4), f(v.share_of_runs_with_any_flag, 3), v.baseline_rolling_rule_flags])) +
          `<p class="small muted">${esc(an.held_out_note)}</p>`)}
        ${card('Missing-data behaviour', '', dl([['Global zero-event days in catalogue', md.global_zero_event_days_2000_to_end],
          ['Windows with 3 removed days detected as gaps', f(md.gap_check_detected_all_removed_days_share, 2)],
          ['Mean share of flags lost', f(md.mean_share_of_flags_lost_when_3_days_missing, 3)]]) + `<p class="small muted">${esc(md.note)}</p>`)}</div>`);
    }

    const af = d.aftershock;
    if (!af) parts.push(missing('Aftershock modelling', 'aftershock'));
    if (af) {
      const o = af.out_of_sample_30_to_90_days, pr = af.p_of_reliable_fits;
      const current = st.as ? `<p class="small">Current analysis: ${esc(st.as.reference_event.id)} — ${esc(ML.explainFit(st.as.fit))}</p>` : '';
      parts.push(`<h2 class="perf-h">Aftershock modelling · modified Omori–Utsu (historical sequences)</h2><div class="grid-2">
        ${card('Fit success across sequences', `${af.sequences_analysed} sequences`, dl([['Candidates (M ≥ 7)', af.candidates],
          ['Excluded (larger event before / within)', `${af.excluded.larger_event_in_prior_30_days} / ${af.excluded.larger_event_within_window}`],
          ...Object.entries(af.full_window_status_counts).map(([k, v]) => [`90-day fit: ${k.replace('_', ' ')}`, v]),
          ['p of reliable fits (median, IQR)', pr ? `${f(pr.median, 2)} (${f(pr.q25, 2)}–${f(pr.q75, 2)})` : '—']]) + `<p class="small muted">${esc(af.selection)}</p>`)}
        ${card('Out of sample: count in days 30–90', `${o.sequences_with_reliable_30_day_fit} sequences`, table(['Predictor', 'Median |log ratio|'],
          [['Omori–Utsu + background', f(o.median_abs_log_ratio_omori)], ['Last-week rate held constant', f(o.median_abs_log_ratio_last_week_rate)], ['Background only', f(o.median_abs_log_ratio_background_only)]]) +
          `<p class="small">Poisson 95% coverage of the Omori prediction: ${f(o.poisson_95_coverage_omori, 2)} — too narrow, because parameter uncertainty and clustering beyond Poisson are ignored.</p><p class="small muted">${esc(o.metric_note)}</p>${current}`)}
        ${o.predictive_intervals ? card('Predictive intervals, days 30–90 (mainshocks 2013+, out of sample)', `${o.predictive_intervals.mainshocks_2013_and_later.sequences} sequences`,
          table(['Method', '80% coverage', '95% coverage', 'Median 95% width'], Object.keys(o.predictive_intervals.methods).map((k) => { const v = o.predictive_intervals.mainshocks_2013_and_later[k];
            return [esc(k.replaceAll('_', ' ')), f(v['80'].coverage, 2), f(v['95'].coverage, 2), f(v['95'].median_width_events, 0)]; })) +
          `<p class="small muted">${esc(o.predictive_intervals.note)} Count dispersion d = ${f(o.predictive_intervals.count_dispersion_d, 3)}. Evaluated offline only: the Aftershock view shows fits of past data, not forecasts.</p>`) : ''}</div>`);
    }

    const lg = d.legacy_severity_model;
    if (lg && lg.status === 'loaded') {
      const t = lg.metrics?.test || {}, b = lg.baselines?.['Constant train base rate'] || {}, lr = lg.baselines?.['Logistic regression (same features)'] || {};
      parts.push(`<h2 class="perf-h">Legacy: conditional severity classifier (point analysis)</h2><div class="grid-2">${card('Test period', `${lg.split?.test?.from} → ${lg.split?.test?.to}`,
        table(['Model', 'ROC-AUC', 'Brier'], [['MLP (served)', f(t.roc_auc), f(t.brier)], ['Logistic regression', f(lr.roc_auc), f(lr.brier)], ['Constant base rate', f(b.roc_auc), f(b.brier)]]) +
        '<p class="small muted">P(M ≥ 5.0 | an M ≥ 4.5 event is recorded). Only marginally better than the base rate; kept for the point-analysis dialog and labelled exploratory.</p>')}</div>`);
    }
    $('perf-panel').innerHTML = parts.join('');
  }

  // ---------------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------------
  const RETRY = { seq: loadSequences, fc: loadForecast, fcd: () => st.fcSel && loadForecastDetail(st.fcSel), an: loadAnomalies,
    and: () => st.anSel && loadAnomalyDetail(st.anSel), as: runAftershock, perf: loadPerformance };

  function bind() {
    const tabs = [...document.querySelectorAll('[role="tab"][data-view]')];
    tabs.forEach((t) => {
      t.addEventListener('click', () => setView(t.dataset.view));
      t.addEventListener('keydown', (e) => {
        const i = ORDER.indexOf(t.dataset.view);
        const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: ORDER.length - 1 }[e.key];
        if (next === undefined) return;
        e.preventDefault();
        setView(ORDER[(next + ORDER.length) % ORDER.length], { focus: true });
      });
    });
    document.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-retry]');
      if (btn && RETRY[btn.dataset.retry]) RETRY[btn.dataset.retry]();
    });
    $('seq-form').addEventListener('submit', (e) => { e.preventDefault(); loadSequences(); });
    $('fc-threshold').addEventListener('change', () => { st.fcDetail = null; loadForecast(); });
    $('an-form').addEventListener('submit', (e) => { e.preventDefault(); loadAnomalies(); });
    ['an-threshold', 'an-window', 'an-reference', 'an-fdr'].forEach((id) => $(id).addEventListener('change', loadAnomalies));
    $('as-form').addEventListener('submit', (e) => { e.preventDefault(); st.loaded.aftershock = true; runAftershock(); });

    A.on('openAftershock', (id) => openAftershock(id));
    A.on('feed', () => { if (st.loaded.sequences) loadSequences(); });
    A.on('refresh', () => {
      if (st.view === 'forecast') loadForecast();
      if (st.view === 'anomaly') loadAnomalies();
    });
    A.on('mapClick', (lat, lon) => {
      if (st.view === 'forecast' && st.fc) {
        const id = regionAt(st.fc.regions, lat, lon);
        if (id) selectForecastRegion(id); else A.toast('No forecast for this cell: too little M4.5+ history in the training period.');
      } else if (st.view === 'anomaly' && st.an) {
        const id = regionAt(st.an.results, lat, lon);
        if (id) selectAnomalyRegion(id); else A.toast('This cell had no qualifying events in the reference or observation window.');
      }
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (window.L && A.state.map) {
      layer = L.layerGroup();
      A.state.layers.module = layer;
    }
    bind();
    const fromHash = () => { const v = location.hash.replace('#', ''); return VIEWS[v] ? v : 'live'; };
    window.addEventListener('hashchange', () => { if (fromHash() !== st.view) setView(fromHash()); });
    setView(fromHash());
  });
})();
