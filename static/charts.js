/*
 * TerraWatcher SVG charts: small, dependency-free, testable in Node.
 * Every chart is returned as an SVG string with labelled axes and units, an
 * accessible name (role="img" + <title>), and a viewBox so it scales to its card.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TerraCharts = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  /** Round tick values covering [min, max]. */
  function niceTicks(min, max, count = 5) {
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
    if (max === min) { max = min + 1; }
    const span = max - min;
    const step0 = span / Math.max(1, count);
    const mag = 10 ** Math.floor(Math.log10(step0));
    const err = step0 / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
    return out;
  }

  function logTicks(min, max) {
    const out = [];
    for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e += 1) {
      const v = 10 ** e;
      if (v >= min * 0.999 && v <= max * 1.001) out.push(v);
    }
    return out.length ? out : [min, max];
  }

  function scale(d0, d1, r0, r1, log = false) {
    if (log) {
      const a = Math.log10(d0), b = Math.log10(d1);
      return (v) => r0 + ((Math.log10(Math.max(v, d0)) - a) / ((b - a) || 1)) * (r1 - r0);
    }
    return (v) => r0 + ((v - d0) / ((d1 - d0) || 1)) * (r1 - r0);
  }

  function formatTime(ms, span) {
    const d = new Date(ms);
    if (span > 400 * 86400000) return `${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
    return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
  }

  function formatNum(v) {
    if (!Number.isFinite(v)) return '';
    const a = Math.abs(v);
    if (a >= 1000) return `${Math.round(v / 100) / 10}k`;
    if (a > 0 && a < 0.01) return v.toExponential(0);
    return String(Number(v.toPrecision(3)));
  }

  function frame(width, height, title, desc, body) {
    return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}" preserveAspectRatio="xMidYMid meet">` +
      `<title>${esc(title)}</title>${desc ? `<desc>${esc(desc)}</desc>` : ''}${body}</svg>`;
  }

  /**
   * Line/step/dot/bar series on numeric, time, year (integer ticks, no “k”) or logarithmic x axes, plus optional
   * interval bands and point markers with error bars.
   */
  function lineChart(opts) {
    const W = opts.width || 460, H = opts.height || 230;
    const m = { top: 14, right: 14, bottom: 42, left: 52, ...(opts.margin || {}) };
    const series = (opts.series || []).filter((s) => s.data && s.data.length);
    const bands = (opts.bands || []).filter((b) => b.data && b.data.length);
    const markers = opts.markers || [];
    const xs = [], ys = [];
    series.forEach((s) => s.data.forEach((p) => { xs.push(p.x); ys.push(p.y); }));
    bands.forEach((b) => b.data.forEach((p) => { xs.push(p.x); ys.push(p.lo, p.hi); }));
    markers.forEach((p) => { xs.push(p.x); ys.push(p.y, p.lo ?? p.y, p.hi ?? p.y); });
    const xType = (opts.x && opts.x.type) || 'linear';
    const finiteX = xs.filter((v) => Number.isFinite(v) && (xType !== 'log' || v > 0));
    const finiteY = ys.filter((v) => Number.isFinite(v) && (!opts.y?.log || v > 0));
    if (!finiteX.length) return `<p class="muted small chart-empty">${esc(opts.empty || 'No data to plot.')}</p>`;
    let x0 = opts.x?.min ?? Math.min(...finiteX), x1 = opts.x?.max ?? Math.max(...finiteX);
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    const yLog = !!opts.y?.log;
    let y0 = opts.y?.min ?? (yLog ? Math.min(...finiteY) : Math.min(0, ...finiteY));
    let y1 = opts.y?.max ?? Math.max(...finiteY, yLog ? y0 * 10 : 1);
    if (!yLog) y1 = y1 + (y1 - y0) * 0.06;
    const X = scale(x0, x1, m.left, W - m.right, xType === 'log');
    const Y = scale(y0, y1, H - m.bottom, m.top, yLog);
    const parts = [];

    // grid + axes
    const yt = yLog ? logTicks(y0, y1) : niceTicks(y0, y1, 4);
    yt.forEach((v) => {
      const y = Y(v).toFixed(1);
      parts.push(`<line class="grid" x1="${m.left}" x2="${W - m.right}" y1="${y}" y2="${y}"/>`);
      parts.push(`<text class="tick" x="${m.left - 6}" y="${y}" text-anchor="end" dominant-baseline="middle">${esc(formatNum(v))}</text>`);
    });
    let xt = xType === 'log' ? logTicks(x0, x1) : niceTicks(x0, x1, xType === 'time' ? 5 : 6);
    if (xType === 'year') xt = xt.filter(Number.isInteger);
    xt.forEach((v) => {
      const x = X(v).toFixed(1);
      const label = xType === 'time' ? formatTime(v, x1 - x0) : xType === 'year' ? String(v) : formatNum(v);
      parts.push(`<line class="axis" x1="${x}" x2="${x}" y1="${H - m.bottom}" y2="${H - m.bottom + 4}"/>`);
      parts.push(`<text class="tick" x="${x}" y="${H - m.bottom + 16}" text-anchor="middle">${esc(label)}</text>`);
    });
    parts.push(`<line class="axis" x1="${m.left}" x2="${W - m.right}" y1="${H - m.bottom}" y2="${H - m.bottom}"/>`);
    parts.push(`<line class="axis" x1="${m.left}" x2="${m.left}" y1="${m.top}" y2="${H - m.bottom}"/>`);
    if (opts.x?.label) parts.push(`<text class="axis-label" x="${(m.left + W - m.right) / 2}" y="${H - 6}" text-anchor="middle">${esc(opts.x.label)}</text>`);
    if (opts.y?.label) parts.push(`<text class="axis-label" transform="translate(13 ${(m.top + H - m.bottom) / 2}) rotate(-90)" text-anchor="middle">${esc(opts.y.label)}</text>`);

    bands.forEach((b) => {
      const pts = b.data.filter((p) => Number.isFinite(p.lo) && Number.isFinite(p.hi));
      if (!pts.length) return;
      const top = pts.map((p) => `${X(p.x).toFixed(1)},${Y(p.hi).toFixed(1)}`);
      const bot = pts.slice().reverse().map((p) => `${X(p.x).toFixed(1)},${Y(p.lo).toFixed(1)}`);
      parts.push(`<polygon class="band" style="fill:${b.color || 'currentColor'}" points="${top.concat(bot).join(' ')}"><title>${esc(b.name || 'interval')}</title></polygon>`);
    });

    series.forEach((s) => {
      const color = s.color || 'currentColor';
      const pts = s.data.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && (!yLog || p.y > 0) && (xType !== 'log' || p.x > 0));
      if (s.kind === 'bar') {
        const n = Math.max(pts.length, 1);
        const bw = Math.max(1, ((W - m.left - m.right) / n) * 0.8);
        pts.forEach((p) => {
          const y = Y(Math.max(p.y, y0));
          parts.push(`<rect class="bar" x="${(X(p.x) - bw / 2).toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(0, Y(y0) - y).toFixed(1)}" style="fill:${p.color || color}"><title>${esc(`${s.name}: ${formatNum(p.y)}`)}</title></rect>`);
        });
      } else if (s.kind === 'dot') {
        pts.forEach((p) => parts.push(`<circle class="dot" cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${p.r || 2.6}" style="fill:${p.color || color}"><title>${esc(p.title || `${s.name}: ${formatNum(p.y)}`)}</title></circle>`));
      } else {
        let d = '';
        pts.forEach((p, i) => {
          const x = X(p.x).toFixed(1), y = Y(p.y).toFixed(1);
          if (i === 0) d += `M${x},${y}`;
          else if (s.kind === 'step') d += `H${x}V${y}`;
          else d += `L${x},${y}`;
        });
        parts.push(`<path class="line" d="${d}" style="stroke:${color}${s.dash ? ';stroke-dasharray:5 4' : ''}"><title>${esc(s.name)}</title></path>`);
      }
    });

    markers.forEach((p) => {
      const x = X(p.x).toFixed(1);
      if (Number.isFinite(p.lo) && Number.isFinite(p.hi)) {
        parts.push(`<line class="errbar" x1="${x}" x2="${x}" y1="${Y(p.lo).toFixed(1)}" y2="${Y(p.hi).toFixed(1)}" style="stroke:${p.color || 'currentColor'}"/>`);
      }
      parts.push(`<circle class="marker" cx="${x}" cy="${Y(p.y).toFixed(1)}" r="4.5" style="fill:${p.color || 'currentColor'}"><title>${esc(p.label || '')}</title></circle>`);
    });

    const legend = [...series.filter((s) => s.name), ...bands.filter((b) => b.name), ...markers.filter((p) => p.legend)];
    const legendHtml = legend.length ? `<div class="chart-legend">${legend.map((s) =>
      `<span><i style="background:${s.color || 'currentColor'}${s.dash ? ';height:2px' : ''}"></i>${esc(s.name || s.legend)}</span>`).join('')}</div>` : '';
    return frame(W, H, opts.title || 'Chart', opts.desc, parts.join('')) + legendHtml;
  }

  /** Vertical histogram with category labels. bins: [{label, count}]. */
  function barChart(opts) {
    const bins = opts.bins || [];
    const W = opts.width || 400, H = opts.height || 190;
    const m = { top: 12, right: 10, bottom: 46, left: 44 };
    if (!bins.length || bins.every((b) => !b.count)) return `<p class="muted small chart-empty">${esc(opts.empty || 'No values to show.')}</p>`;
    const max = Math.max(...bins.map((b) => b.count));
    const yt = niceTicks(0, max, 4);
    const top = Math.max(max, yt[yt.length - 1] || 1);
    const Y = scale(0, top, H - m.bottom, m.top);
    const bw = (W - m.left - m.right) / bins.length;
    const parts = [];
    yt.forEach((v) => {
      parts.push(`<line class="grid" x1="${m.left}" x2="${W - m.right}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}"/>`);
      parts.push(`<text class="tick" x="${m.left - 6}" y="${Y(v).toFixed(1)}" text-anchor="end" dominant-baseline="middle">${esc(formatNum(v))}</text>`);
    });
    const every = Math.ceil(bins.length / 8);
    bins.forEach((b, i) => {
      const x = m.left + i * bw;
      const y = Y(b.count);
      parts.push(`<rect class="bar" x="${(x + bw * 0.1).toFixed(1)}" y="${y.toFixed(1)}" width="${(bw * 0.8).toFixed(1)}" height="${(Y(0) - y).toFixed(1)}" style="fill:${opts.color || 'currentColor'}"><title>${esc(`${b.label}: ${b.count}`)}</title></rect>`);
      if (i % every === 0) parts.push(`<text class="tick" x="${(x + bw / 2).toFixed(1)}" y="${H - m.bottom + 14}" text-anchor="middle">${esc(b.label)}</text>`);
    });
    parts.push(`<line class="axis" x1="${m.left}" x2="${W - m.right}" y1="${H - m.bottom}" y2="${H - m.bottom}"/>`);
    if (opts.xLabel) parts.push(`<text class="axis-label" x="${(m.left + W - m.right) / 2}" y="${H - 6}" text-anchor="middle">${esc(opts.xLabel)}</text>`);
    if (opts.yLabel) parts.push(`<text class="axis-label" transform="translate(12 ${(m.top + H - m.bottom) / 2}) rotate(-90)" text-anchor="middle">${esc(opts.yLabel)}</text>`);
    return frame(W, H, opts.title || 'Histogram', opts.desc, parts.join(''));
  }

  /** Convert [{from, to, count}] histogram rows into labelled bins. */
  function histogramBins(rows, digits = 1) {
    return (rows || []).map((r) => ({ label: `${Number(r.from).toFixed(digits)}–${Number(r.to).toFixed(digits)}`, count: r.count }));
  }

  return { niceTicks, logTicks, scale, lineChart, barChart, histogramBins, formatNum, formatTime };
});
