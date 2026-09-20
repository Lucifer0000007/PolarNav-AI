// PolarNav AI Bridge Console -- vanilla JS, zero build chain, zero CDN.
// Talks only to same-origin API routes (api.py) -- see docs/API_CONTRACT.md.

const COORD_PRESETS = [[5, 5], [10, 15], [20, 20], [35, 35]]; // mirrors app.py's coords_list
let historyPage = 0;
const HISTORY_PAGE_SIZE = 15;
let lastHistoryRows = [];

function fmtCoord([r, c]) { return `(${r}, ${c})`; }
function esc(s) { const d = document.createElement("div"); d.textContent = String(s); return d.innerHTML; }
function fmt1(n) { return (n == null || Number.isNaN(n)) ? "—" : Number(n).toFixed(1); }

// ---- tabs ----
function switchView(name) {
  document.querySelectorAll(".rail-item").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach(v => { v.hidden = v.id !== `view-${name}`; });
  if (name === "detect") refreshDetectView();
  if (name === "forecast") refreshForecastView();
  if (name === "route") refreshRouteView();
  if (name === "data") refreshDataView();
  if (name === "system") refreshSystemView();
}
document.querySelectorAll(".rail-item").forEach(b => b.addEventListener("click", () => switchView(b.dataset.view)));

function populateSelects() {
  const start = document.getElementById("sel-start");
  const goal = document.getElementById("sel-goal");
  COORD_PRESETS.forEach(([r, c]) => {
    const o1 = document.createElement("option");
    o1.value = `${r},${c}`;
    o1.textContent = fmtCoord([r, c]);
    start.appendChild(o1);
    goal.appendChild(o1.cloneNode(true));
  });
  start.selectedIndex = 0;
  goal.selectedIndex = COORD_PRESETS.length - 1;
}
function parseSelected(sel) { return sel.value.split(",").map(Number); }

function tickClock() {
  document.getElementById("utc-clock").textContent = new Date().toISOString().slice(11, 19) + "Z";
}

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

function stateEmpty(msg, icon) {
  return `<div class="state-empty">${icon ? `<span class="state-icon">${icon}</span>` : ""}${esc(msg)}</div>`;
}
function stateError(msg, retryFn) {
  const id = "retry-" + Math.random().toString(36).slice(2, 8);
  window[id] = retryFn;
  return `<div class="state-error"><span class="state-icon">&#9888;</span>${esc(msg)}
    <div class="state-retry"><button class="btn btn-small" onclick="${id}()">Retry</button></div></div>`;
}
function skeleton(n) {
  const widths = ["w80", "w60", "w40"];
  let out = '<div class="skeleton">';
  for (let i = 0; i < n; i++) out += `<div class="skel-line ${widths[i % 3]}"></div>`;
  out += "</div>";
  return out;
}

// ---- hand-rolled JSON syntax highlight (no library) ----
function highlightJson(obj) {
  const json = JSON.stringify(obj, null, 2);
  return esc(json).replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+\.?\d*(?:[eE][+-]?\d+)?)/g,
    (match) => {
      let cls = "jv-num";
      if (/^"/.test(match)) cls = /:$/.test(match) ? "jv-key" : "jv-str";
      else if (/true|false/.test(match)) cls = "jv-bool";
      else if (/null/.test(match)) cls = "jv-null";
      return `<span class="${cls}">${match}</span>`;
    });
}

// ---- /status polling: SYSTEM topology dots, LIVE POS/PINNED badge, stale banner, model KPI ----
async function refreshStatus() {
  let s;
  try {
    s = await getJSON("/status");
  } catch (_) {
    return; // API not reachable this tick -- next poll tries again, never throws up the UI
  }

  setDot("dot-watcher", s.watcher.fresh);
  setDot("dot-bus", s.bus.mode === "kafka");
  setDot("dot-nmea", s.nmea.live);
  setDot("dot-satcom", s.satcom ? s.satcom.running : s.watcher.fresh);
  setDot("dot-app", true); // this page is running, so the app node is definitionally live
  setDot("dot-model", s.model.active_path !== "not yet detected");

  updateKpiText("kpi-model", s.model.active_path);

  const posBadge = document.getElementById("pos-badge");
  if (s.nmea.live) {
    posBadge.textContent = "LIVE POS";
    posBadge.className = "badge badge-ok";
  } else {
    posBadge.textContent = "PINNED";
    posBadge.className = "badge badge-dim";
  }

  const banner = document.getElementById("stale-banner");
  if (s.drop.stale) {
    banner.hidden = false;
    banner.textContent = `LAST DROP: ${Math.round(s.drop.age_h)}h ago — treat as advisory only`;
  } else {
    banner.hidden = true;
  }

  document.getElementById("sim-bus-mode").textContent = s.bus.mode;
  document.getElementById("sim-nmea-hz").textContent = s.nmea.live ? "1.0" : "—";
}

function setDot(id, live) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("live", !!live);
}
function updateKpiText(id, text) {
  const el = document.getElementById(id);
  if (el && el.textContent !== text) { el.textContent = text; flashArrive(id); }
}
function flashArrive(valueId) {
  const el = document.getElementById(valueId);
  if (!el) return;
  el.classList.remove("arrive"); void el.offsetWidth; el.classList.add("arrive");
  const card = el.closest(".kpi-card");
  if (card) { card.classList.add("is-live"); setTimeout(() => card.classList.remove("is-live"), 900); }
}

// ---- /events polling: ticker ----
async function refreshEvents() {
  let events;
  try { events = await getJSON("/events"); } catch (_) { return; }
  const track = document.getElementById("ticker-track");
  if (!events.length) { track.textContent = "waiting for events…"; return; }
  track.textContent = events.map(e => `[${e.topic}] ${JSON.stringify(e.payload)}`).join("     •     ");
}

// ---- /history (backs both the OPS-era cache and the ROUTE tab's paged table) ----
async function refreshHistoryData() {
  try { lastHistoryRows = await getJSON("/history"); } catch (_) { /* keep previous */ }
}

function renderHistoryPage(container) {
  if (!lastHistoryRows.length) { container.innerHTML = stateEmpty("No routes computed yet.", "&#8987;"); return; }
  const totalPages = Math.max(1, Math.ceil(lastHistoryRows.length / HISTORY_PAGE_SIZE));
  historyPage = Math.min(historyPage, totalPages - 1);
  const start = historyPage * HISTORY_PAGE_SIZE;
  const rows = lastHistoryRows.slice(start, start + HISTORY_PAGE_SIZE);
  container.innerHTML = `
    <div class="table-wrap"><table>
      <thead><tr><th>ts</th><th>start</th><th>goal</th><th class="num">distance_km</th><th class="num">risk_red</th></tr></thead>
      <tbody>${rows.map(r => `<tr><td>${esc(r.ts)}</td><td>${esc(JSON.stringify(r.start))}</td>
        <td>${esc(JSON.stringify(r.goal))}</td><td class="num">${fmt1(r.distance_km)}</td>
        <td class="num">${fmt1(r.risk_red)}%</td></tr>`).join("")}</tbody>
    </table></div>
    <div class="pager">
      <button class="btn btn-small" id="hist-prev" ${historyPage === 0 ? "disabled" : ""}>&larr; Prev</button>
      <span>Page ${historyPage + 1} of ${totalPages} (${lastHistoryRows.length} total)</span>
      <button class="btn btn-small" id="hist-next" ${historyPage >= totalPages - 1 ? "disabled" : ""}>Next &rarr;</button>
    </div>`;
  const prev = document.getElementById("hist-prev"), next = document.getElementById("hist-next");
  if (prev) prev.addEventListener("click", () => { historyPage--; renderHistoryPage(container); });
  if (next) next.addEventListener("click", () => { historyPage++; renderHistoryPage(container); });
}

// ---- Sim Deck ----
async function refreshSimDeck() {
  let s;
  try { s = await getJSON("/sims/status"); } catch (_) { return; }
  const chip = document.getElementById("sim-mode-chip");
  chip.textContent = s.mode;
  chip.classList.toggle("live", s.mode === "LIVE AUTO");
  document.getElementById("sim-count-inbox").textContent = s.inbox;
  document.getElementById("sim-count-done").textContent = s.done;
  document.getElementById("sim-count-quarantine").textContent = s.quarantine;
  document.getElementById("btn-sim-toggle").textContent = s.mode === "LIVE AUTO" ? "Stand down" : "Go live";
  document.getElementById("btn-sim-toggle").className = s.mode === "LIVE AUTO" ? "btn btn-danger" : "btn btn-ok";

  const ring = document.getElementById("sim-ring");
  if (s.mode === "LIVE AUTO" && s.next_pass_in_s != null) {
    ring.hidden = false;
    const circumference = 2 * Math.PI * 15;
    const frac = 1 - (s.next_pass_in_s / s.interval_s);
    const progress = document.getElementById("sim-ring-progress");
    progress.style.strokeDasharray = `${circumference}`;
    progress.style.strokeDashoffset = `${circumference * (1 - frac)}`;
  } else {
    ring.hidden = true;
  }

  const failures = Object.entries(s.sims).filter(([, v]) => v && v.ok === false);
  const hint = document.getElementById("sim-hint");
  if (window.__simLastAction && window.__simLastAction.length) {
    hint.hidden = false;
    hint.innerHTML = window.__simLastAction.map(([name, r]) =>
      `<div class="${r.ok ? "caption" : ""}" style="color:${r.ok ? "" : "var(--danger)"}">${esc(name)}: ${esc(r.detail)}</div>`).join("");
  } else {
    hint.hidden = true;
  }
}

async function onSimToggle() {
  const btn = document.getElementById("btn-sim-toggle");
  btn.disabled = true;
  try {
    const wantStop = btn.textContent === "Stand down";
    const result = await getJSON(wantStop ? "/sims/stop" : "/sims/start", { method: "POST" });
    window.__simLastAction = Object.entries(result);
  } catch (e) {
    window.__simLastAction = [["error", { ok: false, detail: e.message }]];
  } finally {
    btn.disabled = false;
    refreshSimDeck();
  }
}

async function onDropNow() {
  const btn = document.getElementById("btn-drop-now");
  btn.disabled = true;
  try {
    await getJSON("/drop/now", { method: "POST" });
    window.__simLastAction = [["drop now", { ok: true, detail: "delivered to drops_in/" }]];
  } catch (e) {
    window.__simLastAction = [["drop now", { ok: false, detail: e.message }]];
  } finally {
    btn.disabled = false;
    refreshSimDeck();
  }
}

// ---- Ingest & Detect ----
async function onDetect() {
  const btn = document.getElementById("btn-detect");
  const caption = document.getElementById("detect-caption");
  btn.disabled = true;
  caption.innerHTML = '<span class="spinner"></span> ingesting…';
  try {
    const r = await getJSON("/detect", { method: "POST" });
    if (r.ingest.error) {
      caption.textContent = r.ingest.error;
    } else if (r.detection && r.detection.error) {
      caption.textContent = r.detection.error;
    } else if (r.detection) {
      caption.textContent = `${r.ingest.source} — ${r.detection.n_cells} ice cells `
        + `(${r.detection.coverage_pct.toFixed(1)}% coverage, ${r.detection.inference_ms.toFixed(0)} ms)`;
    }
  } catch (e) {
    caption.textContent = "detect failed: " + e.message;
  } finally {
    btn.disabled = false;
    refreshStatus();
    if (!document.getElementById("view-detect").hidden) refreshDetectView();
    if (!document.getElementById("view-forecast").hidden) refreshForecastView();
    if (!document.getElementById("view-data").hidden) refreshDataView();
  }
}

// ---- Compute Route ----
async function onRoute() {
  const btn = document.getElementById("btn-route");
  const start = parseSelected(document.getElementById("sel-start"));
  const goal = parseSelected(document.getElementById("sel-goal"));
  btn.disabled = true;
  try {
    const r = await getJSON("/route", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ start, goal }),
    });
    renderRoute(r);
    document.getElementById("map-frame").src = "/map?t=" + Date.now();
    await refreshHistoryData();
    if (!document.getElementById("view-route").hidden) refreshRouteView();
  } catch (e) {
    document.getElementById("alerts").innerHTML = stateError("route failed: " + e.message, "onRoute");
  } finally {
    btn.disabled = false;
  }
}

function renderRoute(r) {
  updateKpiText("kpi-risk-reduction", r.metrics.risk_reduction_pct.toFixed(1) + "%");
  updateKpiText("kpi-distance", r.metrics.path_distance_km.toFixed(1) + " km");
  updateKpiText("kpi-cpa", r.min_cpa_km != null ? r.min_cpa_km.toFixed(1) + " km" : "—");

  const alertsEl = document.getElementById("alerts");
  alertsEl.innerHTML = r.alerts.map(a => `<div class="alert-card level-${a.level} arrive">${esc(a.text)}</div>`).join("");

  window.__lastRoute = r;
}

// ---- DETECT tab ----
async function refreshDetectView() {
  const el = document.getElementById("detect-view-body");
  const iv = window.__lastDetect;
  if (!iv) {
    try {
      const r = await getJSON("/detect", { method: "POST" });
      window.__lastDetect = r.detection && !r.detection.error ? r.detection : null;
    } catch (_) { /* keep null, fall through to empty state below */ }
  }
  renderDetectView();
}

function renderDetectView() {
  const el = document.getElementById("detect-view-body");
  const d = window.__lastDetect;
  if (!d) { el.innerHTML = stateEmpty("Click “Ingest & Detect” on the Ops tab first.", "&#128269;"); return; }
  const histMax = Math.max(1, ...d.histogram);
  const barW = 300 / d.histogram.length;
  const bars = d.histogram.map((v, i) =>
    `<rect x="${i * barW}" y="${80 - (v / histMax) * 76}" width="${barW - 1}" height="${(v / histMax) * 76}" fill="#7fd4ff"/>`).join("");
  const thresholdX = d.otsu_threshold != null ? (d.otsu_threshold / 255) * 300 : null;
  el.innerHTML = `
    <div class="dgrid">
      <div class="panel">
        <div class="panel-title">SAR vs. detected mask</div>
        <div class="img-compare">
          <figure><img src="data:image/png;base64,${d.orig_png_b64}" alt="Original SAR"><figcaption>Original SAR</figcaption></figure>
          <figure><img src="data:image/png;base64,${d.mask_png_b64}" alt="Detected mask"><figcaption>Detected ice mask</figcaption></figure>
        </div>
        <div class="status-grid" style="margin-top:12px">
          <div class="status-row"><span class="k">Ice cells</span><span class="v">${d.n_cells}</span></div>
          <div class="status-row"><span class="k">Coverage</span><span class="v">${d.coverage_pct.toFixed(1)}%</span></div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-title">Model diagnostics</div>
        <div class="status-card">
          <div class="status-row"><span class="k">Active model</span><span class="v">${esc(d.active_path)}</span></div>
          <div class="status-row"><span class="k">Fallback reason</span><span class="v">${esc(d.fallback_reason || "—")}</span></div>
          <div class="status-row"><span class="k">Guard status</span><span class="v">${esc(d.guard_status)}</span></div>
          <div class="status-row"><span class="k">Inference time</span><span class="v">${d.inference_ms.toFixed(1)} ms</span></div>
          <div class="status-row"><span class="k">Otsu threshold</span><span class="v">${d.otsu_threshold != null ? d.otsu_threshold.toFixed(0) : "n/a (SmallUNet path)"}</span></div>
        </div>
        <div class="histogram-wrap" style="margin-top:12px">
          <div class="caption">Pixel intensity histogram</div>
          <svg viewBox="0 0 300 84" preserveAspectRatio="none">${bars}${thresholdX != null ? `<line x1="${thresholdX}" y1="0" x2="${thresholdX}" y2="80" stroke="#ff5c5c" stroke-width="2"/>` : ""}</svg>
        </div>
      </div>
    </div>`;
}

// ---- FORECAST tab ----
async function refreshForecastView() {
  const el = document.getElementById("forecast-view-body");
  el.innerHTML = skeleton(4);
  let f;
  try { f = await getJSON("/forecast"); } catch (e) { el.innerHTML = stateError(e.message, "refreshForecastView"); return; }
  if (f.error) { el.innerHTML = stateError(f.error, "refreshForecastView"); return; }
  if (!f.bergs.length && !f.heatmaps.current) {
    el.innerHTML = stateEmpty("Click “Ingest & Detect” on the Ops tab to forecast drift.", "&#9925;");
    return;
  }
  const kmeansLegend = f.kmeans_diagnostics ? f.kmeans_diagnostics.tier_multipliers.map((m, i) =>
    `<span>Tier ${String.fromCharCode(65 + i)} ×${m.toFixed(1)} (centroid ${f.kmeans_diagnostics.centroids[i][0].toFixed(0)}kt / ${f.kmeans_diagnostics.centroids[i][1].toFixed(1)}m)</span>`).join("") : "";
  el.innerHTML = `
    <div class="panel">
      <div class="panel-title">Ridge drift model <span class="sub">${f.ridge.active ? "active" : "inactive — physics fallback"}</span></div>
      <div class="status-grid">
        <div class="status-row"><span class="k">Held-out R²</span><span class="v">${f.ridge.r2 != null ? f.ridge.r2.toFixed(4) : "not yet measured"}</span></div>
        <div class="status-row"><span class="k">Provenance</span><span class="v" style="font-family:inherit;font-size:0.8rem">${esc(f.ridge.provenance)}</span></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">24h ice concentration <span class="sub">rejected rows: ${f.dropped_count}</span></div>
      <div class="dgrid-3">
        ${["current", "predicted", "advected"].map(k => `
          <div class="heatmap-wrap">${f.heatmaps[k] || '<div class="state-empty">n/a</div>'}
            <div class="heatmap-label">${k === "current" ? "Current" : k === "predicted" ? "Predicted (24h, floored by current)" : "Pure advection (24h)"}</div>
          </div>`).join("")}
      </div>
      ${kmeansLegend ? `<div class="tier-legend">${kmeansLegend}</div>` : ""}
    </div>
    <div class="panel">
      <div class="panel-title">Per-iceberg drift</div>
      <div class="table-wrap"><table>
        <thead><tr><th>id</th><th class="num">mass_kt</th><th class="num">freeboard_m</th><th>now</th><th>+24h</th><th class="num">vector_km</th></tr></thead>
        <tbody>${f.bergs.map(b => `<tr><td>${b.id}</td><td class="num">${fmt1(b.mass_kt)}</td><td class="num">${fmt1(b.freeboard_m)}</td>
          <td>${b.now_ll.map(fmt1).join(", ")}</td><td>${b.plus24h_ll.map(fmt1).join(", ")}</td>
          <td class="num">${fmt1(b.vector_km)}</td></tr>`).join("")}</tbody>
      </table></div>
    </div>`;
}

// ---- ROUTE tab ----
function refreshRouteView() {
  const el = document.getElementById("route-view-body");
  const r = window.__lastRoute;
  if (!r) { el.innerHTML = stateEmpty("Compute a route on the Ops tab first.", "&#8987;"); return; }
  const m = r.metrics;
  const metricRows = [
    ["Distance (optimized)", fmt1(m.path_distance_km) + " km"], ["Distance (direct)", fmt1(m.direct_distance_km) + " km"],
    ["Risk score (optimized)", fmt1(m.path_risk_score)], ["Risk score (direct)", fmt1(m.direct_risk_score)],
    ["Ice crossings (optimized)", m.path_crossings], ["Ice crossings (direct)", m.direct_crossings],
    ["Risk reduction", fmt1(m.risk_reduction_pct) + "%"], ["Fuel/time trade-off", fmt1(m.fuel_penalty_pct) + "%"],
    ["Current exposure", fmt1(m.current_exposure)], ["Predicted exposure (24h)", fmt1(m.predicted_exposure)],
    ["Predicted risk>5 crossings", m.predicted_crossings],
  ];
  el.innerHTML = `
    <div class="dgrid">
      <div class="panel">
        <div class="panel-title">Route metrics</div>
        <div class="table-wrap"><table><tbody>${metricRows.map(([k, v]) => `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join("")}</tbody></table></div>
        ${r.reroute_delta_km != null ? `<div class="status-card" style="margin-top:12px">Suggested reroute: <b>+${fmt1(r.reroute_delta_km)} km</b> for lower predicted exposure (drawn on the map as a dashed grey line).</div>` : ""}
      </div>
      <div class="panel">
        <div class="panel-title">CPA by iceberg</div>
        <div class="table-wrap"><table>
          <thead><tr><th>id</th><th class="num">CPA (km)</th><th>tier</th><th>note</th></tr></thead>
          <tbody>${r.cpa_table.map(c => `<tr><td>${c.id}</td><td class="num">${fmt1(c.cpa_km)}</td><td>${c.tier}</td><td>${esc(c.note)}</td></tr>`).join("")}</tbody>
        </table></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Route history <button id="btn-refresh-hist" class="btn btn-small">Refresh</button></div>
      <div id="route-history-body"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Strict JSON (vessel-API-ready payload schema) <button id="btn-copy-json" class="btn btn-small">Copy</button></div>
      <pre class="json-viewer">${highlightJson(r.strict_json)}</pre>
    </div>`;
  renderHistoryPage(document.getElementById("route-history-body"));
  document.getElementById("btn-refresh-hist").addEventListener("click", async () => { await refreshHistoryData(); renderHistoryPage(document.getElementById("route-history-body")); });
  document.getElementById("btn-copy-json").addEventListener("click", () => {
    navigator.clipboard.writeText(JSON.stringify(r.strict_json, null, 2)).catch(() => {});
  });
}

// ---- DATA tab ----
async function refreshDataView() {
  const el = document.getElementById("data-view-body");
  el.innerHTML = skeleton(3);
  let receipts, nsidc;
  try {
    [receipts, nsidc] = await Promise.all([getJSON("/drop/receipts"), getJSON("/nsidc")]);
  } catch (e) {
    el.innerHTML = stateError(e.message, "refreshDataView");
    return;
  }
  const nsidcCard = nsidc.available
    ? `<div class="status-grid">
        <div class="status-row"><span class="k">NSIDC date</span><span class="v">${nsidc.row.year}-${String(nsidc.row.month).padStart(2, "0")}-${String(nsidc.row.day).padStart(2, "0")}</span></div>
        <div class="status-row"><span class="k">Extent</span><span class="v">${fmt1(nsidc.row.extent)} M km²</span></div>
        <div class="status-row"><span class="k">Coverage mapping</span><span class="v">${nsidc.coverage_fraction != null ? (nsidc.coverage_fraction * 100).toFixed(1) + "% → " + nsidc.blob_count + " ice features" : "—"}</span></div>
      </div>`
    : stateEmpty("No NSIDC sample yet — run Ingest & Detect.", "&#127484;");
  // DATA_LIST_CAP: /drop/receipts itself returns every batch ever validated/
  // quarantined (uncapped, by contract) -- a long-running Sim Deck session
  // can accumulate hundreds of these (confirmed: 189 in one test run), so
  // only the most recent DATA_LIST_CAP are rendered, same reasoning as the
  // Route tab's history pager -- the API stays complete, only display caps.
  const DATA_LIST_CAP = 15;
  const shownValidated = receipts.validated.slice(0, DATA_LIST_CAP);
  const shownQuarantined = receipts.quarantined.slice(0, DATA_LIST_CAP);
  el.innerHTML = `
    <div class="panel"><div class="panel-title">NSIDC sample</div>${nsidcCard}</div>
    <div class="panel">
      <div class="panel-title">Drop receipts <span class="sub">${receipts.validated_count} validated</span></div>
      ${receipts.validated.length ? `<div class="table-wrap"><table><thead><tr><th>batch</th><th>files</th></tr></thead>
        <tbody>${shownValidated.map(v => `<tr><td>${esc(v.batch_id)}</td><td>${esc(v.files.join(", "))}</td></tr>`).join("")}</tbody></table></div>
        ${receipts.validated.length > DATA_LIST_CAP ? `<div class="caption" style="margin-top:8px">showing ${DATA_LIST_CAP} most recent of ${receipts.validated.length}</div>` : ""}`
        : stateEmpty("No validated drops yet — start the Sim Deck on Ops and let drop_watcher run.", "&#128230;")}
    </div>
    <div class="panel">
      <div class="panel-title">Quarantine log <span class="sub">${receipts.quarantined_count} rejected</span></div>
      ${receipts.quarantined.length ? `<div class="alerts">${shownQuarantined.map(q =>
        `<div class="alert-card level-error"><b>${esc(q.batch_id)}</b><br>${esc(q.reason)}</div>`).join("")}</div>
        ${receipts.quarantined.length > DATA_LIST_CAP ? `<div class="caption" style="margin-top:8px">showing ${DATA_LIST_CAP} most recent of ${receipts.quarantined.length}</div>` : ""}`
        : stateEmpty("No quarantined batches.", "&#9989;")}
    </div>`;
}

// ---- SYSTEM tab ----
async function refreshSystemView() {
  const el = document.getElementById("system-view-body");
  el.innerHTML = skeleton(4);
  let sys, st;
  try {
    [sys, st] = await Promise.all([getJSON("/system"), getJSON("/status")]);
  } catch (e) {
    el.innerHTML = stateError(e.message, "refreshSystemView");
    return;
  }
  el.innerHTML = `
    <div class="dgrid">
      <div class="panel">
        <div class="panel-title">System topology</div>
        <svg viewBox="0 0 320 150" class="topology-svg">
          <line x1="30" y1="30" x2="120" y2="30" class="topo-edge"/><line x1="120" y1="30" x2="120" y2="75" class="topo-edge"/>
          <line x1="30" y1="120" x2="120" y2="75" class="topo-edge"/><line x1="120" y1="75" x2="210" y2="75" class="topo-edge"/>
          <line x1="210" y1="75" x2="290" y2="40" class="topo-edge"/><line x1="210" y1="75" x2="290" y2="110" class="topo-edge"/>
          <circle cx="30" cy="30" r="8" class="topo-dot ${st.satcom && st.satcom.running ? "live" : ""}"/><text x="30" y="16" class="topo-label">satcom</text>
          <circle cx="120" cy="75" r="8" class="topo-dot ${st.watcher.fresh ? "live" : ""}"/><text x="120" y="61" class="topo-label">watcher</text>
          <circle cx="30" cy="120" r="8" class="topo-dot ${st.nmea.live ? "live" : ""}"/><text x="30" y="140" class="topo-label">nmea</text>
          <circle cx="210" cy="75" r="8" class="topo-dot live"/><text x="210" y="61" class="topo-label">app</text>
          <circle cx="290" cy="40" r="8" class="topo-dot ${st.bus.mode === "kafka" ? "live" : ""}"/><text x="278" y="26" class="topo-label">bus</text>
          <circle cx="290" cy="110" r="8" class="topo-dot ${st.model.active_path !== "not yet detected" ? "live" : ""}"/><text x="270" y="130" class="topo-label">model</text>
        </svg>
      </div>
      <div class="panel">
        <div class="panel-title">Offline proof</div>
        <div class="status-grid">
          <div class="status-row"><span class="k">Telemetry</span><span class="v">${sys.telemetry_disabled ? "disabled" : "unknown"}</span></div>
          <div class="status-row"><span class="k">Map tiles</span><span class="v">${esc(sys.tiles)}</span></div>
          <div class="status-row"><span class="k">Asset audit</span><span class="v">zero CDN (grep-gated)</span></div>
          <div class="status-row"><span class="k">Bind address</span><span class="v">127.0.0.1 only</span></div>
        </div>
      </div>
    </div>
    <div class="dgrid" style="margin-top:16px">
      <div class="panel">
        <div class="panel-title">Model registry</div>
        <div class="status-grid">
          <div class="status-row"><span class="k">SmallUNet weights</span><span class="v">${sys.model_registry.unet_weights_present ? sys.model_registry.unet_weights_size_kb + " KB" : "not present"}</span></div>
          <div class="status-row"><span class="k">Dice bar</span><span class="v">${esc(sys.model_registry.dice_status)}</span></div>
          <div class="status-row"><span class="k">Quantization</span><span class="v">${esc(sys.model_registry.quantization)}</span></div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-title">Build</div>
        <div class="status-grid">
          <div class="status-row"><span class="k">Commit</span><span class="v">${esc(sys.commit)}</span></div>
          <div class="status-row"><span class="k">Resource budget</span><span class="v" style="font-family:inherit;font-size:0.8rem">${esc(sys.resource_budget)}</span></div>
        </div>
      </div>
    </div>`;
}

// ---- wiring ----
document.getElementById("btn-detect").addEventListener("click", onDetect);
document.getElementById("btn-route").addEventListener("click", onRoute);
document.getElementById("btn-sim-toggle").addEventListener("click", onSimToggle);
document.getElementById("btn-drop-now").addEventListener("click", onDropNow);

populateSelects();
tickClock();
setInterval(tickClock, 1000);
refreshStatus();
setInterval(refreshStatus, 2000);
refreshEvents();
setInterval(refreshEvents, 4000);
refreshHistoryData();
refreshSimDeck();
setInterval(refreshSimDeck, 2000);
