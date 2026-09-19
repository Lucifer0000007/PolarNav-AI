// PolarNav AI Bridge Console -- vanilla JS, zero build chain, zero CDN.
// Talks only to same-origin API routes (api.py) -- see docs/API_CONTRACT.md.

const COORD_PRESETS = [[5, 5], [10, 15], [20, 20], [35, 35]]; // mirrors app.py's coords_list

function fmtCoord([r, c]) { return `(${r}, ${c})`; }

function populateSelects() {
  const start = document.getElementById("sel-start");
  const goal = document.getElementById("sel-goal");
  COORD_PRESETS.forEach(([r, c], i) => {
    const o1 = document.createElement("option");
    o1.value = `${r},${c}`;
    o1.textContent = fmtCoord([r, c]);
    start.appendChild(o1);
    const o2 = o1.cloneNode(true);
    goal.appendChild(o2);
  });
  start.selectedIndex = 0;
  goal.selectedIndex = COORD_PRESETS.length - 1;
}

function parseSelected(sel) {
  return sel.value.split(",").map(Number);
}

function tickClock() {
  const el = document.getElementById("utc-clock");
  const now = new Date();
  el.textContent = now.toISOString().slice(11, 19) + "Z";
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

// ---- /status polling: topology dots, LIVE POS/PINNED badge, stale banner, model KPI ----
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
  setDot("dot-satcom", s.watcher.fresh); // satcom's own activity is only observable via watcher freshness
  setDot("dot-app", true); // this page is running, so the app node is definitionally live
  setDot("dot-model", s.model.active_path !== "not yet detected");

  document.getElementById("kpi-model").textContent = s.model.active_path;

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
}

function setDot(id, live) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle("live", !!live);
}

// ---- /events polling: ticker ----
async function refreshEvents() {
  let events;
  try {
    events = await getJSON("/events");
  } catch (_) {
    return;
  }
  const track = document.getElementById("ticker-track");
  if (!events.length) {
    track.textContent = "waiting for events…";
    return;
  }
  track.textContent = events
    .map(e => `[${e.topic}] ${JSON.stringify(e.payload)}`)
    .join("     •     ");
}

// ---- /history ----
const HISTORY_DISPLAY_LIMIT = 15; // /history itself returns every row (per the locked contract);
                                   // this is a display-only cap so the page stays a fixed length
                                   // however many routes routes.db accumulates over a long session.

async function refreshHistory() {
  let rows;
  try {
    rows = await getJSON("/history");
  } catch (_) {
    return;
  }
  const body = document.getElementById("history-body");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="5">no routes yet.</td></tr>`;
    return;
  }
  const shown = rows.slice(0, HISTORY_DISPLAY_LIMIT);
  body.innerHTML = shown.map(r => `
    <tr>
      <td>${r.ts}</td>
      <td>${JSON.stringify(r.start)}</td>
      <td>${JSON.stringify(r.goal)}</td>
      <td>${r.distance_km.toFixed(1)}</td>
      <td>${r.risk_red.toFixed(1)}%</td>
    </tr>`).join("");
  if (rows.length > HISTORY_DISPLAY_LIMIT) {
    body.innerHTML += `<tr><td colspan="5" class="caption">showing ${HISTORY_DISPLAY_LIMIT} most recent of ${rows.length} total (GET /history itself returns all of them).</td></tr>`;
  }
}

// ---- Ingest & Detect ----
async function onDetect() {
  const btn = document.getElementById("btn-detect");
  const caption = document.getElementById("detect-caption");
  btn.disabled = true;
  caption.textContent = "ingesting…";
  try {
    const r = await getJSON("/detect", { method: "POST" });
    if (r.detection && r.detection.error) {
      caption.textContent = r.detection.error;
    } else if (r.detection) {
      caption.textContent = `${r.detection.source} — ${r.detection.n_cells} ice cells `
        + `(${r.detection.coverage_pct.toFixed(1)}% coverage)`;
    }
    if (r.ingest.error) {
      caption.textContent = r.ingest.error;
    }
  } catch (e) {
    caption.textContent = "detect failed: " + e.message;
  } finally {
    btn.disabled = false;
    refreshStatus();
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
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start, goal }),
    });
    renderRoute(r);
    document.getElementById("map-frame").src = "/map?t=" + Date.now();
    refreshHistory();
  } catch (e) {
    document.getElementById("alerts").textContent = "route failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderRoute(r) {
  document.getElementById("kpi-risk-reduction").textContent = r.metrics.risk_reduction_pct.toFixed(1) + "%";
  document.getElementById("kpi-distance").textContent = r.metrics.path_distance_km.toFixed(1) + " km";
  document.getElementById("kpi-cpa").textContent = r.min_cpa_km != null ? r.min_cpa_km.toFixed(1) + " km" : "—";

  const alertsEl = document.getElementById("alerts");
  alertsEl.innerHTML = r.alerts.map(a =>
    `<div class="alert-card level-${a.level}">${escapeHtml(a.text)}</div>`
  ).join("");

  document.getElementById("json-viewer").textContent = JSON.stringify(r.strict_json, null, 2);
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function onCopyJson() {
  const text = document.getElementById("json-viewer").textContent;
  navigator.clipboard.writeText(text).catch(() => { /* clipboard unavailable -- non-fatal */ });
}

// ---- wiring ----
document.getElementById("btn-detect").addEventListener("click", onDetect);
document.getElementById("btn-route").addEventListener("click", onRoute);
document.getElementById("btn-copy-json").addEventListener("click", onCopyJson);

populateSelects();
tickClock();
setInterval(tickClock, 1000);
refreshStatus();
setInterval(refreshStatus, 2000);
refreshEvents();
setInterval(refreshEvents, 4000);
refreshHistory();
