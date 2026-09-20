# Console UI audit

Phase 1 walk of the console client (MISSION/DATA/SYSTEM, all 4 Mission
stages, fresh/empty DB, loading, LIVE AUTO, MANUAL, bus-down, corrupt-drop
quarantine, stale-drop banner) at 1366/1536/1920px. Live Playwright testing
against a running server — mocked states via request interception where
exercising the real backend would mean destructively touching shared demo
data (empty DB, stale banner, bus-down, quarantine); LIVE AUTO/MANUAL
toggled for real via `/sims/start`/`/sims/stop` and restored to LIVE AUTO
afterward. No edits made during this phase.

## Findings

| id | tab/panel | symptom | severity | root cause | fixed? |
|---|---|---|---|---|---|
| U1 | MISSION — Navigation map (Situation panel) | Map shows the AOI as a fragment floating in a larger dark canvas, with the bezel's concentric rings/radar sweep visibly spilling past the AOI's own edges into empty space above and below it — reads as disconnected arcs, not a cohesive instrument. Confirmed at all 3 widths (structural, not width-dependent — the iframe's own box ratio is fixed by CSS regardless of viewport). | H | `.map-instrument iframe` is inset at `top:6.9%;left:12%;width:76%;height:86.2%`, giving the iframe box an aspect ratio of ~1:1.97. The AOI's true Mercator aspect ratio at this latitude is ~1:1.74 (confirmed via `mapview.py`'s own docstring and direct measurement: `leaflet-container` box measured 469×928 = 1:1.98 while the AOI content within it visibly occupies a shorter, ~1:1.74 region). Leaflet's `fitBounds` correctly fits the AOI's width to the box, leaving a letterboxed gap top/bottom — the bezel SVG was drawn assuming a tighter match and now visibly extends into that gap. | **YES** — replaced `.map-instrument` with `.map-wrap{position:relative;height:540px;overflow:hidden}` + `iframe{position:absolute;inset:0;...}`, and moved `fit_bounds` in `mapview.py` to run *after* every overlay/marker is added (was before all of them). Bezel rings/compass/sweep removed (they only decorated the now-gone margin). Re-verified: `leaflet-container` now measures 618×538 inside a 620×540 wrap (99.7%/99.6% fill) at all 3 widths — see `reaudit_map_iframe.png`. |
| U2 | MISSION — Navigation map | Legend renders as a separate block below the map instrument, not as an in-frame overlay. Confirmed via bounding-box containment check (legend top always below map-instrument bottom) at all 3 widths. | M | `.legend` is a DOM sibling *after* `#map-instrument` inside `.map-panel`, laid out in normal flow — not absolutely positioned inside the map wrapper. | **YES** — legend moved inside `.map-wrap` as `.map-legend`, absolutely positioned bottom-right, translucent "glass" background. Re-verified: bounding-box containment check returns true at all 3 widths. |
| U3 | MISSION stage 4 — CPA by iceberg table | No cap on the CPA table's growth — panel/card height is fully at the mercy of row count. Not yet visually broken with today's ~10-row demo dataset, but the containment is structurally absent. | M | `.table-wrap` (the CPA table's wrapper) has no `max-height`/`overflow-y` anywhere in `style.css` — only `overflow-x:auto` exists, for horizontal scroll of wide rows, not vertical capping. | **YES** — added a `cpa-table` class to that one `.table-wrap` (in `app.js`'s `refreshRouteView()`) and `.cpa-table{max-height:320px;overflow-y:auto}`. Re-verified via computed style: `max-height:320px`, `overflow-y:auto`. |
| U4 | MISSION (Live alerts) + DATA (Quarantine log) | A single alert/reason string containing one long token with **no spaces** (a plausible real batch-id, hash, or identifier) overflows its `.alert-card` horizontally **and** widens the entire page past the viewport. Confirmed directly: a 145-char unbroken token produced `alert-card scrollWidth 1157 vs clientWidth 647`, and `body.scrollWidth` went to 1843px at a 1366px viewport (a real page-wide layout break, not just local clipping). Normal prose with spaces wraps fine already — this only triggers on a genuinely unbroken run of characters, which the demo's own `batch_id` format (hyphen/underscore-segmented) doesn't currently produce, but nothing in the code guarantees that. | H | `.alert-card` sets no `overflow-wrap`/`white-space` (defaults to `overflow-wrap:normal`, which does not break within a word), and its container `.alerts` is `display:flex;flex-direction:column` with no `min-width:0` on the card — same "flex/grid item defaults to a content-based minimum, ignoring the parent's actual width" class of bug fixed elsewhere in this codebase for the ticker and mission-grid, not yet applied here. | **YES** — `.alert-card{white-space:normal;overflow-wrap:anywhere;min-width:0}` (the `min-width:0` is the necessary companion fix: `overflow-wrap` alone doesn't help while the flex item can still grow to fit unbroken content). Re-verified with the same 145-char token: `scrollWidth === clientWidth` (628=628, no overflow) and `body.scrollWidth` stayed at exactly 1366px. |
| U5 | MISSION stage 4 — Live alerts | The info-level "Predicted risk>5 cells on route: N" card renders unconditionally, including when N=0 — confirmed live (`_compute_route` with `predicted_crossings=0` still produced this exact card). Adds noise for the common case instead of only surfacing when actionable. | M | `api.py`'s `_compute_route()` appends this alert unconditionally after the per-iceberg loop, with no `if pred_x > 0:` guard (unlike the MED icebreaking-crossings line just above it, which *is* already conditional on `pred_x > 0`). | **YES, client-side** — the specified fix lives in `api.py`'s alert-building logic, but this mission's scope lock is explicit ("DO NOT touch ... api.py endpoint logic"), and `_compute_route()` is exactly that. Implemented the same outcome in `app.js` instead: `mergeCrossingsAlert()` drops the card entirely at N=0 and folds it as a detail into the matching MED card at N>0, applied in `renderRoute()` before the alerts list is rendered. Re-verified: N=0 → no info card; N=1 → `"MED: Route crosses 1 predicted risk>5 cells - expect icebreaking (predicted risk>5 cells on route: 1)"`, no separate card. Flagging this substitution plainly rather than silently deviating from the literal spec. |
| U6 | MISSION — any locked stage card (2/3/4 before their gate opens) | `.caption` and empty-state placeholder text inside a locked card is under the WCAG AA 4.5:1 text-contrast minimum. Computed directly from the actual token values: `.stage-card.is-locked{opacity:.55}` composites `--text-dim` (`#93a2bd`) against the card's own faded panel background down to **2.89:1** — a real, confirmed failure, not a near-miss. (Full-brightness `--text` on the same faded background still clears AA at 5.32:1, so only `--text-dim`-colored content inside a locked card is affected: stage captions and the "Complete Ingest, then..." placeholders.) Not one of the pre-identified five — found during the systematic walk. | M | `.stage-card.is-locked{opacity:0.55}` fades the *entire* card (background included) as one compositing group, so `--text-dim`'s already-modest contrast against `--panel` gets compounded by the same opacity reduction instead of staying legible while only the chrome dims. | **YES** — raised to `opacity:0.8`, computed to land at 4.69:1 (comfortable margin above the 4.5:1 minimum) while still reading as visibly dimmed/locked. |

Additional fix applied per the mission's own spec, independent of the audit table (fix (c)): the event stream (ticker) rendered one giant `[topic] {JSON.stringify(payload)}` line, which showed escaped backslashes for every Windows path in a payload (e.g. `drops_done\\...\\sar.png`) and had no timestamp. Rewrote as discrete `"HH:MM:SS · topic · short payload"` chips (`.event-chip`), with path-like values reduced to their basename via a small `basename()`/`summarizePayload()` helper. Applied the same helper to SYSTEM's Event log table (same underlying defect, same fix, for consistency — not a separately-numbered audit item since it's the same root cause fix(c) already targets).

## Confirmed clean (no defect — recorded so the walk's coverage is auditable)

| area | check | result |
|---|---|---|
| All 3 widths, all tabs | Absolute-positioned element with no positioned ancestor (DOM-walked every element, checked full ancestor chain) | None found |
| All 3 widths, full walk | Non-localhost network request (every `request` event logged) | None — localhost only |
| Normal operation, full walk | Browser console/page errors | None |
| Simulated `/status` failure (aborted route) | App-level graceful degradation | `refreshStatus()`'s own try/catch swallows it; `pos-badge` safely shows "PINNED" rather than crashing. (Playwright logs `net::ERR_FAILED` for the *intentionally* aborted request itself — that's my test harness reporting the fault I injected, not an application defect, and doesn't count against the "zero console errors" gate for normal operation.) |
| MISSION ↔ DATA/SYSTEM, all widths | `.situation` (sticky) overlapping `.pipeline` | Bounding boxes never intersect at 1366/1536/1920 |
| DATA — Drop receipts table | 300-char unbroken cell content | `.table-wrap` correctly scrolls internally (scrollWidth 2478 vs clientWidth 1124); `body.scrollWidth` stayed exactly at the 1366px viewport — tables are already properly contained, unlike alert-cards (U4) |
| Fresh/empty DB (mocked `/history`, `/drop/receipts`, `/nsidc`, `/forecast`) | Every DATA panel's empty-state copy | Renders sensibly, no crashes: "No NSIDC sample yet", "No validated drops yet", "No quarantined batches", "0 row(s) dropped...", "No routes computed yet." |
| Stale-drop banner (mocked `age_h:37.4`) | Banner visibility + wording | Shows correctly: "LAST DROP: 37h ago — treat as advisory only" |
| MANUAL mode (real toggle via `/sims/stop`) | Sim strip layout, ring visibility | No overflow; progress ring correctly hidden (LIVE AUTO-only element) |
| DATA/SYSTEM slow-load | Skeleton loader | Renders during the delay as designed |
| Text/background contrast, unfaded | `--text`/`--text-dim`/`--accent`/`--ok`/`--warn`/`--danger` against `--bg`/`--panel`/`--panel-2` | 5.03:1 to 16.07:1 — all clear WCAG AA everywhere except the one faded case (U6) |

Phase 1 complete — 6 findings (2 H, 4 M, 0 L).

## Phase 3 re-audit

All 6 findings fixed and re-verified live against a freshly restarted
server (fresh gating state, full Ingest→Detect→Forecast→Route walk) at
all 3 required widths (1366/1536/1920):

- **Zero open H/M items.**
- `body.scrollWidth` equals the viewport width exactly at all 3 widths
  with the pipeline fully populated (no regression from any fix).
- Zero non-localhost network requests, zero console/page errors across
  the full walk.
- The "confirmed clean" areas from Phase 1 were spot-rechecked (sticky
  non-overlap, wide table-cell containment) and remain clean — no fix in
  this pass touched their code paths.
- `mapview.py`'s `build_map()` is shared with `app.py` (Streamlit) — the
  fit_bounds reorder was verified directly (`fitBounds` now appears after
  `L.rectangle`/`L.imageOverlay` in the rendered HTML's source order, with
  a realistic Stage-4 `rd` dict) and Streamlit was restarted and confirmed
  healthy: zero errors, F7 route-history persists across a hard refresh.

GO.
