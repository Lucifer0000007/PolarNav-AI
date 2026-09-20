# Console / Streamlit parity

> **Superseded UI references, 2026-09-20**: the PASS/FAIL entries below that
> name a console tab (OPS/DETECT/FORECAST/ROUTE) were verified against the
> pre-redesign 6-tab layout. The "Mission Control redesign" mission
> collapsed those 4 into one MISSION tab's 4 stage cards (DATA/SYSTEM
> unchanged) — same endpoints, same underlying logic, just relocated. The
> functional PASS results still hold; only the tab names are stale. Left
> as historical record rather than rewritten, since these were genuine
> point-in-time verification runs.

`api.py` cannot import `app.py` (it runs `st.*` calls at module scope on
load, which breaks outside a real Streamlit run), and `app.py` is out of
scope to edit for this mission. So the handful of app.py helpers the
console needs are **ported verbatim into `api.py`**, not shared by import.
This doc names exactly which app.py line ranges each `api.py` function
mirrors — check both sides whenever either file changes.

| app.py source | Mirrored in `api.py` as |
|---|---|
| `_load_csv_safe` (app.py:213-225) | `_load_csv_safe` |
| F1 button body (app.py:231-308) | `_ingest()` |
| F2 button body (app.py:319-348) | `_detect()` — keeps app.py's own hardcoded `sar_path="data/sar_sample.png"` quirk (F2 never reads a live receipt's own `sar_path`) for true behavioral parity, not "fixed" |
| `_compute_route` (app.py:364-481ish) | `_compute_route()` |
| `_along_route_km` / `_km_between` | same names |
| `_dm_to_decimal` / `_parse_nmea_line` / `_nmea_probe` | same names |
| M4 status fragment (app.py's 4 dots) | `GET /status`'s `watcher`/`bus`/`nmea`/`model` fields |
| M5 stale banner | `GET /status`'s `drop.age_h`/`drop.stale` |

**v3 update**: `_concentration_rgba`/`_build_map` are no longer separately
duplicated in each file — both `app.py` and `api.py` now import the same
`mapview.py` (new, shared module) and call `mapview.build_map(rd,
leaflet_prefix=..., ...)`. The only real difference between the two
callers (the Leaflet static-mount prefix: `/app/static` for Streamlit,
`/static` for the console) is a parameter, not a second copy of the
logic. This removes the one place this file previously named as
duplicated-by-necessity.

## Parity matrix

| Streamlit feature | Where in the console | How verified | Result |
|---|---|---|---|
| F1 receipts + data quality | `/detect`'s `ingest` block, shown under the controls row | API-level: `curl -X POST /detect`, compare `messages`/`source` | **PASS** — static fallback shows the 3 success lines + NSIDC context line verbatim; live-receipt path shows "Live drop consumed: \<batch_id\> (validated ...)" verbatim |
| F2 mask + caption | `/detect`'s `detection` block | API-level: curl, compare `n_cells`/`coverage_pct`/`active_path` | **PASS** — `active_path` correctly reads "Active model: OpenCV Otsu (fallback)" (no U-Net weights present, as documented); `n_cells`/`coverage_pct` consistent across repeated calls with the same image |
| Drift arrows + predicted overlay | Baked into `GET /map`'s rendered HTML | Visual: open `http://127.0.0.1:8000/map` directly, compare to the Streamlit map | **USER-VERIFY** — API-level check confirmed the rendered HTML contains the green optimized route, red dashed direct path, iceberg markers, and the risk `ImageOverlay`, with zero CDN references; final visual side-by-side against the Streamlit map is for you to eyeball |
| F5 metrics + alerts + reroute delta | `/route` response → KPI row + alert cards | API-level: curl `/route`, diff metrics against an independent direct call to the same engine.py functions with the same inputs | **PASS** — exact field-for-field match confirmed twice (two different random ingests), see the Test Matrix's engine-number-match run in the mission report |
| Event ticker | `GET /events` → marquee | API-level: curl, compare to `events.log` tail | **PASS** — oldest-first, 20-line cap, real `route.computed`/`alerts.raised` entries confirmed; raw `events.log` bytes independently checked to rule out a display-only encoding artifact seen during testing |
| Topology dots | `GET /status` → SVG pulse classes | API-level: curl with sims on vs. off | **PASS** — `watcher`/`bus`/`model` all flip correctly; `nmea.live` flips too, though see the caveat below |
| History table | `GET /history` | API-level: curl, compare row count/fields to `routes.db` directly | **PASS** — schema and row count match `routes.db` exactly (direct passthrough of `load_routes()`) |
| Strict-JSON viewer + copy | `/route`'s `strict_json` field | API-level: curl, confirm exactly the 7 `strict_json` fields, no more/fewer | **PASS** — field set matches `engine.strict_json` exactly; copy-button behavior is a client-side clipboard call, **USER-VERIFY** in a real browser |
| LIVE POS/PINNED badge | `/status`'s `nmea.live` | API-level: curl with `nmea_sim.py` on/off | **PASS, with a caveat** — see below |
| Stale-drop banner | `/status`'s `drop.stale`/`age_h` | Same 12h-boundary-in-isolation technique used for app.py's M5 (backdating a live clock isn't practical) | **PASS** — confirmed fresh (`stale:false`) immediately after ingest; 12h threshold math verified in isolation (identical to M5's own verification approach) |

**Caveat on LIVE POS/PINNED**: `nmea_sim.py` accepts one TCP client at a time (`srv.listen(1)`, out of scope to change). With *both* a live Streamlit session (its own `_position_badge` fragment polling every 1s) and the console (`/status` polling every 2s) running against the same `nmea_sim.py` at once, the two pollers occasionally contend for that single slot, and either side's probe can intermittently read as "pinned" for one tick even while the feed is live. Reproduced directly: 1 of 6 rapid `/status` calls read `nmea.live:false` while `nmea_sim.py` was genuinely running, and the other 5 (including immediate retries) correctly read `true`. This is a pre-existing characteristic of `nmea_sim.py`'s single-client design, not a console defect — `_nmea_probe()` here is a byte-for-byte port of app.py's own function. It would equally affect two Streamlit browser tabs open at once against the same simulator.

## Known, deliberate non-parity

- `/detect` combines F1 (ingest) and F2 (detect) into one action; app.py
  has them as two separate buttons for a judge-facing, step-by-step demo.
  See `docs/API_CONTRACT.md`'s note on this.

(The v2-era "no image-preview panel" gap is resolved this mission — the
console's DETECT tab now shows both images via `/detect`'s
`orig_png_b64`/`mask_png_b64` fields, which app.py's F2 doesn't expose
since `st.image` renders the array directly with no need to encode it.)

## v3 completeness matrix (console-side; app.py's own gate is narrower — see below)

Every row is a `[SHOWN]`/`[WIRE]`/`[ORPHAN]` item from this mission's own
completeness audit; "panel" names the console tab, per `docs/API_CONTRACT.md`.

| Engine output | Console panel | Verified |
|---|---|---|
| SAR image vs. detected mask | DETECT | **PASS** — side-by-side, same frame treatment |
| Coverage % (both clients now — app.py gained this too, not just console) | DETECT + app.py F2 caption | **PASS** |
| Fallback reason (unet_disabled / weights_missing / unet_failed / guard_rejected / unet_accepted) | DETECT | **PASS** — replaces the old binary unet/otsu-only tag |
| Inference time | DETECT | **PASS** — measured in `api.py`, not engine.py, per the mission's own instruction |
| Otsu threshold + histogram | DETECT | **PASS** — histogram bars + a marker line at the threshold |
| Coverage-guard status | DETECT | **PASS** — same field as fallback reason |
| Risk-grid heatmap: current / predicted / pure-advection | FORECAST | **PASS** — 3 visually distinct images confirmed (predicted is max-floored by current, pure advection is not — see `engine.predict_risk_grid`'s docstring) |
| KMeans tier centroids + multipliers | FORECAST | **PASS** — legend strip under the current-grid heatmap |
| Per-iceberg drift table (id/mass/freeboard/now/+24h/vector km) | FORECAST | **PASS** |
| Ridge status (active + R² + provenance) | FORECAST | **PASS** — `r2: 0.9999...` read live from `drift_training_report.json` |
| Full route metrics table (all 11 `route_metrics` keys) | ROUTE | **PASS** |
| CPA-per-berg table, including LOW tier | ROUTE | **PASS** — previously only HIGH/MED got individual lines |
| Reroute delta + the alternate path's own geometry | ROUTE (delta) + map (geometry, dashed grey line) | **PASS** |
| Drop receipts + quarantine log | DATA | **PASS** |
| NSIDC badge + coverage-mapping detail | DATA | **PASS** |
| Rejected-row count | FORECAST | **PASS** — computed adapter-side from the same NaN check `build_risk_grid` uses internally |
| Model registry (weights present? Dice? INT8?) | SYSTEM | **PASS** — Dice/INT8 honestly reported as "not yet measured," not fabricated |
| Offline-proof (telemetry, tiles, zero-CDN, bind address) | SYSTEM | **PASS** |
| Commit hash | SYSTEM | **PASS** — `git rev-parse --short HEAD`, "unknown" on failure |
| Sim Deck (mode, counters, countdown, NMEA/bus chips, Drop Now) | OPS | **PASS** — start/stop idempotency confirmed twice in each direction |

**app.py's own gate is narrower by design**: per this mission's plan, the
completeness gate targets the console specifically (your own Step 1b
wording: "shown/missing **in console today**"). app.py only gained the
shared map builder (mandatory) plus `coverage_pct` and the
`use_container_width` map-sizing fix (both essentially free while already
touching those exact lines) — it was not given the 6-tab redesign or the
rest of the new data panels above, which remain console-only.
