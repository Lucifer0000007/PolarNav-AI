# Bridge Console API contract

`api.py` is a thin FastAPI adapter in front of `engine.py`'s existing
functions. It introduces no new math — every number below comes from the
same functions `app.py` already calls. Binds `127.0.0.1:8000` only (see
`console.bat`). Interactive OpenAPI docs are available at `/docs` while
the server is running.

## Endpoints

| Method + path | Request | Response | Engine function(s) called |
|---|---|---|---|
| `GET /` | – | `console/index.html` | – |
| `GET /style.css`, `GET /app.js` | – | the static console files | – |
| `GET /healthz` | – | `{"status": "ok"}` | – |
| `GET /status` | – | see below | mirrors the watcher-mtime, `bus.transport_mode()`, NMEA-probe, and drop-age checks app.py's M4/M5 already do |
| `POST /detect` | *(empty body)* | see below | `resolve_sar_path`, `detect_ice`, `build_risk_grid` |
| `POST /route` | `{"start":[r,c], "goal":[r,c]}` | see below | `predict_iceberg_drift`, `build_drift_field`, `predict_risk_grid`, `kmeans_band_edges`, `astar`, `direct_path`, `route_metrics`, `cpa_km`, `classify_threat`, `suggest_reroute`, `save_route`, `strict_json` |
| `GET /map` | – | `text/html`, a full folium document for the map `<iframe>` | `grid_to_latlon` |
| `GET /events` | – | last 20 events, oldest-first | `bus.tail(20)` |
| `GET /history` | – | all saved routes, newest-first | `load_routes()` |

### `GET /status`

```json
{
  "watcher": {"fresh": true, "age_s": 4.2},
  "bus": {"mode": "file"},
  "nmea": {"live": false},
  "model": {"active_path": "Active model: OpenCV Otsu (fallback)"},
  "drop": {"age_h": 0.1, "stale": false}
}
```
`watcher.fresh` mirrors app.py's M4 dot (`drops_done/latest_receipt.json`
mtime < 60s). `drop.stale` mirrors app.py's M5 banner (`age_h > 12`).

### `POST /detect`

Runs the same ingest-then-detect sequence F1+F2 perform in app.py: check
for a live `drop_watcher.py` receipt, else fall back to the static
`data/` files, then run ice detection on whatever was ingested.

```json
{
  "ingest": {
    "source": "static",
    "batch_id": null,
    "messages": [{"level": "success", "text": "data/icebergs.csv ingested successfully."}],
    "seaice_context": {"year": 1988, "month": 11, "day": 8, "extent": 17.0},
    "sar_path": "data/sar_sample.png"
  },
  "detection": {
    "n_cells": 812,
    "coverage_pct": 5.1,
    "source": "Source: synthetic sample",
    "active_path": "Active model: OpenCV Otsu (fallback)"
  }
}
```

**Note on scope**: the locked endpoint list has no separate ingest route,
and a real console has no "simulate a drop" button — drops arrive from
`satcom_sim.py`/`drop_watcher.py` on their own. `/detect` therefore
combines F1's ingest-resolution and F2's detection into one action.

### `POST /route`

```json
{
  "start_ll": [-67.86, 59.64],
  "goal_ll": [-67.03, 60.96],
  "metrics": {"path_distance_km": 162.7, "risk_reduction_pct": 71.7, "...": "..."},
  "min_cpa_km": 20.3,
  "alerts": [{"level": "error", "text": "HIGH: 1 will cross your route in 25.2 km — suggested deviation +1.6 km"}],
  "notes": [],
  "reroute_delta_km": 1.6,
  "strict_json": {
    "status": "OK", "mode": "OFFLINE",
    "route_waypoints": [[-67.86, 59.64], "..."],
    "total_distance_km": 162.7, "risk_exposure": 0.35,
    "drift_predictions": [["..."]],
    "generated_at": "2026-09-20T12:00:00Z"
  }
}
```
Returns `409` if `/detect` hasn't been called yet this process (no risk
grid to route against). `start`/`goal` are grid cells `(r, c)` — the same
representation `engine.astar` takes natively and the same 4-preset
convention (`(5,5)`, `(10,15)`, `(20,20)`, `(35,35)`) app.py's dropdowns
use; there is no lat/lon conversion at this boundary.

`strict_json`'s `route_waypoints`/`total_distance_km`/`risk_exposure`/
`drift_predictions`/`generated_at` are exactly `engine.strict_json`'s own
7 fields (`status`, `mode` included) — nothing added, nothing renamed.
Note that `strict_json` itself never echoes `start`/`goal` even though it
accepts them as parameters (verified by reading `engine.py`), which is
why the outer `/route` response carries `start_ll`/`goal_ll` alongside it
instead of expecting them inside `strict_json`.
