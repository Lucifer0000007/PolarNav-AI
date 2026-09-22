# 🧊 PolarNav AI

**Offline-first navigation decision support for Antarctic research vessels.**

SAR ice detection → 24-hour iceberg drift prediction → risk-aware A* routing.
No internet. No API keys. No cloud. Runs on a laptop in the Southern Ocean.

`SIH26059` · TEKATHON-5.0 (2026) · **Team Zero Output**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Offline](https://img.shields.io/badge/network-zero%20calls-success)
![Stack](https://img.shields.io/badge/stack-Streamlit%20%2B%20FastAPI%20%2B%20OpenCV%20%2B%20PyTorch*%20%2B%20scikit--learn*-orange)

\* see Model status below. Two frontends ship on the same engine — see
[Two frontends, one engine](#two-frontends-one-engine).

---

## Model status (read first)

PyTorch is optional — the SmallUNet path activates only once trained weights
clear the accuracy bar (none yet: Otsu is active). scikit-learn is live: the
Ridge drift model in `models/drift_model.joblib` and on-the-fly KMeans
(iceberg risk tiers, overlay legend bands). See TRAINING_REPORT.md.

---

## The problem

NCPOR vessels operating around Antarctica lose reliable connectivity for days at a
time. Satellite bandwidth is metered, latency is brutal, and cloud routing services
are simply unreachable below the ice edge — exactly where iceberg collision risk is
highest. A bridge officer needs a route *now*, from data already on board.

**PolarNav AI runs the entire pipeline locally.** Drop a Sentinel-1 crop into a
folder and you get a risk-scored route in seconds, with zero packets leaving the
machine.

---

## What it does

| # | Capability | Implementation |
|---|---|---|
| **F1** | Satellite data drop | Ingests SAR imagery + iceberg coords + wind/current CSVs |
| **F2** | Ice hazard detection | OpenCV Otsu (default): Gaussian blur → threshold → morphological opening. Optionally a trained SmallUNet (PyTorch, architecture matched to `sea-ice-segmentation-u-net.ipynb`) when `models/unet_weights.pth` is present, with an automatic fallback to Otsu on a missing/undertrained model |
| **F3** | Risk grid | 40×40 cells, 0–10 risk, iceberg positions stamped as hard hazards |
| **F4** | 24-h drift prediction | Vector kinematics — current + 3% wind forcing (scikit-learn Ridge model in `models/drift_model.joblib`, trained on documented synthetic physics samples, with the formula as per-row fallback). The same field advects the risk grid into a 24-h **predicted concentration** overlay (max-combined; A* plans on max(current, predicted)) |
| **F5** | Risk-aware routing | Modified 8-directional A*, green optimal vs. red direct baseline |
| **F6** | Live metrics | Distance, risk score, ice crossings, risk reduction %, fuel penalty %; current vs 24-h predicted exposure; display-only **Live Alerts** (per-iceberg CPA → HIGH/MED/LOW, reroute suggestion as text only) |
| **F7** | Route history | Local SQLite (WAL mode), persists with no server |
| **F8** | Fully offline | No keys, no CDN, no non-loopback sockets — tiles=None basemap; Leaflet 1.9.3 vendored in `static/`, served by whichever frontend is running (Streamlit's `server.enableStaticServing`, or `api.py`'s own `/static/` mount for the Bridge Console), so the map renders with Wi-Fi off either way |
| **F9** | Vessel-API-ready schema | Strict JSON payload structured for NCPOR shipboard systems (transport = Phase 2) |

---

## Two frontends, one engine

`engine.py` has no UI code in it at all, so two independent frontends run
on top of the exact same routing/detection/drift math — pick whichever
fits the moment.

| | **Streamlit dashboard** | **Bridge Console** |
|---|---|---|
| Entry point | `app.py` | `api.py` (FastAPI) + `console/` (vanilla JS, zero build step) |
| Run it | `streamlit run app.py` | `console.bat`, or `python -m uvicorn api:app --host 127.0.0.1 --port 8000` |
| URL | `http://localhost:8501` | `http://127.0.0.1:8000` |
| Shape | One long page, top to bottom | 3 tabs — **MISSION** (a 4-stage gated pipeline: Ingest → Detect → Forecast → Route, each stage unlocking the next, next to a sticky live map), **DATA** (drop/quarantine archive, route history), **SYSTEM** (topology, model registry, offline proof, full event log) |
| Best for | A single operator working through the pipeline once, top to bottom | Demoing the realtime/offline story — a bridge console that can run unattended in **LIVE AUTO** and react to its own simulated sensor feed |

Both read and write the **same** `routes.db` (SQLite) and `events.log` —
a route computed in one shows up in the other's history immediately.
`api.py` never imports `app.py` (Streamlit executes UI calls at import
time, which breaks outside a real Streamlit run); the handful of helpers
the console needs are ported verbatim instead, and `docs/CONSOLE_PARITY.md`
tracks exactly which `app.py` lines each one mirrors, checked line-for-line
whenever either side changes. The map itself (`mapview.py`) is one shared
module both frontends call directly, not a duplicate.

**Realtime simulation layer** (Bridge Console only, though the Streamlit
dashboard also reads the same live GPS feed for its own auto-replan):
`satcom_sim.py` drops a pre-built data package into `drops_in/` on a
timer, `drop_watcher.py` validates it (checksum + schema) and quarantines
anything malformed, `nmea_sim.py` streams live `$GPGGA`/`$GPRMC` sentences
over a loopback TCP socket, and `bus.py` tails every event to `events.log`
(with an optional Kafka path if a broker happens to be reachable — file
tail always runs regardless). The console's **Sim Deck** strip starts and
stops this whole chain with one click; in **LIVE AUTO**, a validated drop
auto-unlocks the Ingest stage and a >2 km GPS drift auto-recomputes the
route, with no manual click — mirroring the drift-triggered replan
`app.py`'s own `_position_badge()` fragment already does. None of this
is imported by `engine.py`, `app.py`, or `api.py`'s core routes — it's a
separate, optional process tree, off by default (**MANUAL** mode).

---

## Proof it works

The demo scenario is pinned (DEMO_SEED = 13) so results are reproducible on any
machine. Routing from grid (5,5) to (35,35):

| Metric | Direct route | PolarNav A* | Result |
|---|---|---|---|
| Ice crossings (risk > 5) | **4** | **1** | 3 hazards avoided |
| Mean risk exposure | 1.234 | 0.325 | **↓ 73.7% risk** |
| Distance | 117.7 km | 129.1 km | +9.7% fuel |

**Trading 9.7% fuel for 73.7% less ice exposure** — the tradeoff a master actually
wants to make. Verify it yourself: `python engine.py` prints a PASS/FAIL scenario
check.

---

## How it works

```
  SAR image (real or synthetic)
          │
          ▼
  ┌───────────────────┐
  │ detect_ice()      │  blur(7x7) -> Otsu -> open(5x5)
  └───────────────────┘
          │ binary mask
          ▼
  ┌───────────────────┐   <- icebergs.csv stamped in as risk=10
  │ build_risk_grid() │      resize to 40x40, blur, clip 0..10
  └───────────────────┘
          │ risk[40,40]
          ├──────────────────────────┐
          ▼                          ▼
  ┌───────────────────┐    ┌───────────────────┐
  │ astar()           │    │ direct_path()     │  straight-line baseline
  │ f = g + h + 50*r  │    └───────────────────┘
  └───────────────────┘             │
          │                         │
          └────────┬────────────────┘
                   ▼
          ┌───────────────────┐
          │ route_metrics()   │ -> Folium map + strict JSON + SQLite
          └───────────────────┘
```

**Cost function.** Each A* step costs Euclidean distance (1 or √2) plus
`risk_weight × (risk/10)`, with `risk_weight = 50`. That weighting makes a single
high-risk cell roughly as expensive as 50 cells of open water — so the planner
detours hard around ice rather than shaving distance.

**Drift model.** Icebergs advect with ocean current plus ~3% of wind speed, a
standard freeboard-driven approximation:

```
u_total   = u_current + 0.03 × u_wind
Δposition = u_total × 24h × 3600s,  converted at 1° ≈ 111 km
```

**Grid.** 40×40 cells over lat [-68.0, -67.0], lon [59.5, 61.0] — about
**2.775 km per cell**. `latlon_to_grid()` clamps out-of-bounds coordinates to the
nearest edge cell, so a malformed CSV row never produces a negative index.

---

## Quick start

```bash
git clone https://github.com/Lucifer0000007/PolarNav-AI.git
cd PolarNav-AI

python -m venv venv
venv\Scripts\activate                 # Windows
# source venv/bin/activate            # macOS / Linux

pip install -r requirements.txt

# Optional ML core (SIH26059 AI/ML requirement) - CPU-only, installed
# separately since it needs the CPU wheel index:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# (see setup_demo.bat for a one-shot offline-prep install of everything)
```

Run the offline self-test (no browser, validates the whole engine):

```bash
python engine.py
```

Launch the Streamlit dashboard:

```bash
streamlit run app.py        # -> http://localhost:8501
```

On Windows, `demo.bat` does both steps in one double-click.

Or launch the Bridge Console instead (same engine, different UI — see
[Two frontends, one engine](#two-frontends-one-engine)):

```bash
pip install -r requirements-api.txt   # additive: just fastapi/uvicorn/pydantic
python -m uvicorn api:app --host 127.0.0.1 --port 8000   # -> http://127.0.0.1:8000
```

On Windows, `console.bat` does this in one double-click. Both frontends
can run at the same time (different ports) and share the same
`routes.db`/`events.log`.

### Demo flow

1. **📡 Simulate Satellite Data Drop** — ingests CSVs; samples a random Antarctic
   (south) row from `seaice.csv` (real NSIDC extent + date) and generates a
   synthetic SAR patch whose ice density reflects that real historical extent
2. **🔍 Detect Ice (U-Net / OpenCV Surrogate)** — shows source imagery beside
   the detected mask
3. **⚓ Compute Risk-Aware Route (Modified A\* + p(n))** — drift arrows, both
   routes, live metrics, a human-in-the-loop caption, and a provenance
   caption (data source, model architecture, active inference path)
4. Expand **📋 Strict JSON** and scroll to **route history**

---

## Using real Sentinel-1 imagery

The app ships with a synthetic SAR generator so it runs on a clean checkout. To use
real data, drop a grayscale crop at either path:

```
data/sar_real.png
data/sar_real.tif
```

`resolve_sar_path()` picks it up automatically — no flags, no config. The UI caption
switches from *Source: synthetic sample* to *Source: real Sentinel-1 crop*, so it is
always clear which image produced the mask. Remove the file and the synthetic
fallback returns.

The target footprint is described in `data/sar_metadata.json` (Sentinel-1 IW, VV
polarization, 40 m resolution, bbox [-68.0, 59.5, -67.0, 61.0]).

---

## Physics-informed synthetic SAR

When no real crop is present, the synthetic SAR patch isn't just random noise —
its ice density is driven by a real historical measurement. Each **Simulate
Satellite Data Drop** samples one random Antarctic (`hemisphere == 'south'`)
row from `seaice.csv` (the NSIDC Sea Ice Index) and scales the generated
ice-blob count to that row's real extent (2.0 M km² → sparse, 19.0 M km² →
dense). The UI shows exactly which historical day drove the patch, e.g.
`Context: 1990-10-20 | Extent: 18.0 M km² (NSIDC)`.

## Notebook-synced SmallUNet + training pipeline

`SmallUNet` in `engine.py` mirrors the architecture in
`sea-ice-segmentation-u-net.ipynb` layer-for-layer: the same 4-level encoder
(32→64→128→256 filters), 512-channel bottleneck, `Dropout(0.5)` after every
pool/concat, and an upsample-then-conv decoder (not a transposed convolution)
with skip connections — adapted to a single grayscale input channel and a
single binary (ice/no-ice) output channel instead of the notebook's 3-channel
RGB input and 8-class ice-concentration output.

`train_unet.py` trains this exact class externally (Colab/Kaggle/CPU — it is
never run by the app itself) using the same augmentation (random flips, ±5°
rotation) and `ReduceLROnPlateau` schedule as the notebook, then only commits
`models/unet_weights.pth` if the trained model clears the acceptance bar:
Dice ≥ 0.60 **and** ≥ 0.05 better than the existing Otsu baseline, measured on
the same held-out patches for both. Until that bar is cleared, Otsu remains
the active path — the UI's provenance caption always says which one actually
ran: `Active Path: U-Net` or `Active Path: Otsu`.

---

## Project structure

```
PolarNav-AI/
├── engine.py                          # All navigation + optional ML logic
│   ├── grid_to_latlon()               #   grid <-> geographic conversion (clamped)
│   ├── resolve_sar_path()             #   real-SAR preference with synthetic fallback
│   ├── load_seaice_south()            #   NSIDC seaice.csv, Antarctic rows only
│   ├── sample_seaice_row()            #   picks one random real (date, extent) row
│   ├── make_synthetic_sar()           #   optional target_extent -> ice-blob density
│   ├── SmallUNet                      #   architecture matched to the reference notebook
│   ├── detect_ice()                   #   F2 - returns (original, mask, pixel_count, active_path)
│   ├── build_risk_grid()              #   F3 - 0..10 risk field, optional KMeans profiling
│   ├── predict_drift()                #   F4 - current + 3% wind, Ridge model when models/drift_model.joblib loads
│   ├── build_drift_field()            #   F4 - wind/current CSV resampled to the 40x40 grid
│   ├── predict_risk_grid()            #   F4 - 24-h forecast: advect risk by drift, max-combine
│   ├── kmeans_band_edges()            #   overlay legend bands (KMeans, fixed 20% fallback)
│   ├── astar()                        #   F5 - risk-weighted, None if unreachable
│   ├── route_metrics()                #   F6 - + current/predicted exposure, predicted crossings
│   ├── cpa_km() / classify_threat()   #   F6 - closest point of approach -> HIGH/MED/LOW
│   ├── suggest_reroute()              #   F6 - 2x-penalty A* offered as text, never auto-applied
│   ├── save_route()                   #   F7 - SQLite WAL, returns bool, never raises
│   └── strict_json()                  #   F9 - vessel-API-ready payload schema (transport = Phase 2)
├── app.py                             # Streamlit UI - rendering only, no algorithms
├── api.py                             # Bridge Console backend (FastAPI) - same engine.py calls
│                                       #   as app.py, ported not imported (see docs/CONSOLE_PARITY.md)
├── mapview.py                         # Shared Folium map builder - both app.py and api.py call
│                                       #   this directly, so the map is one copy, not a fork
├── console/
│   ├── index.html                     #   MISSION / DATA / SYSTEM tabs, zero build step
│   ├── app.js                         #   fetch()-only, no framework, no bundler
│   └── style.css
├── console.bat                        # One-click Windows launcher for the Bridge Console
├── requirements-api.txt               # Additive to requirements.txt: fastapi/uvicorn/pydantic
├── bus.py                             # Event bus - always tails to events.log; Kafka only if a
│                                       #   broker happens to be reachable (loopback-only probe)
├── satcom_sim.py                      # Realtime sim layer (Bridge Console's LIVE AUTO mode) -
├── drop_watcher.py                    #   simulated satellite pass -> validate/quarantine ->
├── nmea_sim.py                        #   simulated live GPS feed -> drift-triggered auto-replan.
│                                       #   Separate processes, off by default (MANUAL mode); none
│                                       #   of this is imported by engine.py, app.py, or api.py's
│                                       #   core routes
├── build_drops_stock.py               # One-time authoring tool for drops_stock/*.zip - not run
│                                       #   by the app or by satcom_sim.py
├── drops_stock/                       # Pre-built data-drop packages satcom_sim.py delivers
├── train_unet.py                      # External SmallUNet training (Colab/Kaggle/CPU;
│                                       #   not run by the app) -> models/unet_weights.pth
├── train_drift.py                     # External Ridge drift training (synthetic physics samples
│                                       #   unless data/features.csv exists) -> models/drift_model.joblib
├── TRAINING_REPORT.md                 # What was / was not trained, with numbers and provenance
├── sea-ice-segmentation-u-net.ipynb   # Reference notebook SmallUNet's architecture is synced to
├── seaice.csv                         # NSIDC Sea Ice Index (drives synthetic SAR density)
├── requirements.txt / setup_demo.bat  # Pinned deps + one-shot offline-prep install
├── static/                            # Vendored leaflet.js + leaflet.css (offline map, no CDN)
├── data/
│   ├── icebergs.csv                  # id, lat, lon, mass_kt, freeboard_m
│   ├── wind_current.csv              # lat, lon, u_wind, v_wind, u_current, v_current
│   ├── coast.geojson                 # Bundled coastline - inlined, never fetched
│   └── sar_metadata.json             # Sentinel-1 acquisition parameters
├── models/
│   └── drift_model.joblib            # Ridge drift model (tracked; <1 KB). unet_weights.pth is
│                                      #   git-ignored and absent - no labelled patches yet
├── docs/
│   ├── API_CONTRACT.md               # Every api.py route: request/response shape, example JSON
│   ├── CONSOLE_PARITY.md             # Which app.py lines each api.py helper mirrors, line-for-line
│   ├── UI_AUDIT.md                   # Console UI audit log (defects found + fixed, by severity)
│   ├── JUDGE_GUIDE.md / .pdf         # Judge-facing walkthrough
│   ├── DEMO_DAY_RUNBOOK.md           # Step-by-step run-of-show for a live demo
│   └── REVIEW_REMEDIATION.md         # Security-review findings and how they were closed
└── demo.bat                          # One-click Windows launcher (Streamlit)
```

`engine.py` has **no Streamlit import** — the full pipeline is testable headless,
which is why the self-test can gate every change.

---

## Vessel-API-ready payload schema (F9, transport = Phase 2)

`strict_json()` returns exactly these keys, ready for a shipboard system:

| Key | Type | Meaning |
|---|---|---|
| `status` | string | `OK` on a successful plan |
| `mode` | string | Always `OFFLINE` — the payload asserts its own provenance |
| `route_waypoints` | list of [lat, lon] | The optimized A* route |
| `total_distance_km` | float | Path length, e.g. 129.112 |
| `risk_exposure` | float | Mean risk along the route, e.g. 0.325 |
| `drift_predictions` | list of [lat, lon] | Where each iceberg is expected in 24 h |
| `generated_at` | string | UTC ISO-8601 with a Z suffix |

Because `mode` is hardcoded, a receiving system always knows the route was computed
without external input.

---

## Offline guarantees

This is not offline-capable; it is offline by construction.

- **No `requests`, `urllib`, or any cloud SDK, anywhere.** Grep the source
  and you will find none.
- **Every `socket` use is loopback-only, by grep-able construction.**
  `nmea_sim.py` runs a stdlib TCP server on `127.0.0.1:10110` (a stage-safe
  GPS mimic); `app.py`/`api.py` connect to that exact address to read it;
  `bus.py`'s optional Kafka path only ever probes `localhost:9092`. None
  of this reaches past the machine it runs on — there is no code path to
  any other host.
- **`folium.Map(tiles=None)`** — the browser never requests basemap tiles from a
  CDN. Geography comes from `data/coast.geojson`, which Folium inlines into the page.
  The Bridge Console vendors the same `leaflet.js`/`leaflet.css` under `static/`
  and serves them itself (`GET /static/...`, `127.0.0.1:8000` only) — checked
  directly with a live network-request log: zero non-localhost requests,
  iframe sub-resources included (see `docs/UI_AUDIT.md`).
- **SQLite, not a database server** — history survives restarts with nothing running.
- **No keys, no `.env` requirement, no account.**

### Failure handling

Every stage degrades instead of crashing, because a demo — or a bridge — cannot
afford a traceback.

| Failure | Behavior |
|---|---|
| Missing or malformed CSV | Empty defaults plus a visible info note |
| No SAR image present | Synthetic sample generated on the fly |
| A* finds no route | Falls back to the direct path plus a warning |
| Malformed iceberg row | Row skipped, grid still builds |
| SQLite locked (two tabs) | `save_route()` returns False, UI continues |
| Zero-risk direct baseline | Division guarded, metrics return 0.0 |
| Corrupt/malformed simulated drop (Bridge Console) | `drop_watcher.py` quarantines it (checksum/schema check) instead of ingesting it; visible in the console's DATA tab quarantine log with a reason, nothing crashes downstream |

Streamlit reruns are handled explicitly: every step output lives in `session_state`
as plain data, and the Folium map is rebuilt fresh at page level from that data on
each run. There is exactly **one** `st_folium()` call site in the app, so the map
persists through widget changes and file-watcher reruns instead of flickering away.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI (dashboard) | Streamlit | Zero-config local server, no frontend build |
| UI (console) | FastAPI + uvicorn + vanilla JS | A second, tab-based frontend on the same engine — no framework, no bundler, no build step, same offline guarantees |
| Mapping | Folium (+ streamlit-folium for the dashboard) | Renders client-side with tiles disabled; `mapview.py` is the one shared builder both frontends call |
| Vision | OpenCV (+ optional SmallUNet) | Otsu/morphology is the shipped default; a PyTorch U-Net activates only once trained weights pass the accuracy bar |
| Drift | scikit-learn Ridge (+ physics fallback) | `models/drift_model.joblib` ships and loads; trained on synthetic physics samples (see TRAINING_REPORT.md), so it reproduces the kinematics rather than adding observed-drift skill. The formula remains the per-row fallback |
| Compute | NumPy | Vectorized risk grid |
| Data | pandas | CSV ingest with column validation |
| Storage | SQLite (stdlib) | Serverless persistence |

**Eight core dependencies, CPU-only.** PyTorch and scikit-learn are optional
additions mandated by SIH26059's AI/ML requirement — no GPU required, and every
ML path has a non-ML fallback that is what's actually active until trained
weights ship (see `requirements.txt` / `setup_demo.bat`).

---

## Team Zero Output

Built for Smart India Hackathon problem statement **SIH26059** — offline navigation
and route optimization for polar research vessels — for NCPOR, Ministry of Earth
Sciences.
