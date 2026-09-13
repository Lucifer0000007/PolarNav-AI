# 🧊 PolarNav AI

**Offline-first navigation decision support for Antarctic research vessels.**

SAR ice detection → 24-hour iceberg drift prediction → risk-aware A* routing.
No internet. No API keys. No cloud. Runs on a laptop in the Southern Ocean.

`SIH26059` · TEKATHON-5.0 (2026) · **Team Zero Output**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Offline](https://img.shields.io/badge/network-zero%20calls-success)
![Stack](https://img.shields.io/badge/stack-Streamlit%20%2B%20OpenCV%20%2B%20NumPy-orange)

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
| **F2** | Ice hazard detection | OpenCV: Gaussian blur → Otsu threshold → morphological opening |
| **F3** | Risk grid | 40×40 cells, 0–10 risk, iceberg positions stamped as hard hazards |
| **F4** | 24-h drift prediction | Vector kinematics — current + 3% wind forcing |
| **F5** | Risk-aware routing | Modified 8-directional A*, green optimal vs. red direct baseline |
| **F6** | Live metrics | Distance, risk score, ice crossings, risk reduction %, fuel penalty % |
| **F7** | Route history | Local SQLite (WAL mode), persists with no server |
| **F8** | Fully offline | No sockets, no keys, no CDN — tiles=None basemap |
| **F9** | Vessel API output | Strict JSON payload for NCPOR shipboard systems |

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

pip install streamlit folium streamlit-folium pandas numpy opencv-python
```

Run the offline self-test (no browser, validates the whole engine):

```bash
python engine.py
```

Launch the dashboard:

```bash
streamlit run app.py        # -> http://localhost:8501
```

On Windows, `demo.bat` does both steps in one double-click.

### Demo flow

1. **📡 Simulate Satellite Data Drop** — ingests CSVs, generates/loads SAR
2. **🔍 Detect Ice Hazards** — shows source imagery beside the detected mask
3. **🧭 Predict Drift + Generate Route** — drift arrows, both routes, live metrics
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

## Project structure

```
PolarNav-AI/
├── engine.py              # All navigation logic - pure numpy/OpenCV/stdlib
│   ├── grid_to_latlon()   #   grid <-> geographic conversion (clamped)
│   ├── resolve_sar_path() #   real-SAR preference with synthetic fallback
│   ├── detect_ice()       #   F2 - returns (original, mask, pixel_count)
│   ├── build_risk_grid()  #   F3 - 0..10 risk field
│   ├── predict_drift()    #   F4 - current + 3% wind
│   ├── astar()            #   F5 - risk-weighted, None if unreachable
│   ├── route_metrics()    #   F6 - guarded against divide-by-zero
│   ├── save_route()       #   F7 - SQLite WAL, returns bool, never raises
│   └── strict_json()      #   F9 - NCPOR vessel API contract
├── app.py                 # Streamlit UI - rendering only, no algorithms
├── data/
│   ├── icebergs.csv       # id, lat, lon, mass_kt, freeboard_m
│   ├── wind_current.csv   # lat, lon, u_wind, v_wind, u_current, v_current
│   ├── coast.geojson      # Bundled coastline - inlined, never fetched
│   └── sar_metadata.json  # Sentinel-1 acquisition parameters
└── demo.bat               # One-click Windows launcher
```

`engine.py` has **no Streamlit import** — the full pipeline is testable headless,
which is why the self-test can gate every change.

---

## Vessel API contract (F9)

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

- **No network imports anywhere** — no `requests`, `urllib`, `socket`, or any cloud
  SDK. Grep the source and you will find none.
- **`folium.Map(tiles=None)`** — the browser never requests basemap tiles from a
  CDN. Geography comes from `data/coast.geojson`, which Folium inlines into the page.
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

Streamlit reruns are handled explicitly: every step output lives in `session_state`
as plain data, and the Folium map is rebuilt fresh at page level from that data on
each run. There is exactly **one** `st_folium()` call site in the app, so the map
persists through widget changes and file-watcher reruns instead of flickering away.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit | Zero-config local server, no frontend build |
| Mapping | Folium + streamlit-folium | Renders client-side with tiles disabled |
| Vision | OpenCV | Otsu and morphology, no model weights to ship |
| Compute | NumPy | Vectorized risk grid |
| Data | pandas | CSV ingest with column validation |
| Storage | SQLite (stdlib) | Serverless persistence |

**Six dependencies. No ML frameworks, no model downloads, no GPU.** The install
works on any vessel laptop.

---

## Team Zero Output

Built for Smart India Hackathon problem statement **SIH26059** — offline navigation
and route optimization for polar research vessels — for NCPOR, Ministry of Earth
Sciences.
