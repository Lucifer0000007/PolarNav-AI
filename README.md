# 🧊 PolarNav AI

**Offline-first navigation decision support for Antarctic research vessels.**

SAR ice detection → 24-hour iceberg drift prediction → risk-aware A* routing.
No internet. No API keys. No cloud. Runs on a laptop in the Southern Ocean.

`SIH26059` · TEKATHON-5.0 (2026) · **Team Zero Output**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Offline](https://img.shields.io/badge/network-zero%20calls-success)
![Stack](https://img.shields.io/badge/stack-Streamlit%20%2B%20OpenCV%20%2B%20PyTorch*%20%2B%20scikit--learn*-orange)

\* see Model status below.

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
| **F8** | Fully offline | No sockets, no keys, no CDN — tiles=None basemap; Leaflet 1.9.3 vendored in `static/` and served by Streamlit itself (`server.enableStaticServing`), so the map renders with Wi-Fi off |
| **F9** | Vessel-API-ready schema | Strict JSON payload structured for NCPOR shipboard systems (transport = Phase 2) |

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

Launch the dashboard:

```bash
streamlit run app.py        # -> http://localhost:8501
```

On Windows, `demo.bat` does both steps in one double-click.

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
└── demo.bat                          # One-click Windows launcher
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
