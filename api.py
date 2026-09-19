"""
api.py - Bridge Console API (SIH26059, additive-only, app.py untouched).

A thin FastAPI adapter in front of engine.py's existing math, for the new
vanilla-JS console (console/*). Every number this API returns comes from
the exact same engine.py functions app.py already calls -- no new math
lives here, only orchestration + JSON shaping.

Why this file duplicates a handful of small app.py helpers (the folium map
builder, the NMEA loopback probe, the along-route-distance helper, the
F1/F2/route orchestration bodies): app.py cannot be imported as a module
(it calls st.* at module scope on load, which breaks outside a real
Streamlit run) and it is explicitly out of scope to edit. So the only way
to reuse its behavior here is to port it verbatim, not to share code with
it. docs/CONSOLE_PARITY.md names exactly which app.py line ranges each
function below mirrors, so a future app.py edit knows to check here too.

State model: this is a single-vessel, single-operator offline console, not
a multi-tenant web service, so "session state" is one process-wide dict
(_state below) -- the same module-level-globals idiom bus.py already uses
for its own _mode/_producer cache, not a new pattern.

Binds 127.0.0.1:8000 only (see console.bat) -- loopback-only by the same
source-level convention nmea_sim.py and bus.py already use, matching this
repo's actual "offline" enforcement point (Python constants, not config).
"""
import json
import math
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import folium
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import bus  # existing module, unmodified -- same dual-write event log app.py uses

from engine import (
    make_synthetic_sar, detect_ice, build_risk_grid, predict_iceberg_drift,
    astar, direct_path, route_metrics, save_route, load_routes, strict_json,
    grid_to_latlon, resolve_sar_path, sample_seaice_row,
    build_drift_field, predict_risk_grid, kmeans_band_edges,
    cpa_km, classify_threat, suggest_reroute,
    GRID, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
)

# ---- offline Leaflet override -- mirrors app.py:98-99 exactly, but pointed
# at this process's OWN static mount (Streamlit's /app/static/ prefix is
# Streamlit-specific and not reachable from this separate uvicorn process).
# Must be set before any folium.Map(...) is constructed. ----
folium.Map.default_js = [("leaflet", "/static/leaflet.js")]
folium.Map.default_css = [("leaflet_css", "/static/leaflet.css")]

app = FastAPI(title="PolarNav AI Bridge Console API")

# -----------------------------------------------------------------------------
# Voyage state -- the API-process equivalent of app.py's st.session_state.
# Single-process, single-vessel demo: one dict is the whole story, same
# module-level-globals idiom bus.py already uses for _mode/_producer.
# -----------------------------------------------------------------------------
_DEFAULT_ICEBERGS = pd.DataFrame(columns=["id", "lat", "lon", "mass_kt", "freeboard_m"])
_DEFAULT_WIND = pd.DataFrame(columns=["lat", "lon", "u_wind", "v_wind", "u_current", "v_current"])
REQUIRED_ICEBERG_COLS = {"id", "lat", "lon", "mass_kt", "freeboard_m"}
REQUIRED_WIND_COLS = {"lat", "lon", "u_wind", "v_wind", "u_current", "v_current"}

_state: dict[str, Any] = {
    "icebergs_df": _DEFAULT_ICEBERGS,
    "wind_df": _DEFAULT_WIND,
    "seaice_context": None,
    "risk_grid": None,
    "ice_view": None,          # {n_cells, coverage_pct, source, active_path}
    "route_data": None,        # mirrors app.py's session_state.route_data shape
    "last_drop_at": None,      # UTC datetime; drives /status's drop.age_h/stale
    "last_route_start_ll": None,
}


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class RouteRequest(BaseModel):
    start: tuple[int, int]
    goal: tuple[int, int]


class StrictJsonModel(BaseModel):
    status: str
    mode: str
    route_waypoints: list[list[float]]
    total_distance_km: float
    risk_exposure: float
    drift_predictions: list[list[float]]
    generated_at: str


class AlertItem(BaseModel):
    level: str
    text: str


class RouteResponse(BaseModel):
    start_ll: tuple[float, float]
    goal_ll: tuple[float, float]
    metrics: dict[str, Any]
    min_cpa_km: Optional[float]
    alerts: list[AlertItem]
    notes: list[AlertItem]
    reroute_delta_km: Optional[float]
    strict_json: StrictJsonModel


# -----------------------------------------------------------------------------
# Mirrored helpers -- verbatim ports, see docs/CONSOLE_PARITY.md for the
# exact app.py line ranges each of these reproduces.
# -----------------------------------------------------------------------------
def _load_csv_safe(path: str, required_cols: set, empty_df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors app.py:213-225 verbatim."""
    if not os.path.exists(path):
        return empty_df.copy()
    try:
        df = pd.read_csv(path)
    except Exception:
        return empty_df.copy()
    if not required_cols.issubset(df.columns):
        return empty_df.copy()
    return df


def _dm_to_decimal(dm_str: str, hemi: str) -> float:
    """Mirrors app.py's _dm_to_decimal verbatim (NMEA source, pre-M4 move)."""
    dot = dm_str.index(".")
    deg = int(dm_str[:dot - 2])
    minutes = float(dm_str[dot - 2:])
    dec = deg + minutes / 60.0
    return -dec if hemi in ("S", "W") else dec


def _parse_nmea_line(line: str):
    """Mirrors app.py's _parse_nmea_line verbatim."""
    line = line.strip()
    if not line.startswith("$") or "*" not in line:
        return None
    try:
        import pynmea2
        msg = pynmea2.parse(line)
        return float(msg.latitude), float(msg.longitude)
    except Exception:
        pass
    try:
        body = line.split("*")[0][1:]
        fields = body.split(",")
        sid = fields[0]
        if sid.endswith("GGA"):
            return _dm_to_decimal(fields[2], fields[3]), _dm_to_decimal(fields[4], fields[5])
        if sid.endswith("RMC"):
            return _dm_to_decimal(fields[3], fields[4]), _dm_to_decimal(fields[5], fields[6])
    except Exception:
        pass
    return None


def _nmea_probe():
    """Mirrors app.py's _nmea_probe verbatim: 0.2s loopback-only connect
    probe to 127.0.0.1:10110; reads one sentence. Never raises."""
    try:
        with socket.create_connection(("127.0.0.1", 10110), timeout=0.2) as s:
            s.settimeout(1.0)
            buf = b""
            while b"\n" not in buf and len(buf) < 1024:
                chunk = s.recv(256)
                if not chunk:
                    break
                buf += chunk
        return _parse_nmea_line(buf.split(b"\n")[0].decode("ascii", errors="ignore"))
    except Exception:
        return None


def _km_between(ll1, ll2) -> float:
    """Mirrors app.py's _km_between verbatim."""
    lat1, lon1 = ll1
    lat2, lon2 = ll2
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2.0))
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * cos_lat
    return math.hypot(dlat, dlon)


def _along_route_km(route_ll, track_ll, samples: int = 25):
    """Mirrors app.py's _along_route_km verbatim."""
    try:
        route = np.asarray(route_ll, dtype=np.float64).reshape(-1, 2)
        track = np.asarray(track_ll, dtype=np.float64).reshape(-1, 2)
        if route.size == 0 or track.size == 0:
            return None
        if len(track) > 1:
            track = np.vstack([np.linspace(track[i], track[i + 1], samples)
                                for i in range(len(track) - 1)])
        cos_lat = math.cos(math.radians(float(np.mean(route[:, 0]))))
        dlat = (route[:, None, 0] - track[None, :, 0]) * 111.0
        dlon = (route[:, None, 1] - track[None, :, 1]) * 111.0 * cos_lat
        d = np.sqrt(dlat ** 2 + dlon ** 2)
        route_idx = int(np.nanargmin(d) // d.shape[1])
        return sum(_km_between(tuple(route[i]), tuple(route[i + 1])) for i in range(route_idx))
    except Exception:
        return None


_BAND_RGB = [(198, 226, 255), (140, 196, 250), (86, 152, 236), (44, 98, 208), (16, 44, 128)]


def _concentration_rgba(risk_pred, edges) -> np.ndarray:
    """Mirrors app.py's _concentration_rgba verbatim."""
    conc = np.clip(np.asarray(risk_pred, dtype=np.float64) * 10.0, 0.0, 100.0)
    band = np.digitize(conc, edges)
    rgba = np.zeros(conc.shape + (4,), dtype=np.uint8)
    for k, rgb in enumerate(_BAND_RGB):
        rgba[band == k, :3] = rgb
    rgba[..., 3] = np.where(conc > 0.0, 255, 0).astype(np.uint8)
    return rgba


def _build_map(rd: Optional[dict]) -> folium.Map:
    """Mirrors app.py's _build_map verbatim, plus an empty-state AOI-only
    map (rectangle + coastline, no route/icebergs) when rd is None -- the
    console can request /map before any route has been computed."""
    m = folium.Map(location=[-67.5, 60.2], zoom_start=8, tiles=None)
    folium.Rectangle(
        bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        color="#1b2a4a", fill=True, fill_opacity=0.6, weight=0,
    ).add_to(m)
    coast_path = "data/coast.geojson"
    if os.path.exists(coast_path):
        folium.GeoJson(
            coast_path, name="Coastline",
            style_function=lambda f: {"color": "#9aa4b2", "weight": 1.5, "fillOpacity": 0},
        ).add_to(m)

    if rd is None:
        return m

    if rd.get("risk_pred") is not None:
        folium.raster_layers.ImageOverlay(
            image=_concentration_rgba(rd["risk_pred"], rd.get("band_edges") or [20, 40, 60, 80]),
            bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
            opacity=0.45, origin="upper", mercator_project=False,
            name="Predicted ice concentration (24 h)",
        ).add_to(m)

    for curr_loc, pred_loc in rd["icebergs"]:
        folium.CircleMarker(curr_loc, color='red', radius=4, fill=True).add_to(m)
        folium.CircleMarker(pred_loc, color='orange', radius=4, fill=True).add_to(m)
        folium.PolyLine([curr_loc, pred_loc], color='orange', dash_array='5', weight=2).add_to(m)

    opt_latlon = [grid_to_latlon(r, c) for r, c in rd["path"]]
    dir_latlon = [grid_to_latlon(r, c) for r, c in rd["direct"]]
    folium.PolyLine(opt_latlon, color='green', weight=4, opacity=0.8).add_to(m)
    folium.PolyLine(dir_latlon, color='red', dash_array='10', weight=2, opacity=0.6).add_to(m)
    return m


def _ingest() -> dict:
    """Mirrors app.py's F1 button body verbatim (app.py:231-308): prefer a
    live, validated drop_watcher receipt, else fall back to the static
    data/ files. Never raises -- any failure yields an ingest dict with an
    'error' key instead."""
    os.makedirs("data", exist_ok=True)

    live_receipt = None
    try:
        receipt_path = os.path.join("drops_done", "latest_receipt.json")
        if os.path.exists(receipt_path):
            with open(receipt_path, encoding="utf-8") as f:
                candidate = json.load(f)
            if all(os.path.exists(candidate.get(k, "")) for k in ("sar_path", "icebergs_csv", "wind_csv")):
                live_receipt = candidate
    except Exception:
        live_receipt = None

    if live_receipt is not None:
        _state["icebergs_df"] = _load_csv_safe(live_receipt["icebergs_csv"], REQUIRED_ICEBERG_COLS, _DEFAULT_ICEBERGS)
        _state["wind_df"] = _load_csv_safe(live_receipt["wind_csv"], REQUIRED_WIND_COLS, _DEFAULT_WIND)
    else:
        _state["icebergs_df"] = _load_csv_safe("data/icebergs.csv", REQUIRED_ICEBERG_COLS, _DEFAULT_ICEBERGS)
        _state["wind_df"] = _load_csv_safe("data/wind_current.csv", REQUIRED_WIND_COLS, _DEFAULT_WIND)

    sar_path = live_receipt["sar_path"] if live_receipt is not None else "data/sar_sample.png"
    messages: list[dict] = []
    try:
        if live_receipt is None:
            seaice_row = sample_seaice_row()
            if seaice_row is not None:
                make_synthetic_sar(sar_path, target_extent=seaice_row["extent"])
            elif not os.path.exists(sar_path):
                make_synthetic_sar(sar_path)
        else:
            seaice_row = None
        _state["seaice_context"] = seaice_row
        _state["last_drop_at"] = datetime.now(timezone.utc)

        if live_receipt is not None:
            messages.append({"level": "success",
                              "text": f"Live drop consumed: {live_receipt['batch_id']} "
                                      f"(validated {live_receipt['validated_at']})"})
        else:
            messages.append({"level": "success", "text": "data/icebergs.csv ingested successfully."})
            messages.append({"level": "success", "text": "data/wind_current.csv ingested successfully."})
            messages.append({"level": "success", "text": f"SAR imagery available at {sar_path}."})
        if seaice_row is not None:
            messages.append({"level": "info",
                              "text": f"Context: {seaice_row['year']:04d}-{seaice_row['month']:02d}-"
                                      f"{seaice_row['day']:02d} | Extent: {seaice_row['extent']:.1f} M km² (NSIDC)"})
        if _state["icebergs_df"].empty:
            messages.append({"level": "info",
                              "text": f"{'live iceberg data' if live_receipt else 'data/icebergs.csv'} "
                                      f"unavailable — using empty iceberg defaults."})
        if _state["wind_df"].empty:
            messages.append({"level": "info",
                              "text": f"{'live wind/current data' if live_receipt else 'data/wind_current.csv'} "
                                      f"unavailable — using empty wind/current defaults."})
        return {
            "source": "live" if live_receipt is not None else "static",
            "batch_id": live_receipt["batch_id"] if live_receipt is not None else None,
            "messages": messages,
            "seaice_context": seaice_row,
            "sar_path": sar_path,
        }
    except Exception as e:
        return {"source": None, "batch_id": None, "messages": [], "seaice_context": None,
                "sar_path": sar_path, "error": f"Satellite data simulation failed: {e}"}


def _detect(sar_path: str) -> dict:
    """Mirrors app.py's F2 button body verbatim (app.py:319-348), including
    its exact hardcoded sar_path="data/sar_sample.png" quirk (F2 never
    reads a live receipt's own sar_path, even right after a live /detect
    ingest) -- kept as-is for true behavioral parity with app.py, not
    'fixed', since app.py itself is out of scope to change."""
    sar_path = "data/sar_sample.png"
    orig_img, mask_img, n_cells, ice_path = detect_ice(sar_path)
    _state["risk_grid"] = build_risk_grid(mask_img, GRID, icebergs_df=_state["icebergs_df"])
    try:
        coverage_pct = 100.0 * float(np.count_nonzero(mask_img)) / float(mask_img.size)
    except Exception:
        coverage_pct = 0.0
    _src = resolve_sar_path(sar_path)
    ice_view = {
        "n_cells": int(n_cells),
        "coverage_pct": coverage_pct,
        "source": "Source: real Sentinel-1 crop" if _src != sar_path else "Source: synthetic sample",
        "active_path": ("Active model: SmallUNet (trained weights)" if ice_path == "unet"
                         else "Active model: OpenCV Otsu (fallback)"),
    }
    _state["ice_view"] = ice_view
    return ice_view


def _compute_route(start_coord, goal_coord) -> dict:
    """Mirrors app.py's _compute_route verbatim (app.py:413-546), state
    dict in place of st.session_state. Raises on failure (unlike app.py's
    bool-return form) -- the route caller below turns that into a 4xx."""
    risk_grid = _state["risk_grid"]
    notes = []

    pred_df = predict_iceberg_drift(_state["icebergs_df"], _state["wind_df"], hours=24.0)

    drift_list = []
    iceberg_pairs = []
    tracks = []
    for _, row in pred_df.iterrows():
        if pd.isna(row.get('pred_lat')) or pd.isna(row.get('pred_lon')):
            continue
        curr_loc = [row['lat'], row['lon']]
        pred_loc = [row['pred_lat'], row['pred_lon']]
        drift_list.append(pred_loc)
        iceberg_pairs.append((curr_loc, pred_loc))
        _id = row.get('id', '?')
        if isinstance(_id, float) and _id.is_integer():
            _id = int(_id)
        tracks.append((_id, curr_loc, pred_loc))

    drift_field = build_drift_field(_state["wind_df"])
    risk_pred = predict_risk_grid(risk_grid, drift_field, hours=24.0)
    risk_combined = np.maximum(risk_grid, risk_pred) if risk_grid is not None else None
    band_edges, band_src = kmeans_band_edges(risk_pred)

    opt_path = astar(risk_combined, start_coord, goal_coord)
    dir_path = direct_path(risk_grid, start_coord, goal_coord)
    if opt_path is None:
        notes.append({"level": "warning",
                      "text": "No A* route found to the goal — showing the direct path instead."})
        opt_path = dir_path

    metrics = route_metrics(risk_grid, opt_path, dir_path, predicted_grid=risk_pred)

    route_ll = [grid_to_latlon(r, c) for r, c in opt_path]
    cpas = [(ib_id, cpa_km(route_ll, [curr_loc, pred_loc]), curr_loc, pred_loc)
            for ib_id, curr_loc, pred_loc in tracks]
    min_cpa = min((c for _, c, _, _ in cpas), default=float('inf'))
    pred_x = metrics['predicted_crossings']
    threat = classify_threat(min_cpa, 0)
    reroute = suggest_reroute(risk_combined, start_coord, goal_coord, opt_path, risk_pred) if threat == "HIGH" else None
    suggestion = (f"suggested deviation +{reroute[1]:.1f} km" if reroute
                  else "no lower-exposure alternative found")
    alerts = []
    for ib_id, cpa, curr_loc, pred_loc in sorted(cpas, key=lambda t: t[1]):
        lvl = classify_threat(cpa, 0)
        if lvl not in ("HIGH", "MED"):
            continue
        along = _along_route_km(route_ll, [curr_loc, pred_loc])
        if lvl == "HIGH":
            if along is not None:
                alerts.append({"level": "error", "text": f"HIGH: {ib_id} will cross your route in {along:.1f} km — {suggestion}"})
            else:
                alerts.append({"level": "error", "text": f"HIGH: Iceberg {ib_id} CPA {cpa:.1f} km — {suggestion}"})
        elif lvl == "MED":
            if along is not None:
                alerts.append({"level": "warning", "text": f"MED: {ib_id} passes near your route in {along:.1f} km — monitor"})
            else:
                alerts.append({"level": "warning", "text": f"MED: route passes within {cpa:.1f} km of {ib_id} drift corridor"})
    if pred_x > 0:
        alerts.append({"level": "warning", "text": f"MED: Route crosses {pred_x} predicted risk>5 cells - expect icebreaking"})
    if not alerts:
        alerts.append({"level": "info", "text": "LOW: corridor clear for 24 h"})
    alerts.append({"level": "info", "text": f"Predicted risk>5 cells on route: {pred_x}"})

    start_ll = grid_to_latlon(*start_coord)
    goal_ll = grid_to_latlon(*goal_coord)

    save_route(
        start=start_ll, goal=goal_ll,
        distance_km=metrics['path_distance_km'],
        risk_red=metrics['risk_reduction_pct'],
        path_json=[grid_to_latlon(r, c) for r, c in opt_path],
    )

    _state["route_data"] = {
        "path": opt_path, "direct": dir_path, "drift_list": drift_list,
        "metrics": metrics, "icebergs": iceberg_pairs,
        "start_ll": start_ll, "goal_ll": goal_ll,
        "risk_pred": risk_pred, "band_edges": band_edges, "band_src": band_src,
    }
    _state["last_route_start_ll"] = start_ll

    bus.publish("route.computed", {
        "start_ll": start_ll, "goal_ll": goal_ll,
        "distance_km": metrics["path_distance_km"],
        "risk_reduction_pct": metrics["risk_reduction_pct"],
    })
    bus.publish("alerts.raised", {"alerts": alerts})

    sj = strict_json(start_ll, goal_ll, route_ll, metrics, drift_list)

    return {
        "start_ll": start_ll, "goal_ll": goal_ll,
        "metrics": metrics, "min_cpa_km": None if math.isinf(min_cpa) else min_cpa,
        "alerts": alerts, "notes": notes,
        "reroute_delta_km": reroute[1] if reroute else None,
        "strict_json": sj,
    }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/status")
def status():
    try:
        receipt_path = os.path.join("drops_done", "latest_receipt.json")
        age_s = time.time() - os.path.getmtime(receipt_path) if os.path.exists(receipt_path) else None
        watcher = {"fresh": age_s is not None and age_s < 60, "age_s": age_s}
    except Exception:
        watcher = {"fresh": False, "age_s": None}

    try:
        bus_status = {"mode": bus.transport_mode()}
    except Exception:
        bus_status = {"mode": "file"}

    try:
        nmea_status = {"live": _nmea_probe() is not None}
    except Exception:
        nmea_status = {"live": False}

    iv = _state.get("ice_view")
    model_status = {"active_path": iv["active_path"] if iv else "not yet detected"}

    last = _state.get("last_drop_at")
    if last is None:
        drop_status = {"age_h": None, "stale": False}
    else:
        age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
        drop_status = {"age_h": age_h, "stale": age_h > 12}

    return {"watcher": watcher, "bus": bus_status, "nmea": nmea_status,
            "model": model_status, "drop": drop_status}


@app.post("/detect")
def detect():
    ingest_result = _ingest()
    if ingest_result.get("error"):
        return {"ingest": ingest_result, "detection": None}
    try:
        detection = _detect(ingest_result["sar_path"])
    except Exception as e:
        detection = {"error": f"Ice detection failed: {e}"}
    return {"ingest": ingest_result, "detection": detection}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    if _state["risk_grid"] is None:
        raise HTTPException(status_code=409, detail="Run /detect before /route — no risk grid yet.")
    try:
        return _compute_route(tuple(req.start), tuple(req.goal))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route generation failed: {e}")


@app.get("/map", response_class=HTMLResponse)
def get_map():
    m = _build_map(_state["route_data"])
    return m.get_root().render()


@app.get("/events")
def events():
    return bus.tail(20)


@app.get("/history")
def history():
    return load_routes()


# -----------------------------------------------------------------------------
# Static mounts -- registered LAST so the explicit API routes above always
# win first; StaticFiles is Starlette's catch-all fallback for anything
# else. /static reuses the SAME committed static/ folder app.py's Streamlit
# process serves (nothing copied); / serves the console itself, with
# html=True making GET / resolve to console/index.html.
# -----------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="console", html=True), name="console")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
