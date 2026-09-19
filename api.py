"""
api.py - Bridge Console API (SIH26059, additive-only, app.py untouched).

A thin FastAPI adapter in front of engine.py's existing math, for the new
vanilla-JS console (console/*). Every number this API returns comes from
the exact same engine.py functions app.py already calls -- no new math
lives here, only orchestration + JSON/SVG shaping + wall-clock timing.

Why this file duplicates a handful of small app.py helpers (the folium map
builder, the NMEA loopback probe, the along-route-distance helper, the
F1/F2/route orchestration bodies): app.py cannot be imported as a module
(it calls st.* at module scope on load, which breaks outside a real
Streamlit run) and it is explicitly out of scope to edit. So the only way
to reuse its behavior here is to port it verbatim, not to share code with
it. docs/CONSOLE_PARITY.md names exactly which app.py line ranges each
function below mirrors, so a future app.py edit knows to check here too.

v3 additions beyond the original Bridge Console: richer /detect and a new
/forecast (surfacing engine.py diagnostics that were previously computed
and discarded -- see engine.py's own docstrings for exactly which),
/nsidc, /drop/receipts, /system, and a Sim Deck (/sims/status,
/sims/start, /sims/stop, /drop/now) that manages the 3 sim scripts as
guarded subprocesses via a PID file. Every engine.py signature this file
now unpacks differently was extended (not behaviorally changed) this same
mission -- see engine.py's own docstrings for detect_ice, build_risk_grid,
_kmeans_risk_weights and predict_risk_grid.

State model: this is a single-vessel, single-operator offline console, not
a multi-tenant web service, so "session state" is one process-wide dict
(_state below) -- the same module-level-globals idiom bus.py already uses
for its own _mode/_producer cache, not a new pattern.

Binds 127.0.0.1:8000 only (see console.bat) -- loopback-only by the same
source-level convention nmea_sim.py and bus.py already use, matching this
repo's actual "offline" enforcement point (Python constants, not config).
"""
import base64
import glob
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import folium
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import bus  # existing module, unmodified -- same dual-write event log app.py uses
import mapview  # shared map builder -- see mapview.py; app.py uses the same module

from engine import (
    make_synthetic_sar, detect_ice, build_risk_grid, predict_iceberg_drift,
    astar, direct_path, route_metrics, save_route, load_routes, strict_json,
    grid_to_latlon, resolve_sar_path, sample_seaice_row, extent_to_coverage,
    build_drift_field, predict_risk_grid, kmeans_band_edges, load_drift_model,
    cpa_km, classify_threat, suggest_reroute,
    GRID, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) or "."

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
    "kmeans_diagnostics": None,  # {centroids, tier_multipliers} from the last /detect, or None
    "ice_view": None,           # see _detect()'s docstring for the full shape
    "route_data": None,         # mirrors app.py's session_state.route_data shape
    "last_drop_at": None,       # UTC datetime; drives /status's drop.age_h/stale
    "last_route_start_ll": None,
}

# Sim Deck process-management state.
RUNTIME_DIR = os.path.join(REPO_ROOT, "runtime")
PIDS_PATH = os.path.join(RUNTIME_DIR, "sims.pids")
SIM_SCRIPTS = {
    "satcom": ["satcom_sim.py", "--interval", "20"],
    "watcher": ["drop_watcher.py"],
    "nmea": ["nmea_sim.py"],
}
_SIM_INTERVAL_S = 20.0
_sim_started_at: dict[str, float] = {}  # in-process only; used for the OPS countdown ring, not correctness
_drop_now_counter = 0


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
    cpa_table: list[dict[str, Any]]
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


def _build_map(rd: Optional[dict]) -> folium.Map:
    """Thin wrapper around the shared mapview.build_map -- see mapview.py
    for the actual construction (used identically by app.py)."""
    return mapview.build_map(
        rd, leaflet_prefix="/static",
        lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX,
        grid_to_latlon=grid_to_latlon,
    )


def _encode_png_b64(img: np.ndarray) -> str:
    """Encodes a numpy image array as a base64 PNG data string -- pure
    formatting, no math, so the DETECT panel can show the SAR/mask images
    the engine already produced without a separate image-serving route."""
    try:
        ok, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
    except Exception:
        return ""


def _grid_to_svg(grid: Optional[np.ndarray], edges: list) -> Optional[str]:
    """Thin wrapper around the shared mapview.grid_to_svg_heatmap."""
    return mapview.grid_to_svg_heatmap(grid, edges)


def _ridge_status() -> dict:
    """Whether Ridge is active, plus its held-out R^2 and provenance label
    -- reads an existing engine.py function and an existing report file
    that nothing in either UI reads at runtime today. No engine.py change."""
    try:
        active = load_drift_model() is not None
    except Exception:
        active = False
    r2 = None
    provenance = "not yet measured"
    try:
        report_path = os.path.join(REPO_ROOT, "drift_training_report.json")
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            r2 = report.get("heldout_r2_ridge")
            provenance = "synthetic physics samples — pipeline certification, not observed data"
    except Exception:
        pass
    return {"active": active, "r2": r2, "provenance": provenance}


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


_GUARD_REASON = {
    "unet_disabled": "SmallUNet not requested for this call.",
    "weights_missing": "No trained weights found (or torch unavailable) — using Otsu.",
    "unet_failed": "SmallUNet inference raised an exception — using Otsu.",
    "guard_rejected": "SmallUNet coverage fell outside the accepted 1–60% band — using Otsu.",
    "unet_accepted": "SmallUNet accepted; Otsu not needed.",
}


def _detect(sar_path: str) -> dict:
    """Mirrors app.py's F2 button body (app.py:319-348), including its
    exact hardcoded sar_path="data/sar_sample.png" quirk (F2 never reads a
    live receipt's own sar_path) -- kept as-is for true behavioral parity
    with app.py, not 'fixed'. Extended this mission with everything
    detect_ice/build_risk_grid now additionally return: Otsu threshold,
    guard status + a human fallback reason, coverage %, inference time
    (measured here, in the adapter, per this mission's own instruction --
    not inside engine.py), a 32-bin intensity histogram of the original
    image (a plain numpy summary stat, not new decision logic), and base64
    PNG previews of the original image and mask so the console's DETECT
    tab can show them without a separate image-serving endpoint."""
    sar_path = "data/sar_sample.png"
    t0 = time.perf_counter()
    orig_img, mask_img, n_cells, ice_path, ice_diag = detect_ice(sar_path)
    inference_ms = (time.perf_counter() - t0) * 1000.0
    _state["risk_grid"], _state["kmeans_diagnostics"] = build_risk_grid(
        mask_img, GRID, icebergs_df=_state["icebergs_df"])
    _src = resolve_sar_path(sar_path)
    hist_counts, _ = np.histogram(orig_img, bins=32, range=(0, 255))

    ice_view = {
        "n_cells": int(n_cells),
        "coverage_pct": ice_diag["coverage_pct"],
        "source": "Source: real Sentinel-1 crop" if _src != sar_path else "Source: synthetic sample",
        "active_path": ("Active model: SmallUNet (trained weights)" if ice_path == "unet"
                         else "Active model: OpenCV Otsu (fallback)"),
        "guard_status": ice_diag["guard_status"],
        "fallback_reason": _GUARD_REASON.get(ice_diag["guard_status"], ""),
        "otsu_threshold": ice_diag["otsu_threshold"],
        "inference_ms": inference_ms,
        "histogram": hist_counts.tolist(),
        "orig_png_b64": _encode_png_b64(orig_img),
        "mask_png_b64": _encode_png_b64(mask_img),
    }
    _state["ice_view"] = ice_view
    return ice_view


def _forecast_bundle() -> dict:
    """Everything the FORECAST tab needs, independent of whether a route
    has been computed yet (only needs an /detect to have run). Never
    raises -- returns {"error": ...} on failure."""
    try:
        icebergs_df = _state["icebergs_df"]
        wind_df = _state["wind_df"]
        risk_grid = _state["risk_grid"]

        dropped_count = 0
        if not icebergs_df.empty and {"lat", "lon"}.issubset(icebergs_df.columns):
            coerced = icebergs_df[["lat", "lon"]].apply(pd.to_numeric, errors="coerce")
            dropped_count = int(coerced.isna().any(axis=1).sum())

        bergs = []
        risk_pred = risk_advected = None
        band_edges = [20.0, 40.0, 60.0, 80.0]
        if risk_grid is not None:
            pred_df = predict_iceberg_drift(icebergs_df, wind_df, hours=24.0)
            for _, row in pred_df.iterrows():
                if pd.isna(row.get("pred_lat")) or pd.isna(row.get("pred_lon")):
                    continue
                _id = row.get("id", "?")
                if isinstance(_id, float) and _id.is_integer():
                    _id = int(_id)
                bergs.append({
                    "id": _id,
                    "mass_kt": None if pd.isna(row.get("mass_kt")) else float(row["mass_kt"]),
                    "freeboard_m": None if pd.isna(row.get("freeboard_m")) else float(row["freeboard_m"]),
                    "now_ll": [float(row["lat"]), float(row["lon"])],
                    "plus24h_ll": [float(row["pred_lat"]), float(row["pred_lon"])],
                    "vector_km": _km_between((row["lat"], row["lon"]), (row["pred_lat"], row["pred_lon"])),
                })
            drift_field = build_drift_field(wind_df)
            risk_pred, risk_advected = predict_risk_grid(risk_grid, drift_field, hours=24.0)
            band_edges, _ = kmeans_band_edges(risk_pred)

        return {
            "bergs": bergs, "ridge": _ridge_status(),
            "risk_grid": risk_grid, "risk_pred": risk_pred, "risk_advected": risk_advected,
            "band_edges": band_edges, "kmeans_diagnostics": _state.get("kmeans_diagnostics"),
            "dropped_count": dropped_count,
        }
    except Exception as e:
        return {"error": f"Forecast computation failed: {e}"}


def _compute_route(start_coord, goal_coord) -> dict:
    """Mirrors app.py's _compute_route (app.py:364-546), state dict in
    place of st.session_state. Raises on failure (unlike app.py's
    bool-return form) -- the route caller below turns that into a 4xx.
    Extended this mission: keeps the full per-iceberg CPA table (not just
    HIGH/MED prose lines -- LOW-tier icebergs used to be silently dropped
    from all per-iceberg display) and the reroute path's own geometry
    (previously discarded, now drawn on the map as a 3rd polyline)."""
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
    risk_pred, _ = predict_risk_grid(risk_grid, drift_field, hours=24.0)
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

    # Full CPA table -- every iceberg, not just HIGH/MED (those got silently
    # dropped from all per-iceberg display before this mission).
    cpa_table = []
    alerts = []
    for ib_id, cpa, curr_loc, pred_loc in sorted(cpas, key=lambda t: t[1]):
        lvl = classify_threat(cpa, 0)
        along = _along_route_km(route_ll, [curr_loc, pred_loc]) if lvl in ("HIGH", "MED") else None
        cpa_table.append({"id": ib_id, "cpa_km": cpa, "tier": lvl,
                           "note": "monitor" if lvl == "MED" else ("reroute considered" if lvl == "HIGH" else "clear")})
        if lvl not in ("HIGH", "MED"):
            continue
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
        "reroute_path": reroute[0] if reroute else None,
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
        "cpa_table": cpa_table,
        "alerts": alerts, "notes": notes,
        "reroute_delta_km": reroute[1] if reroute else None,
        "strict_json": sj,
    }


# -----------------------------------------------------------------------------
# Sim Deck process management
# -----------------------------------------------------------------------------
def _read_pids() -> dict:
    if not os.path.exists(PIDS_PATH):
        return {}
    try:
        with open(PIDS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_pids(pids: dict):
    try:
        os.makedirs(RUNTIME_DIR, exist_ok=True)
        with open(PIDS_PATH, "w", encoding="utf-8") as f:
            json.dump(pids, f)
    except Exception:
        pass


def _pid_alive(pid) -> bool:
    """Cross-platform liveness check, stdlib only. Required, not optional:
    confirmed this mission that on this Windows machine, nmea_sim.py's own
    SO_REUSEADDR setting lets a SECOND instance silently bind to the same
    port with no exception -- so "try to start, catch an error" cannot be
    trusted to detect an already-running sim; only an explicit liveness
    check on our own recorded PID can."""
    if not pid:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                  capture_output=True, text=True, timeout=3)
            return str(pid) in out.stdout
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _dir_count(path: str, dirs_only: bool = False) -> int:
    if not os.path.isdir(path):
        return 0
    try:
        names = os.listdir(path)
        if dirs_only:
            return sum(1 for n in names if os.path.isdir(os.path.join(path, n)))
        return len(names)
    except Exception:
        return 0


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

    try:
        pids = _read_pids()
        satcom_status = {"running": _pid_alive(pids.get("satcom"))}
    except Exception:
        satcom_status = {"running": False}

    iv = _state.get("ice_view")
    model_status = {"active_path": iv["active_path"] if iv else "not yet detected"}

    last = _state.get("last_drop_at")
    if last is None:
        drop_status = {"age_h": None, "stale": False}
    else:
        age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
        drop_status = {"age_h": age_h, "stale": age_h > 12}

    return {"watcher": watcher, "bus": bus_status, "nmea": nmea_status,
            "satcom": satcom_status, "model": model_status, "drop": drop_status}


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


@app.get("/forecast")
def forecast():
    b = _forecast_bundle()
    if b.get("error"):
        return b
    edges = b["band_edges"]
    heatmaps = {
        "current": _grid_to_svg(b["risk_grid"], edges),
        "predicted": _grid_to_svg(b["risk_pred"], edges),
        "advected": _grid_to_svg(b["risk_advected"], edges),
    }
    return {
        "bergs": b["bergs"], "ridge": b["ridge"], "heatmaps": heatmaps,
        "band_edges": edges, "kmeans_diagnostics": b["kmeans_diagnostics"],
        "dropped_count": b["dropped_count"],
    }


@app.get("/nsidc")
def nsidc():
    row = _state.get("seaice_context")
    if row is None:
        return {"available": False}
    try:
        coverage = extent_to_coverage(row["extent"])
        blob_count = int(round(2 + coverage * 18))  # mirrors make_synthetic_sar's own inline formula
    except Exception:
        coverage, blob_count = None, None
    return {"available": True, "row": row, "coverage_fraction": coverage, "blob_count": blob_count}


@app.get("/drop/receipts")
def drop_receipts():
    validated = []
    if os.path.isdir("drops_done"):
        for name in sorted(os.listdir("drops_done"), reverse=True):
            path = os.path.join("drops_done", name)
            if os.path.isdir(path):
                validated.append({"batch_id": name, "files": sorted(os.listdir(path))})
    quarantined = []
    if os.path.isdir("drops_quarantine"):
        for name in sorted(os.listdir("drops_quarantine"), reverse=True):
            path = os.path.join("drops_quarantine", name)
            if not os.path.isdir(path):
                continue
            reason = ""
            reason_path = os.path.join(path, "reason.txt")
            if os.path.exists(reason_path):
                try:
                    with open(reason_path, encoding="utf-8") as f:
                        reason = f.read()
                except Exception:
                    reason = ""
            quarantined.append({"batch_id": name, "reason": reason})
    return {"validated": validated, "quarantined": quarantined,
            "validated_count": len(validated), "quarantined_count": len(quarantined)}


@app.get("/system")
def system():
    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        pass

    telemetry_disabled = None
    try:
        with open(os.path.join(REPO_ROOT, ".streamlit", "config.toml"), encoding="utf-8") as f:
            cfg_text = f.read()
        telemetry_disabled = "gatherUsageStats" in cfg_text and "false" in cfg_text.split("gatherUsageStats")[1][:20]
    except Exception:
        pass

    unet_path = os.path.join(REPO_ROOT, "models", "unet_weights.pth")
    unet_exists = os.path.exists(unet_path)

    return {
        "commit": commit,
        "telemetry_disabled": telemetry_disabled,
        "tiles": "None (offline -- no basemap CDN)",
        "model_registry": {
            "unet_weights_present": unet_exists,
            "unet_weights_size_kb": round(os.path.getsize(unet_path) / 1024.0, 1) if unet_exists else None,
            "dice_status": "not yet measured (0 labelled training patches)",
            "quantization": "not yet measured (no trained U-Net exists)",
        },
        "resource_budget": "CPU-only inference, target <2GB RAM",
    }


@app.get("/sims/status")
def sims_status():
    pids = _read_pids()
    sims = {name: {"running": _pid_alive(pids.get(name)), "pid": pids.get(name)} for name in SIM_SCRIPTS}
    started_at = _sim_started_at.get("satcom")
    next_pass_in_s = None
    if started_at and sims["satcom"]["running"]:
        next_pass_in_s = max(0.0, _SIM_INTERVAL_S - ((time.time() - started_at) % _SIM_INTERVAL_S))
    return {
        "sims": sims,
        "mode": "LIVE AUTO" if any(s["running"] for s in sims.values()) else "MANUAL",
        "inbox": _dir_count("drops_in"), "done": _dir_count("drops_done", dirs_only=True),
        "quarantine": _dir_count("drops_quarantine", dirs_only=True),
        "interval_s": _SIM_INTERVAL_S, "next_pass_in_s": next_pass_in_s,
    }


@app.post("/sims/start")
def sims_start():
    pids = _read_pids()
    results = {}
    for name, argv in SIM_SCRIPTS.items():
        if _pid_alive(pids.get(name)):
            results[name] = {"ok": True, "detail": "already running"}
            continue
        if name == "satcom" and not os.path.isdir(os.path.join(REPO_ROOT, "drops_stock")):
            results[name] = {"ok": False, "detail": "drops_stock/ not found -- run build_drops_stock.py first"}
            continue
        try:
            proc = subprocess.Popen([sys.executable, *argv], cwd=REPO_ROOT)
            pids[name] = proc.pid
            if name == "satcom":
                _sim_started_at["satcom"] = time.time()
            results[name] = {"ok": True, "detail": f"started pid {proc.pid}"}
        except Exception as e:
            results[name] = {"ok": False, "detail": str(e)}
    _write_pids(pids)
    return results


@app.post("/sims/stop")
def sims_stop():
    pids = _read_pids()
    results = {}
    for name in SIM_SCRIPTS:
        pid = pids.get(name)
        if not _pid_alive(pid):
            results[name] = {"ok": True, "detail": "already stopped"}
            continue
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=3)
            else:
                os.kill(pid, 15)
            results[name] = {"ok": True, "detail": f"stopped pid {pid}"}
        except Exception as e:
            results[name] = {"ok": False, "detail": str(e)}
    _write_pids({})
    _sim_started_at.clear()
    return results


@app.post("/drop/now")
def drop_now():
    """Performs satcom_sim.py's own atomic delivery action once, directly,
    independent of whether its timer loop is running -- there's no IPC to
    the running process, so this mirrors its copy-to-.part+os.rename logic
    verbatim (satcom_sim.py:51-58) rather than signaling it."""
    global _drop_now_counter
    stock_dir = os.path.join(REPO_ROOT, "drops_stock")
    stock = sorted(f for f in os.listdir(stock_dir) if f.endswith(".zip")) if os.path.isdir(stock_dir) else []
    if not stock:
        raise HTTPException(status_code=409, detail="drops_stock/ has no .zip files -- run build_drops_stock.py first")
    name = stock[_drop_now_counter % len(stock)]
    _drop_now_counter += 1
    in_dir = os.path.join(REPO_ROOT, "drops_in")
    os.makedirs(in_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(in_dir, f"{ts}_{name}")
    tmp_dest = dest + ".part"
    try:
        shutil.copy(os.path.join(stock_dir, name), tmp_dest)
        os.rename(tmp_dest, dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"drop failed: {e}")
    return {"ok": True, "delivered": os.path.basename(dest)}


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
