import os
import hashlib
import math
import socket
import time
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone
import json

import bus  # M3: event bus -- kafka-python-ng optional inside bus.py itself; this import is always safe

# Offline navigation engine imports
from engine import (
    make_synthetic_sar, detect_ice, build_risk_grid, predict_iceberg_drift,
    astar, direct_path, route_metrics, save_route, load_routes, strict_json,
    grid_to_latlon, latlon_to_grid, resolve_sar_path, sample_seaice_row,
    build_drift_field, predict_risk_grid, kmeans_band_edges,
    cpa_km, classify_threat, suggest_reroute,
    GRID, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)

# ---- M2: NMEA live position (loopback probe only; unreachable -> pinned, never raises) ----
# Defined this early (before the sidebar/M4 status block below) so both the
# M4 Production Mimic dot and the position-badge fragment further down can
# call _nmea_probe() -- a plain top-level def's globals are resolved at call
# time, so a call site earlier in the file than the def would NameError on
# the script's very first pass.
def _dm_to_decimal(dm_str: str, hemi: str) -> float:
    """NMEA ddmm.mmmm / dddmm.mmmm -> decimal degrees. The last 2 integer
    digits before the decimal point are always minutes, regardless of
    whether the degree part is 2 digits (lat) or 3 (lon) -- verified by
    round-tripping nmea_sim.py's own sentence builder against this."""
    dot = dm_str.index(".")
    deg = int(dm_str[:dot - 2])
    minutes = float(dm_str[dot - 2:])
    dec = deg + minutes / 60.0
    return -dec if hemi in ("S", "W") else dec


def _parse_nmea_line(line: str):
    """One GPGGA/RMC sentence -> (lat, lon), or None. Tries pynmea2 first
    if importable, else the stdlib fallback above -- verified to agree
    with pynmea2 on the same sentences before being relied on here."""
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
    """0.2s loopback-only connect probe; reads one sentence. Returns
    (lat, lon) or None. Never raises -- nmea_sim.py absence is a normal,
    fully-supported state, not an error."""
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


# -----------------------------------------------------------------------------
# Configuration & Layout
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="🧊 PolarNav AI")

# F8 offline map: streamlit-folium loads Leaflet from folium's default CDN
# links, so with Wi-Fi off the map never appears. Point the Map class at the
# vendored copies in ./static (served by Streamlit itself via
# server.enableStaticServing in .streamlit/config.toml). jQuery/Bootstrap/
# awesome-markers are dropped: nothing on this map uses them.
folium.Map.default_js = [("leaflet", "/app/static/leaflet.js")]
folium.Map.default_css = [("leaflet_css", "/app/static/leaflet.css")]

st.title("🧊 PolarNav AI — Edge-Native Antarctic Decision Support")
st.info("Model status: Otsu is active for ice detection (U-Net exists but hasn't cleared its Dice bar); "
        "the Ridge drift model is trained on synthetic physics samples, not observed data.")


# ---- M5: stale-drop banner. session_state may not have last_drop_at yet
# (first-ever page load, before _defaults is applied below) -- .get()
# with a None fallback handles that as a normal no-banner state, not an
# error, matching every other optional-signal check in this file. ----
@st.fragment(run_every="60s")
def _stale_drop_banner():
    last = st.session_state.get("last_drop_at")
    if last is None:
        return
    age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
    if age_h > 12:
        st.warning(f"LAST DROP: {age_h:.0f}h ago — treat as advisory only")


_stale_drop_banner()

# Sidebar — the toggle's value is now actually used (see map tiles below)
# instead of being discarded.
offline_mode = st.sidebar.toggle("Edge-Native Mode (Offline)", value=True)
st.sidebar.caption("Connectivity: OUTAGE SIMULATED" if offline_mode else "Connectivity: ONLINE")
st.sidebar.success("Server: OPERATIONAL")

# ---- M4: Production Mimic status page. Pure reads, no side effects -- ----
# each of the 4 checks is independently wrapped so one failing can never
# take down the other 3 or the page itself.
with st.sidebar.expander("🛰 Production Mimic"):
    st.code(
        "satcom_sim -> drops_in/ -> drop_watcher -> drops_done/ (or drops_quarantine/)\n"
        "                                   |\n"
        "nmea_sim -----------------> app <--+--> bus -> kafka (if reachable)\n"
        "                                          `--> events.log (always)",
        language=None,
    )

    @st.fragment(run_every="2s")
    def _production_mimic_status():
        try:
            receipt_path = os.path.join("drops_done", "latest_receipt.json")
            age_s = time.time() - os.path.getmtime(receipt_path) if os.path.exists(receipt_path) else None
            if age_s is not None and age_s < 60:
                st.success(f"Watcher: fresh ({age_s:.0f}s ago)")
            else:
                st.info("Watcher: no recent validated drop")
        except Exception:
            st.info("Watcher: unknown")

        try:
            mode = bus.transport_mode()
            if mode == "kafka":
                st.success(f"Bus: {mode}")
            else:
                st.info(f"Bus: {mode}")
        except Exception:
            st.info("Bus: unknown")

        try:
            if _nmea_probe() is not None:
                st.success("NMEA: live")
            else:
                st.info("NMEA: pinned (no feed)")
        except Exception:
            st.info("NMEA: unknown")

        try:
            _iv = st.session_state.get("ice_view")
            active = _iv["active_path"] if _iv else "not yet detected"
            st.info(f"Model: {active}")
        except Exception:
            st.info("Model: unknown")

    _production_mimic_status()

# -----------------------------------------------------------------------------
# Session State Initialization — every key any button reads is created up
# front, so pressing buttons out of order never raises a KeyError/AttributeError.
#
# Buttons only ever WRITE here; all rendering happens at page level below. A
# rerun (file-watcher, widget change, second tab) re-runs the script with every
# button False, so anything drawn inside a button branch would vanish.
# -----------------------------------------------------------------------------
_DEFAULT_ICEBERGS = pd.DataFrame(columns=["id", "lat", "lon", "mass_kt", "freeboard_m"])
_DEFAULT_WIND = pd.DataFrame(columns=["lat", "lon", "u_wind", "v_wind", "u_current", "v_current"])
REQUIRED_ICEBERG_COLS = {"id", "lat", "lon", "mass_kt", "freeboard_m"}
REQUIRED_WIND_COLS = {"lat", "lon", "u_wind", "v_wind", "u_current", "v_current"}

_defaults = {
    "sat_data_loaded": False,
    "ice_detected": False,
    "icebergs_df": _DEFAULT_ICEBERGS,
    "wind_df": _DEFAULT_WIND,
    "risk_grid": None,
    "drop_receipts": [],   # F1 receipts, replayed every rerun
    "drop_error": None,
    "seaice_context": None,  # F1: sampled NSIDC {year, month, day, extent} or None
    "ice_view": None,      # F2: {orig, mask, n_cells, source}
    "ice_error": None,
    "route_data": None,    # F5: plain data, never a folium.Map object
    "route_error": None,
    "last_drop_at": None,  # M5: UTC datetime of the last successful F1 (live or static); drives the stale-drop banner
    "last_route_start_ll": None,  # M2: (lat, lon) the current route was computed from; drift-hysteresis baseline
    "nmea_live_start": None,      # M2: (r, c) from the live GPS feed, or None when unreachable (read by M4's status page)
}
for _key, _val in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val.copy() if hasattr(_val, "copy") else _val


def _load_csv_safe(path: str, required_cols: set, empty_df: pd.DataFrame) -> pd.DataFrame:
    """Read a CSV if it exists and has the expected columns; otherwise fall
    back to an empty (but correctly-shaped) frame instead of crashing on a
    missing file, a parse error, or an unexpected schema."""
    if not os.path.exists(path):
        return empty_df.copy()
    try:
        df = pd.read_csv(path)
    except Exception:
        return empty_df.copy()
    if not required_cols.issubset(df.columns):
        return empty_df.copy()
    return df


# -----------------------------------------------------------------------------
# Button 1: Simulate Satellite Data Drop
# -----------------------------------------------------------------------------
if st.button("📡 Simulate Satellite Data Drop"):
    os.makedirs("data", exist_ok=True)

    # M1 (Realtime Core): prefer a live, validated satellite drop over the
    # static data/ files when drop_watcher.py has produced one. Any failure
    # anywhere in this check falls straight through to today's exact
    # existing behavior — a live pipeline is additive, never required.
    _live_receipt = None
    try:
        _receipt_path = os.path.join("drops_done", "latest_receipt.json")
        if os.path.exists(_receipt_path):
            with open(_receipt_path, encoding="utf-8") as _f:
                _candidate = json.load(_f)
            if all(os.path.exists(_candidate.get(_k, "")) for _k in ("sar_path", "icebergs_csv", "wind_csv")):
                _live_receipt = _candidate
    except Exception:
        _live_receipt = None

    if _live_receipt is not None:
        st.session_state.icebergs_df = _load_csv_safe(
            _live_receipt["icebergs_csv"], REQUIRED_ICEBERG_COLS, _DEFAULT_ICEBERGS)
        st.session_state.wind_df = _load_csv_safe(
            _live_receipt["wind_csv"], REQUIRED_WIND_COLS, _DEFAULT_WIND)
    else:
        st.session_state.icebergs_df = _load_csv_safe(
            "data/icebergs.csv", REQUIRED_ICEBERG_COLS, _DEFAULT_ICEBERGS)
        st.session_state.wind_df = _load_csv_safe(
            "data/wind_current.csv", REQUIRED_WIND_COLS, _DEFAULT_WIND)

    sar_path = _live_receipt["sar_path"] if _live_receipt is not None else "data/sar_sample.png"
    try:
        if _live_receipt is None:
            # Physics-informed synthetic SAR: pick a random real Antarctic extent
            # from the NSIDC Sea Ice Index and scale the ice-blob density to it,
            # instead of an arbitrary fixed blob count. Regenerated every press so
            # each data drop reflects a (possibly different) real historical day.
            seaice_row = sample_seaice_row()
            if seaice_row is not None:
                make_synthetic_sar(sar_path, target_extent=seaice_row["extent"])
            elif not os.path.exists(sar_path):
                make_synthetic_sar(sar_path)  # seaice.csv unavailable — pinned default field
        else:
            seaice_row = None  # a live drop's SAR is the sim's own image, not NSIDC-extent-scaled synthetic
        st.session_state.seaice_context = seaice_row

        st.session_state.sat_data_loaded = True
        st.session_state.drop_error = None
        st.session_state.last_drop_at = datetime.now(timezone.utc)
        if _live_receipt is not None:
            st.session_state.drop_receipts = [
                ("success", f"📡 Live drop consumed: {_live_receipt['batch_id']} "
                            f"(validated {_live_receipt['validated_at']})"),
            ]
        else:
            st.session_state.drop_receipts = [
                ("success", "✅ data/icebergs.csv ingested successfully."),
                ("success", "✅ data/wind_current.csv ingested successfully."),
                ("success", f"✅ SAR imagery available at {sar_path}."),
            ]
        if seaice_row is not None:
            st.session_state.drop_receipts.append(
                ("info", f"Context: {seaice_row['year']:04d}-{seaice_row['month']:02d}-"
                         f"{seaice_row['day']:02d} | Extent: {seaice_row['extent']:.1f} M km² (NSIDC)"))
        # An empty frame means the CSV was missing or malformed — say so rather
        # than silently planning around zero icebergs.
        if st.session_state.icebergs_df.empty:
            st.session_state.drop_receipts.append(
                ("info", f"{'live iceberg data' if _live_receipt else 'data/icebergs.csv'} "
                         f"unavailable — using empty iceberg defaults."))
        if st.session_state.wind_df.empty:
            st.session_state.drop_receipts.append(
                ("info", f"{'live wind/current data' if _live_receipt else 'data/wind_current.csv'} "
                         f"unavailable — using empty wind/current defaults."))
    except Exception as e:
        st.session_state.sat_data_loaded = False
        st.session_state.drop_receipts = []
        st.session_state.seaice_context = None
        st.session_state.drop_error = f"Satellite data simulation failed: {e}"

# Page-level replay of the F1 receipts (survives every rerun).
for _kind, _msg in st.session_state.drop_receipts:
    (st.success if _kind == "success" else st.info)(_msg)
if st.session_state.drop_error:
    st.error(st.session_state.drop_error)

# -----------------------------------------------------------------------------
# Button 2: Detect Ice Hazards
# -----------------------------------------------------------------------------
if st.button("🔍 Detect Ice (U-Net / OpenCV Surrogate)"):
    if not st.session_state.sat_data_loaded:
        st.warning("Please simulate satellite data drop first.")
    else:
        sar_path = "data/sar_sample.png"
        try:
            orig_img, mask_img, n_cells, ice_path = detect_ice(sar_path)

            st.session_state.risk_grid = build_risk_grid(
                mask_img, GRID, icebergs_df=st.session_state.icebergs_df)
            st.session_state.ice_detected = True
            st.session_state.ice_error = None

            # Say which imagery the mask came from — a real Sentinel-1 crop
            # dropped into data/, or the bundled synthetic sample — and which
            # inference path actually produced the mask.
            _src = resolve_sar_path(sar_path)
            st.session_state.ice_view = {
                "orig": orig_img,
                "mask": mask_img,
                "n_cells": n_cells,
                "source": ("Source: real Sentinel-1 crop" if _src != sar_path
                           else "Source: synthetic sample"),
                "active_path": ("Active model: SmallUNet (trained weights)" if ice_path == "unet"
                                 else "Active model: OpenCV Otsu (fallback)"),
            }
        except Exception as e:
            st.session_state.ice_detected = False
            st.session_state.ice_view = None
            st.session_state.ice_error = f"Ice detection failed: {e}"

# Page-level replay of the F2 mask + count (survives every rerun).
if st.session_state.ice_view is not None:
    _iv = st.session_state.ice_view
    col1, col2 = st.columns(2)
    with col1:
        st.image(_iv["orig"], caption="Original SAR Imagery")
    with col2:
        st.image(_iv["mask"], caption="Detected Ice Mask")
    st.caption(_iv["source"])
    st.caption(_iv["active_path"])
    st.metric("Ice cells", _iv["n_cells"])
if st.session_state.ice_error:
    st.error(st.session_state.ice_error)

# -----------------------------------------------------------------------------
# Button 3: Predict Drift + Generate Route
# -----------------------------------------------------------------------------
st.markdown("### ⚓ Risk-Aware Routing")
st.caption("🧭 Forecast Drift (Vector Kinematics) — computed together with the route below.")

# Preset grid coordinates for selection
coords_list = [(5, 5), (10, 15), (20, 20), (35, 35)]


def _coord_label(t):
    """Show the grid cell with its real position, e.g. "(5, 5) · -67.125, 59.688".
    Display only — the (r, c) tuple itself is what gets passed to the planner."""
    lat, lon = grid_to_latlon(*t)
    return f"{t} · {lat:.3f}, {lon:.3f}"


# ---- M5: captain-language alerts. engine.py math is off-limits this
# mission, so this mirrors engine.cpa_km's own equirectangular distance
# matrix (same formula, see engine.py's cpa_km) but additionally recovers
# WHICH route waypoint the closest approach happens at, letting us report
# "will cross your route in X km" (distance-to-go, along the route from
# its start) instead of a raw closest-approach distance -- a small but
# real captain-language upgrade (matches how ship radar ARPA reports
# report a target's distance along one's own track, not just its CPA). ----
def _along_route_km(route_ll, track_ll, samples: int = 25):
    """Along-route distance (km) from the route's start to the route
    waypoint nearest the iceberg's track. Returns None on any degenerate
    input (empty route/track, all-NaN distances) -- callers fall back to
    the existing CPA-distance wording rather than showing a broken
    figure; never raises."""
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


def _compute_route(start_coord, goal_coord) -> bool:
    """Shared route computation: the manual button below and M2's
    drift-triggered auto-replan both call this. Precondition (checked by
    each caller, not here, since the right UX on a missing precondition
    differs): st.session_state.ice_detected must already be True.
    Writes route_data/route_error exactly as the original inline button
    body did; returns True on success."""
    try:
        risk_grid = st.session_state.risk_grid
        notes = []

        # Predict iceberg drift (batch, tolerant of empty/missing data)
        pred_df = predict_iceberg_drift(
            st.session_state.icebergs_df, st.session_state.wind_df, hours=24.0)

        drift_list = []     # predicted positions only (feeds the strict JSON)
        iceberg_pairs = []  # (current, predicted) pairs for the drift arrows
        tracks = []         # (id, current, predicted) for the CPA alerts
        for _, row in pred_df.iterrows():
            if pd.isna(row.get('pred_lat')) or pd.isna(row.get('pred_lon')):
                continue
            curr_loc = [row['lat'], row['lon']]
            pred_loc = [row['pred_lat'], row['pred_lon']]
            drift_list.append(pred_loc)
            iceberg_pairs.append((curr_loc, pred_loc))
            _id = row.get('id', '?')
            if isinstance(_id, float) and _id.is_integer():
                _id = int(_id)  # iterrows upcasts int ids to float; show "4", not "4.0"
            tracks.append((_id, curr_loc, pred_loc))

        # 24-h sea-ice concentration forecast: advect today's risk grid by
        # the drift field, max-combine, and plan against the combination so
        # the route avoids ice that will be there tomorrow, not just today.
        drift_field = build_drift_field(st.session_state.wind_df)
        risk_pred = predict_risk_grid(risk_grid, drift_field, hours=24.0)
        risk_combined = np.maximum(risk_grid, risk_pred) if risk_grid is not None else None
        band_edges, band_src = kmeans_band_edges(risk_pred)

        # Pathfinding — fall back to the direct path if A* can't reach the goal
        # (unreachable goal, missing grid, or out-of-bounds coordinates).
        opt_path = astar(risk_combined, start_coord, goal_coord)
        dir_path = direct_path(risk_grid, start_coord, goal_coord)
        if opt_path is None:
            notes.append(("warning",
                          "No A* route found to the goal — showing the direct path instead."))
            opt_path = dir_path

        # Compute metrics (risk_grid first — matches route_metrics' real signature)
        metrics = route_metrics(risk_grid, opt_path, dir_path, predicted_grid=risk_pred)

        # Live alerts (display-only): per-iceberg CPA against its 24-h track
        # drives proximity threat (HIGH/MED/LOW, CPA-only per F3); predicted
        # risk>5 crossings is a SEPARATE, independent advisory line — it
        # never escalates a distant iceberg to HIGH by itself. A HIGH
        # proximity threat asks for a reroute SUGGESTION — the shown route
        # never auto-switches; the captain decides.
        route_ll = [grid_to_latlon(r, c) for r, c in opt_path]
        cpas = [(ib_id, cpa_km(route_ll, [curr_loc, pred_loc]), curr_loc, pred_loc)
                for ib_id, curr_loc, pred_loc in tracks]
        min_cpa = min((c for _, c, _, _ in cpas), default=float('inf'))
        pred_x = metrics['predicted_crossings']
        threat = classify_threat(min_cpa, 0)  # proximity-only; predicted_crossings no longer affects this
        reroute = suggest_reroute(risk_combined, start_coord, goal_coord,
                                  opt_path, risk_pred) if threat == "HIGH" else None
        suggestion = (f"suggested deviation +{reroute[1]:.1f} km" if reroute
                      else "no lower-exposure alternative found")
        alerts = []
        for ib_id, cpa, curr_loc, pred_loc in sorted(cpas, key=lambda t: t[1]):
            lvl = classify_threat(cpa, 0)  # proximity class of this iceberg alone
            if lvl not in ("HIGH", "MED"):
                continue
            along = _along_route_km(route_ll, [curr_loc, pred_loc])
            if lvl == "HIGH":
                if along is not None:
                    alerts.append(("error", f"HIGH: {ib_id} will cross your route in {along:.1f} km — {suggestion}"))
                else:
                    alerts.append(("error", f"HIGH: Iceberg {ib_id} CPA {cpa:.1f} km — {suggestion}"))
            elif lvl == "MED":
                if along is not None:
                    alerts.append(("warning", f"MED: {ib_id} passes near your route in {along:.1f} km — monitor"))
                else:
                    alerts.append(("warning", f"MED: route passes within {cpa:.1f} km of {ib_id} drift corridor"))
        if pred_x > 0:
            alerts.append(("warning", f"MED: Route crosses {pred_x} predicted risk>5 cells - expect icebreaking"))
        if not alerts:
            alerts.append(("info", "LOW: corridor clear for 24 h"))
        alerts.append(("info", f"Predicted risk>5 cells on route: {pred_x}"))

        start_ll = grid_to_latlon(*start_coord)
        goal_ll = grid_to_latlon(*goal_coord)

        saved = save_route(
            start=start_ll, goal=goal_ll,
            distance_km=metrics['path_distance_km'],
            risk_red=metrics['risk_reduction_pct'],
            path_json=[grid_to_latlon(r, c) for r, c in opt_path],
        )
        if not saved:
            notes.append(("info",
                          "Route computed, but the local route log couldn't be written this run."))

        # Store DATA, never the folium.Map — the map is rebuilt fresh at
        # page level on every run from exactly this dict.
        st.session_state.route_data = {
            "path": opt_path,
            "direct": dir_path,
            "drift_list": drift_list,
            "metrics": metrics,
            "icebergs": iceberg_pairs,
            "start_ll": start_ll,
            "goal_ll": goal_ll,
            "notes": notes,
            "risk_pred": risk_pred,        # 24-h forecast grid (overlay)
            "band_edges": band_edges,      # concentration band edges, %
            "band_src": band_src,          # "kmeans" | "fixed"
            "alerts": alerts,              # [(kind, text)] replayed every rerun
        }
        st.session_state.route_error = None

        # M3: publish, display-only observability. bus.publish() never
        # raises (kafka if reachable, always also events.log) so a bus
        # hiccup can never take the route computation itself down.
        bus.publish("route.computed", {
            "start_ll": start_ll, "goal_ll": goal_ll,
            "distance_km": metrics["path_distance_km"],
            "risk_reduction_pct": metrics["risk_reduction_pct"],
        })
        bus.publish("alerts.raised", {"alerts": [{"level": k, "text": m} for k, m in alerts]})

        return True
    except Exception as e:
        st.session_state.route_error = f"Route generation failed: {e}"
        return False


def _km_between(ll1, ll2) -> float:
    """Quick equirectangular distance in km for the 2 km drift-hysteresis
    check only -- UI-side, not core routing math (engine.py untouched)."""
    lat1, lon1 = ll1
    lat2, lon2 = ll2
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2.0))
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * cos_lat
    return math.hypot(dlat, dlon)


@st.fragment(run_every="1s")
def _position_badge():
    pos = _nmea_probe()
    if pos is not None:
        lat, lon = pos
        r, c = latlon_to_grid(lat, lon)
        st.session_state.nmea_live_start = (r, c)
        st.success(f"🛰 LIVE POS: ({r}, {c}) · {lat:.3f}, {lon:.3f} — auto-replans past 2 km drift "
                   f"(manual Start dropdown below still works independently)")
        goal = st.session_state.get("goal_coord_select", coords_list[-1])
        if (st.session_state.ice_detected and st.session_state.last_route_start_ll is not None
                and _km_between(st.session_state.last_route_start_ll, (lat, lon)) > 2.0):
            if _compute_route((r, c), goal):
                st.session_state.last_route_start_ll = (lat, lon)
                # The map/metrics/alerts below live OUTSIDE this fragment's own
                # render scope, so a fragment-only rerun would update
                # route_data silently without the visible page catching up.
                # scope="app" (the default) forces the full page to re-render
                # with the new route -- documented Streamlit pattern for a
                # fragment whose state change affects content outside itself.
                st.rerun()
    else:
        st.session_state.nmea_live_start = None
        st.caption("📍 PINNED — no live GPS feed (nmea_sim.py not reachable on 127.0.0.1:10110); using the manual dropdown below")


_position_badge()

col_s, col_g = st.columns(2)
start_coord = col_s.selectbox("Start Grid Coordinate", coords_list, index=0,
                               format_func=_coord_label, key="start_coord_select")
goal_coord = col_g.selectbox("Goal Grid Coordinate", coords_list, index=3,
                              format_func=_coord_label, key="goal_coord_select")

if st.button("⚓ Compute Risk-Aware Route (Modified A* + p(n))"):
    if not st.session_state.ice_detected:
        st.warning("Please detect ice hazards first.")
    else:
        if _compute_route(start_coord, goal_coord):
            st.session_state.last_route_start_ll = grid_to_latlon(*start_coord)  # M2: drift-hysteresis baseline


# Ice-blue ramp, one colour per concentration band (light -> deep).
_BAND_RGB = [(198, 226, 255), (140, 196, 250), (86, 152, 236), (44, 98, 208), (16, 44, 128)]


def _concentration_rgba(risk_pred, edges) -> np.ndarray:
    """Predicted risk grid (0..10) -> RGBA uint8 image banded by concentration
    (%); open water (0 %) stays transparent. Pure numpy: folium base64-inlines
    the array as a PNG, so the overlay needs no file, no CDN, no matplotlib."""
    conc = np.clip(np.asarray(risk_pred, dtype=np.float64) * 10.0, 0.0, 100.0)
    band = np.digitize(conc, edges)  # 0..len(edges)
    rgba = np.zeros(conc.shape + (4,), dtype=np.uint8)
    for k, rgb in enumerate(_BAND_RGB):
        rgba[band == k, :3] = rgb
    rgba[..., 3] = np.where(conc > 0.0, 255, 0).astype(np.uint8)
    return rgba


def _build_map(rd: dict) -> folium.Map:
    """Build a brand-new folium.Map from plain route data. Called once per
    script run — a Map object is never cached or reused across reruns."""
    # True offline mode: tiles=None means the browser never requests basemap
    # images from any CDN. A flat rectangle stands in for the ocean, and the
    # bundled schematic coastline gives it geography.
    m = folium.Map(
        location=[-67.5, 60.2],
        zoom_start=8,
        tiles=None,
    )
    folium.Rectangle(
        bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        color="#1b2a4a", fill=True, fill_opacity=0.6, weight=0,
    ).add_to(m)
    coast_path = "data/coast.geojson"
    if os.path.exists(coast_path):  # bundled locally; folium inlines it, no fetch
        folium.GeoJson(
            coast_path, name="Coastline",
            style_function=lambda f: {"color": "#9aa4b2", "weight": 1.5, "fillOpacity": 0},
        ).add_to(m)

    # 24-h predicted ice concentration (GIS overlay). Row 0 of the grid is the
    # northern edge (see grid_to_latlon), matching origin="upper".
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

    # Map grid coords to lat/lon for mapping
    opt_latlon = [grid_to_latlon(r, c) for r, c in rd["path"]]
    dir_latlon = [grid_to_latlon(r, c) for r, c in rd["direct"]]

    folium.PolyLine(opt_latlon, color='green', weight=4, opacity=0.8).add_to(m)
    folium.PolyLine(dir_latlon, color='red', dash_array='10', weight=2, opacity=0.6).add_to(m)
    return m


# Page-level render: outside every button, so a rerun redraws the same map
# instead of dropping it. This is the app's only map-component call site, and
# its fixed key keeps the component's identity stable across reruns.
_rd = st.session_state.route_data
if _rd is not None:
    for _kind, _msg in _rd["notes"]:
        (st.warning if _kind == "warning" else st.info)(_msg)

    _metrics = _rd["metrics"]
    # Use the metric keys route_metrics() actually returns.
    c1, c2, c3, c4, c5 = st.columns(5)
    risk_diff = _metrics['path_risk_score'] - _metrics['direct_risk_score']
    c1.metric("Risk Exposure", round(_metrics['path_risk_score'], 2),
               delta=f"{risk_diff:.2f} (vs Direct)", delta_color="inverse")
    c2.metric("Distance (km)", round(_metrics['path_distance_km'], 1))
    c3.metric("Ice Crossings", _metrics['path_crossings'])
    c4.metric("Risk Reduction %", round(_metrics['risk_reduction_pct'], 1))
    c5.metric("Fuel/Time Trade-off", f"{round(_metrics['fuel_penalty_pct'], 1)}%")

    st_folium(_build_map(_rd), width=1200, height=500,
              returned_objects=[], key="nav_map")

    st.caption("Forecast horizon: 24 h | overlay = predicted concentration")
    if _rd.get("risk_pred") is not None:
        _e = _rd.get("band_edges") or [20, 40, 60, 80]
        _lo = [0] + list(_e)
        _hi = list(_e) + [100]
        _sw = "".join(
            f'<span style="display:inline-block;width:12px;height:12px;background:rgb{_BAND_RGB[i]};'
            f'border:1px solid #888;margin:0 4px 0 10px;vertical-align:middle"></span>'
            f'{_lo[i]:.0f}–{_hi[i]:.0f} %'
            for i in range(5))
        _src = ("bands: KMeans ice classes" if _rd.get("band_src") == "kmeans"
                else "bands: fixed 20 % steps (KMeans unavailable)")
        st.markdown(f'<div style="font-size:0.85em;color:#888">Ice concentration legend:{_sw}'
                    f' &nbsp;|&nbsp; {_src}</div>', unsafe_allow_html=True)

    # PS-mandated notification centre. Display-only: replayed from session
    # state on every rerun, never switches the shown route.
    st.markdown("### ⚠ Live Alerts (24-h horizon)")
    for _kind, _msg in _rd.get("alerts", []):
        {"error": st.error, "warning": st.warning}.get(_kind, st.info)(_msg)

    st.info("🧑‍✈️ Human-in-the-loop: AI recommends the safest path, captain retains final authority.")

    _iv2 = st.session_state.ice_view
    if _iv2 is not None:
        _active_txt = "U-Net" if "SmallUNet" in _iv2["active_path"] else "Otsu"
        st.caption(f"Data Source: NSIDC Sea Ice Index (Surrogate) | "
                   f"Model: Notebook U-Net Architecture | Active Path: {_active_txt}")
if st.session_state.route_error:
    st.error(st.session_state.route_error)

# -----------------------------------------------------------------------------
# Route History & JSON Exporter
# -----------------------------------------------------------------------------
st.markdown("### 🗂 Route History")
routes = load_routes()  # never raises — returns [] if nothing's been saved yet
if routes:
    st.dataframe(pd.DataFrame(routes), use_container_width=True)
else:
    st.info("No route history found locally.")

with st.expander("📋 Strict JSON — vessel-API-ready payload schema (transport = Phase 2)"):
    if _rd is not None:
        json_output = strict_json(
            _rd["start_ll"],
            _rd["goal_ll"],
            [grid_to_latlon(r, c) for r, c in _rd["path"]],
            _rd["metrics"],
            _rd["drift_list"],
        )
        payload_str = json.dumps(json_output, indent=2)
        st.code(payload_str, language="json")

        # F5: file-drop so the payload is inspectable outside the browser too.
        # sha256 is computed over the payload alone (before the field is added)
        # so the checksum isn't self-referential.
        export_dir = "routes_out"
        export_path = os.path.join(export_dir, "latest.json")
        checksum = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        try:
            os.makedirs(export_dir, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump({**json_output, "sha256": checksum}, f, indent=2)
            st.caption(f"Exported to `{export_path}` · sha256 `{checksum[:12]}…`")
        except OSError as e:
            st.caption(f"Export to {export_path} failed: {e}")
    else:
        st.write("Generate a route to preview the API payload.")

with st.expander("📡 Event Stream (last 20)"):
    st.caption(f"Transport: {bus.transport_mode()} — always also appended to {bus.EVENTS_LOG} "
               f"regardless of transport, so this tail works either way.")
    _events = bus.tail(20)
    if _events:
        for _ev in _events:
            st.text(f"{_ev.get('ts', '?')}  {_ev.get('topic', '?')}  {json.dumps(_ev.get('payload', {}))}")
    else:
        st.write("No events yet — compute a route to publish route.computed and alerts.raised.")
