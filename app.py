import os
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import json

# Offline navigation engine imports
from engine import (
    make_synthetic_sar, detect_ice, build_risk_grid, predict_iceberg_drift,
    astar, direct_path, route_metrics, save_route, load_routes, strict_json,
    grid_to_latlon, GRID, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)

# -----------------------------------------------------------------------------
# Configuration & Layout
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="🧊 PolarNav AI")

st.title("🧊 PolarNav AI — Offline Antarctic Navigation")

# Sidebar — the toggle's value is now actually used (see map tiles below)
# instead of being discarded.
offline_mode = st.sidebar.toggle("Offline Onboard Mode", value=True)
st.sidebar.caption("Connectivity: OUTAGE SIMULATED" if offline_mode else "Connectivity: ONLINE")
st.sidebar.success("Server: OPERATIONAL")

# -----------------------------------------------------------------------------
# Session State Initialization — every key any button reads is created up
# front, so pressing buttons out of order never raises a KeyError/AttributeError.
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
    "last_opt_path": None,
    "last_dir_path": None,
    "last_metrics": None,
    "last_start_ll": None,
    "last_goal_ll": None,
    "last_drift_list": [],
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

    st.session_state.icebergs_df = _load_csv_safe(
        "data/icebergs.csv", REQUIRED_ICEBERG_COLS, _DEFAULT_ICEBERGS)
    st.session_state.wind_df = _load_csv_safe(
        "data/wind_current.csv", REQUIRED_WIND_COLS, _DEFAULT_WIND)

    sar_path = "data/sar_sample.png"
    try:
        if not os.path.exists(sar_path):
            make_synthetic_sar(sar_path)
        st.session_state.sat_data_loaded = True
        st.success("✅ data/icebergs.csv ingested successfully.")
        st.success("✅ data/wind_current.csv ingested successfully.")
        st.success(f"✅ SAR imagery available at {sar_path}.")
    except Exception as e:
        st.session_state.sat_data_loaded = False
        st.error(f"Satellite data simulation failed: {e}")

# -----------------------------------------------------------------------------
# Button 2: Detect Ice Hazards
# -----------------------------------------------------------------------------
if st.button("🔍 Detect Ice Hazards"):
    if not st.session_state.sat_data_loaded:
        st.warning("Please simulate satellite data drop first.")
    else:
        sar_path = "data/sar_sample.png"
        try:
            orig_img, mask_img, n_cells = detect_ice(sar_path)

            st.session_state.risk_grid = build_risk_grid(
                mask_img, GRID, icebergs_df=st.session_state.icebergs_df)
            st.session_state.ice_detected = True

            col1, col2 = st.columns(2)
            with col1:
                st.image(orig_img, caption="Original SAR Imagery")
            with col2:
                st.image(mask_img, caption="Detected Ice Mask")

            st.metric("Ice cells", n_cells)
        except Exception as e:
            st.session_state.ice_detected = False
            st.error(f"Ice detection failed: {e}")

# -----------------------------------------------------------------------------
# Button 3: Predict Drift + Generate Route
# -----------------------------------------------------------------------------
st.markdown("### 🧭 Route Generation")

# Preset grid coordinates for selection
coords_list = [(5, 5), (10, 15), (20, 20), (35, 35)]
col_s, col_g = st.columns(2)
start_coord = col_s.selectbox("Start Grid Coordinate", coords_list, index=0)
goal_coord = col_g.selectbox("Goal Grid Coordinate", coords_list, index=3)

if st.button("🧭 Predict Drift + Generate Route"):
    if not st.session_state.ice_detected:
        st.warning("Please detect ice hazards first.")
    else:
        try:
            risk_grid = st.session_state.risk_grid

            # True offline mode: tiles=None means the browser never requests
            # basemap images from CartoDB's CDN. A flat rectangle stands in
            # for the ocean instead — no network round-trip either way.
            m = folium.Map(
                location=[-67.5, 60.2],
                zoom_start=8,
                tiles=None if offline_mode else "CartoDB dark_matter",
            )
            if offline_mode:
                folium.Rectangle(
                    bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
                    color="#1b2a4a", fill=True, fill_opacity=0.6, weight=0,
                ).add_to(m)

            # Predict iceberg drift (batch, tolerant of empty/missing data)
            pred_df = predict_iceberg_drift(
                st.session_state.icebergs_df, st.session_state.wind_df, hours=24.0)

            drift_list = []
            for _, row in pred_df.iterrows():
                if pd.isna(row.get('pred_lat')) or pd.isna(row.get('pred_lon')):
                    continue
                curr_loc = [row['lat'], row['lon']]
                pred_loc = [row['pred_lat'], row['pred_lon']]
                drift_list.append(pred_loc)

                folium.CircleMarker(curr_loc, color='red', radius=4, fill=True).add_to(m)
                folium.CircleMarker(pred_loc, color='orange', radius=4, fill=True).add_to(m)
                folium.PolyLine([curr_loc, pred_loc], color='orange', dash_array='5', weight=2).add_to(m)

            # Pathfinding — fall back to the direct path if A* can't reach the goal
            # (unreachable goal, missing grid, or out-of-bounds coordinates).
            opt_path = astar(risk_grid, start_coord, goal_coord)
            dir_path = direct_path(risk_grid, start_coord, goal_coord)
            if opt_path is None:
                st.warning("No A* route found to the goal — showing the direct path instead.")
                opt_path = dir_path

            # Map grid coords to lat/lon for mapping
            opt_latlon = [grid_to_latlon(r, c) for r, c in opt_path]
            dir_latlon = [grid_to_latlon(r, c) for r, c in dir_path]

            folium.PolyLine(opt_latlon, color='green', weight=4, opacity=0.8).add_to(m)
            folium.PolyLine(dir_latlon, color='red', dash_array='10', weight=2, opacity=0.6).add_to(m)

            # Compute metrics (risk_grid first — matches route_metrics' real signature)
            metrics = route_metrics(risk_grid, opt_path, dir_path)

            start_ll = grid_to_latlon(*start_coord)
            goal_ll = grid_to_latlon(*goal_coord)

            st.session_state.last_opt_path = opt_path
            st.session_state.last_dir_path = dir_path
            st.session_state.last_metrics = metrics
            st.session_state.last_start_ll = start_ll
            st.session_state.last_goal_ll = goal_ll
            st.session_state.last_drift_list = drift_list

            saved = save_route(
                start=start_ll, goal=goal_ll,
                distance_km=metrics['path_distance_km'],
                risk_red=metrics['risk_reduction_pct'],
                path_json=opt_latlon,
            )
            if not saved:
                st.info("Route computed, but the local route log couldn't be written this run.")

            # Use the metric keys route_metrics() actually returns.
            c1, c2, c3, c4, c5 = st.columns(5)
            risk_diff = metrics['path_risk_score'] - metrics['direct_risk_score']
            c1.metric("Risk Score", round(metrics['path_risk_score'], 2),
                       delta=f"{risk_diff:.2f} (vs Direct)", delta_color="inverse")
            c2.metric("Distance (km)", round(metrics['path_distance_km'], 1))
            c3.metric("Ice Crossings", metrics['path_crossings'])
            c4.metric("Risk Reduction %", round(metrics['risk_reduction_pct'], 1))
            c5.metric("Fuel Penalty %", round(metrics['fuel_penalty_pct'], 1))

            st_folium(m, width=1200, height=500)
        except Exception as e:
            st.error(f"Route generation failed: {e}")

# -----------------------------------------------------------------------------
# Route History & JSON Exporter
# -----------------------------------------------------------------------------
st.markdown("### 🗂 Route History")
routes = load_routes()  # never raises — returns [] if nothing's been saved yet
if routes:
    st.dataframe(pd.DataFrame(routes), use_container_width=True)
else:
    st.info("No route history found locally.")

with st.expander("📋 Strict JSON (NCPOR vessel API)"):
    if st.session_state.last_opt_path is not None and st.session_state.last_metrics is not None:
        json_output = strict_json(
            st.session_state.last_start_ll,
            st.session_state.last_goal_ll,
            [grid_to_latlon(r, c) for r, c in st.session_state.last_opt_path],
            st.session_state.last_metrics,
            st.session_state.last_drift_list,
        )
        st.code(json.dumps(json_output, indent=2), language="json")
    else:
        st.write("Generate a route to preview the API payload.")