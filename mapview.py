"""
mapview.py - Shared folium map builder for both clients (SIH26059, v3).

Both app.py (Streamlit) and api.py (the console) need the exact same map:
same AOI rectangle, same coastline, same risk overlay, same route/iceberg
rendering. Before this mission each file carried its own copy (confirmed
byte-for-byte identical except one intentional divergence: the two
processes serve the vendored Leaflet files from different static-mount
prefixes). This module is that one shared copy -- build_map()'s only
required parameter beyond the route data is leaflet_prefix, so each
caller points it at its own static mount.

No new math: every value drawn here (route paths, iceberg positions, risk
grid) is already computed by engine.py/the caller before build_map() is
invoked. This file is presentation only.
"""
import base64
import os
from typing import Optional

import cv2
import numpy as np
import folium

# Ice-blue ramp, one colour per concentration band (light -> deep). Public
# (no leading underscore): app.py's own concentration-legend swatches use
# this directly, not just this module's internal rendering.
BAND_RGB = [(198, 226, 255), (140, 196, 250), (86, 152, 236), (44, 98, 208), (16, 44, 128)]

_MARKER_ICON_FIX = (
    "L.Icon.Default.prototype._getIconUrl = function(){"
    "return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';"
    "};"
)

_DARK_LEAFLET_CSS = (
    "<style>"
    ".leaflet-container{background:#0b1526 !important;}"
    ".leaflet-bar{background:#101c30 !important;border:1px solid #223252 !important;box-shadow:none !important;}"
    ".leaflet-bar a{background:#101c30 !important;color:#93a2bd !important;border-bottom:1px solid #223252 !important;}"
    ".leaflet-bar a:hover{background:#142544 !important;color:#eaf1fb !important;}"
    ".leaflet-control-attribution{background:rgba(16,28,48,0.8) !important;color:#93a2bd !important;}"
    ".leaflet-control-attribution a{color:#7fd4ff !important;}"
    "</style>"
)


def concentration_rgba(risk_pred, edges) -> np.ndarray:
    """Predicted risk grid (0..10) -> RGBA uint8 image banded by
    concentration (%); open water (0%) stays transparent. Pure numpy:
    folium base64-inlines the array as a PNG, so the overlay needs no
    file, no CDN, no matplotlib."""
    conc = np.clip(np.asarray(risk_pred, dtype=np.float64) * 10.0, 0.0, 100.0)
    band = np.digitize(conc, edges)
    rgba = np.zeros(conc.shape + (4,), dtype=np.uint8)
    for k, rgb in enumerate(BAND_RGB):
        rgba[band == k, :3] = rgb
    rgba[..., 3] = np.where(conc > 0.0, 255, 0).astype(np.uint8)
    return rgba


def grid_to_svg_heatmap(grid: Optional[np.ndarray], edges: list) -> Optional[str]:
    """Formats a numeric risk grid as an inline SVG heatmap -- an <svg>
    wrapper around one embedded PNG texture (reusing concentration_rgba's
    same band coloring), not thousands of individual <rect> elements, so
    rendering several of these per request stays fast. Presentation only."""
    if grid is None:
        return None
    try:
        rgba = concentration_rgba(grid, edges)
        bgr = cv2.cvtColor(rgba[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        h, w = grid.shape
        return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
                f'preserveAspectRatio="none" style="image-rendering:pixelated">'
                f'<image href="data:image/png;base64,{b64}" width="{w}" height="{h}"/></svg>')
    except Exception:
        return None


def build_map(rd: Optional[dict], *, leaflet_prefix: str,
              lat_min: float, lat_max: float, lon_min: float, lon_max: float,
              grid_to_latlon) -> folium.Map:
    """Builds the navigation map both clients show. rd is the same plain
    route-data dict both app.py's session_state.route_data and api.py's
    _state["route_data"] already use ({path, direct, icebergs, risk_pred,
    band_edges, reroute_path}), or None for the empty-state AOI-only map
    (no route computed yet). leaflet_prefix is the ONE real difference
    between callers -- Streamlit serves the vendored Leaflet files at
    /app/static/, the console's own uvicorn process at /static/ -- set via
    folium.Map.default_js/default_css by the caller *before* calling this
    (that override is a process-wide class attribute, so it belongs to
    each app's own startup, not here).

    fit_bounds (instead of a fixed zoom) plus a dark re-theme of Leaflet's
    own chrome fix the map's worst-rated slop-audit findings this mission:
    the AOI sitting in a narrow band inside light-gray voids, and a stock
    white zoom control.

    fit_bounds is called ONCE, at the end, after every overlay/marker/
    rectangle has been added -- not before, as an earlier version of this
    function did. Confirmed live (UI audit U1) that computing the fit
    before the AOI rectangle/overlays exist let the console's own iframe
    box (a fixed aspect ratio picked independently of the AOI's true
    shape) letterbox the map, so the AOI rendered as a fragment inside a
    larger dark canvas rather than filling the frame. Calling fit_bounds
    last, against the same aoi_bounds every overlay already used, is the
    simple, robust fix -- Leaflet fits the view to the real content's
    extent instead of a box picked in advance. A small pixel padding keeps
    the AOI's own edge from touching the frame border exactly.
    """
    aoi_bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], tiles=None)
    m.get_root().header.add_child(folium.Element(_DARK_LEAFLET_CSS))
    # Every marker below uses a custom DivIcon; Leaflet's L.Icon.Default
    # still resolves its stock marker-icon.png/marker-shadow.png paths
    # internally regardless (confirmed: a same-origin 404 to
    # <leaflet_prefix>/images/marker-icon.png with nothing in this file
    # requesting it) -- harmless, but cheap to silence with a 1x1
    # transparent placeholder. Added to .script, not .header: confirmed by
    # inspecting rendered output that .script content lands after the
    # leaflet.js <script> tag, so `L` already exists (.header lands
    # before it, which is why the CSS above needs !important instead).
    m.get_root().script.add_child(folium.Element(_MARKER_ICON_FIX))

    folium.Rectangle(
        bounds=aoi_bounds,
        color="#1b2a4a", fill=True, fill_opacity=0.6, weight=0,
    ).add_to(m)
    coast_path = "data/coast.geojson"
    if os.path.exists(coast_path):
        folium.GeoJson(
            coast_path, name="Coastline",
            style_function=lambda f: {"color": "#9aa4b2", "weight": 1.5, "fillOpacity": 0},
        ).add_to(m)

    if rd is not None:
        # Stage 2 frame: ice just detected, no drift predicted yet -- a plain
        # tint from the raw (non-predicted) risk grid + undrifted iceberg dots.
        # Superseded by the richer Stage-3 blocks below once forecast has run
        # (guarded by "not already have icebergs pairs" so both never draw at
        # once).
        if rd.get("risk_grid") is not None and not rd.get("icebergs"):
            overlay_bounds = aoi_bounds
            assert overlay_bounds == aoi_bounds, "ice-concentration overlay must cover exactly the AOI box"
            folium.raster_layers.ImageOverlay(
                image=concentration_rgba(rd["risk_grid"], rd.get("band_edges") or [20, 40, 60, 80]),
                bounds=overlay_bounds,
                opacity=0.30, origin="upper", mercator_project=False,
                name="Detected ice concentration",
            ).add_to(m)
            for lat, lon in rd.get("icebergs_current") or []:
                folium.CircleMarker([lat, lon], color='#ff5c5c', radius=5, fill=True, fill_opacity=0.9,
                                     tooltip="iceberg (detected)").add_to(m)

        # Stage 3 frame: forecast has run -- predicted overlay + drift arrows.
        if rd.get("risk_pred") is not None:
            overlay_bounds = aoi_bounds
            assert overlay_bounds == aoi_bounds, "predicted-concentration overlay must cover exactly the AOI box"
            folium.raster_layers.ImageOverlay(
                image=concentration_rgba(rd["risk_pred"], rd.get("band_edges") or [20, 40, 60, 80]),
                bounds=overlay_bounds,
                opacity=0.45, origin="upper", mercator_project=False,
                name="Predicted ice concentration (24 h)",
            ).add_to(m)

        for curr_loc, pred_loc in rd.get("icebergs") or []:
            folium.CircleMarker(curr_loc, color='#ff5c5c', radius=5, fill=True, fill_opacity=0.9,
                                 tooltip="iceberg (now)").add_to(m)
            folium.CircleMarker(pred_loc, color='#f59e0b', radius=5, fill=True, fill_opacity=0.9,
                                 tooltip="iceberg (+24h)").add_to(m)
            folium.PolyLine([curr_loc, pred_loc], color='#f59e0b', dash_array='5', weight=2).add_to(m)

        # Stage 4 frame: route computed -- pins + optimized/direct/reroute lines.
        if rd.get("path") and rd.get("direct"):
            opt_latlon = [grid_to_latlon(r, c) for r, c in rd["path"]]
            dir_latlon = [grid_to_latlon(r, c) for r, c in rd["direct"]]
            # Casing + glow: a wide, translucent line under a crisp core line, so
            # the route reads as the clear focal element instead of a thin stroke.
            folium.PolyLine(opt_latlon, color='#22c55e', weight=12, opacity=0.22).add_to(m)
            folium.PolyLine(opt_latlon, color='#22c55e', weight=6, opacity=1.0).add_to(m)
            folium.PolyLine(dir_latlon, color='#ff5c5c', dash_array='10 10', weight=4, opacity=0.6).add_to(m)
            if rd.get("reroute_path"):
                reroute_latlon = [grid_to_latlon(r, c) for r, c in rd["reroute_path"]]
                folium.PolyLine(reroute_latlon, color='#9aa4b2', dash_array='4 6', weight=3, opacity=0.8).add_to(m)

            start_icon = folium.DivIcon(html=(
                '<div style="width:22px;height:22px;border-radius:50%;background:#7fd4ff;'
                'border:2px solid #0b1526;box-shadow:0 0 0 4px rgba(127,212,255,0.25);"></div>'))
            goal_icon = folium.DivIcon(html=(
                '<div style="width:0;height:0;border-left:11px solid transparent;'
                'border-right:11px solid transparent;border-bottom:18px solid #4ade80;"></div>'))
            if opt_latlon:
                folium.Marker(opt_latlon[0], icon=start_icon, tooltip="Start / ship").add_to(m)
                folium.Marker(opt_latlon[-1], icon=goal_icon, tooltip="Goal / station").add_to(m)

    # fit_bounds LAST, after every overlay/marker above -- the actual fix
    # for U1 (map fragment/letterboxing). Small padding keeps the AOI's own
    # edge off the frame border.
    m.fit_bounds(aoi_bounds, padding=(10, 10))
    return m
