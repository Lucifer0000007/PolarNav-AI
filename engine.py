"""
engine.py - Offline Antarctic navigation prototype (SIH26059)
Core: numpy, opencv-python, pandas, and Python standard library.
Optional ML core (SmallUNet segmentation, Ridge drift, KMeans ice-class
profiling) loads only if torch/scikit-learn are installed and trained
weights are present; every ML path falls back to the original OpenCV/
heuristic behavior otherwise. No network calls, no API keys.
"""

import numpy as np
import pandas as pd
import cv2
import sqlite3
import json
import math
import heapq
import os
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

# Constants
GRID = 40
KM_PER_CELL = 111.0 / 40.0          # ~2.775 km per cell at 1° latitude
LAT_MIN, LAT_MAX = -68.0, -67.0
LON_MIN, LON_MAX = 59.5, 61.0

# Demo scenario: pinned ice field + the start/goal the app defaults to, tuned so
# the direct route runs through ice and A* buys a visible risk reduction.
DEMO_SEED = 13

# Hoisted A* constants (see astar)
_SQRT2 = math.sqrt(2.0)
_INF = float('inf')
DEMO_START, DEMO_GOAL = (5, 5), (35, 35)


# ----------------------------------------------------------------------
# Grid <-> lat/lon conversion
def grid_to_latlon(r: int, c: int) -> Tuple[float, float]:
    """
    Map grid cell (r,c) to geographic coordinates.
    r=0 -> lat=-67 (north), r=39 -> lat≈-67.975 (south)
    c=0 -> lon=59.5, c=39 -> lon≈60.9625 (east)
    """
    lat = LAT_MIN + 1.0 * (1.0 - r / 40.0)
    lon = LON_MIN + (LON_MAX - LON_MIN) * (c / 40.0)
    return lat, lon


def latlon_to_grid(lat: float, lon: float, size: int = GRID) -> Tuple[int, int]:
    """
    Inverse of grid_to_latlon, clamped to the grid. A stray CSV row with a
    coordinate outside [LAT_MIN,LAT_MAX]/[LON_MIN,LON_MAX] lands on the
    nearest edge cell instead of raising or producing a negative index.
    """
    r = size * (1.0 - (lat - LAT_MIN))
    lon_span = (LON_MAX - LON_MIN) or 1.0  # guard: never divide by zero
    c = size * (lon - LON_MIN) / lon_span
    r = max(0, min(size - 1, int(round(r))))
    c = max(0, min(size - 1, int(round(c))))
    return r, c


# ----------------------------------------------------------------------
# Real-SAR source resolution
SAR_REAL_PATHS = ("data/sar_real.png", "data/sar_real.tif")


def resolve_sar_path(path: str = "data/sar_sample.png") -> str:
    """
    Prefer a real Sentinel-1 crop dropped into data/ (sar_real.png/.tif);
    fall back to the synthetic sample when none has been supplied.
    """
    for p in SAR_REAL_PATHS:
        if os.path.exists(p):
            return p
    return path


# ----------------------------------------------------------------------
# 1. Generate synthetic SAR image (grayscale, uint8)
def make_synthetic_sar(path: str = "data/sar_sample.png", size: int = 400,
                       seed: Optional[int] = DEMO_SEED) -> str:
    """
    Create a synthetic SAR image with:
      - dark ocean background (10–40)
      - 6–10 bright elliptical ice blobs (150–255)
      - speckle noise (multiplicative)
    Save as PNG and return the path.

    seed pins the ice field so the demo scenario is repeatable run to run;
    pass seed=None for a fresh random field.
    """
    if seed is not None:
        np.random.seed(seed)

    dirpath = os.path.dirname(path)
    if dirpath:  # os.makedirs("") raises FileNotFoundError, so only call it when there IS a dir
        os.makedirs(dirpath, exist_ok=True)

    img = np.random.randint(10, 41, (size, size), dtype=np.uint8)
    n_blobs = np.random.randint(6, 11)

    for _ in range(n_blobs):
        center = (np.random.randint(0, size), np.random.randint(0, size))
        axes = (np.random.randint(15, 60), np.random.randint(10, 40))
        angle = np.random.randint(0, 180)
        brightness = np.random.randint(150, 256)

        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

        inds = mask == 255
        blob_vals = brightness + np.random.randint(-20, 21, size=int(np.sum(inds)))
        blob_vals = np.clip(blob_vals, 0, 255).astype(np.uint8)
        img[inds] = blob_vals

    noise = np.random.normal(0, 0.08, img.shape).astype(np.float32)
    img_float = img.astype(np.float32)
    img_noisy = img_float + img_float * noise
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

    cv2.imwrite(path, img_noisy)
    return path


# ----------------------------------------------------------------------
# ML core: SmallUNet segmentation (optional, CPU-only inference).
# Otsu (below) is the permanent fallback — never removed, always reachable.
UNET_WEIGHTS_PATH = "unet_weights.pth"

if torch is not None:
    class SmallUNet(nn.Module):
        """3-level U-Net, base width 16 — small enough for CPU inference."""

        def __init__(self, base: int = 16):
            super().__init__()

            def block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
                )

            self.enc1 = block(1, base)
            self.enc2 = block(base, base * 2)
            self.enc3 = block(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = block(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)


_unet_model_cache: Dict[str, Any] = {}


def load_unet_model(weights_path: str = UNET_WEIGHTS_PATH):
    """
    Load SmallUNet + trained weights for CPU inference. Returns None (never
    raises) when torch isn't installed, the weights file is absent, or
    loading fails for any reason — callers must treat None as "use Otsu".
    """
    if torch is None or not os.path.exists(weights_path):
        return None
    if weights_path in _unet_model_cache:
        return _unet_model_cache[weights_path]
    try:
        model = SmallUNet()
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _unet_model_cache[weights_path] = model
        return model
    except Exception:
        return None


def _unet_segment(img: np.ndarray, model) -> np.ndarray:
    """Run SmallUNet inference on a grayscale image; return a uint8 0/255 mask."""
    with torch.no_grad():
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        probs = torch.sigmoid(model(x))[0, 0].numpy()
    return (probs > 0.5).astype(np.uint8) * 255


# ----------------------------------------------------------------------
# 2. Ice detection
def detect_ice(image_path: str, use_unet: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load image, segment ice, return (original_image, binary_mask,
    ice_pixel_count) — a 3-tuple, so the caller can display both the source
    SAR image and the detected mask.

    If use_unet and a trained SmallUNet is available (see load_unet_model),
    tries it first. Falls back to Gaussian blur + Otsu threshold +
    morphological opening when no model is loaded, inference raises, or the
    predicted mask covers <1% or >60% of the frame (a sanity-check guard
    against a broken/undertrained model, not a latency guard).
    """
    image_path = resolve_sar_path(image_path)  # real crop if present, else the synthetic sample
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    mask = None
    if use_unet:
        model = load_unet_model()
        if model is not None:
            try:
                candidate = _unet_segment(img, model)
                coverage = np.count_nonzero(candidate) / candidate.size
                if 0.01 <= coverage <= 0.60:
                    mask = candidate
            except Exception:
                mask = None

    if mask is None:
        blurred = cv2.GaussianBlur(img, (7, 7), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    ice_cells = int(np.count_nonzero(mask))
    return img, mask, ice_cells


# ----------------------------------------------------------------------
# ML core: KMeans ice-class profiling (optional, internal-only — no new
# output fields). Falls back to a flat risk value when unavailable.
def _kmeans_risk_weights(icebergs_df: pd.DataFrame, n_clusters: int = 3) -> np.ndarray:
    """
    Cluster icebergs by size (mass_kt, freeboard_m) into risk tiers via
    KMeans so bigger/taller icebergs stamp a higher risk value than smaller
    ones. Returns one risk value per row in [7, 10]; falls back to a flat
    10.0 per row when scikit-learn is unavailable, there are too few rows to
    cluster, the required columns are missing, or clustering fails for any
    reason — this must never crash risk-grid construction.
    """
    n = len(icebergs_df)
    flat = np.full(n, 10.0)
    if KMeans is None or n < n_clusters or not {'mass_kt', 'freeboard_m'}.issubset(icebergs_df.columns):
        return flat
    try:
        features = icebergs_df[['mass_kt', 'freeboard_m']].to_numpy(dtype=np.float64)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=DEMO_SEED).fit(features)
        cluster_means = [features[km.labels_ == k].mean() if np.any(km.labels_ == k) else 0.0
                          for k in range(n_clusters)]
        order = np.argsort(cluster_means)  # smallest cluster first
        tier_risk = np.linspace(7.0, 10.0, n_clusters)
        risk_by_cluster = {cluster: tier_risk[rank] for rank, cluster in enumerate(order)}
        return np.array([risk_by_cluster[label] for label in km.labels_])
    except Exception:
        return flat


# ----------------------------------------------------------------------
# 3. Build risk grid (0..10)
def build_risk_grid(mask: np.ndarray, size: int = GRID,
                     icebergs_df: Optional[pd.DataFrame] = None,
                     use_kmeans: bool = True) -> np.ndarray:
    """
    Resize mask to (size,size), convert to float, set >0 to 10, optionally
    stamp known iceberg positions in as hard-risk cells, blur, clip to 0..10.

    icebergs_df is optional and defensive: a missing df, an empty df, a df
    without lat/lon columns, or a row with a bad value are all tolerated —
    none of them should crash a first-run demo. When use_kmeans is True and
    scikit-learn is available, iceberg risk is profiled by size (see
    _kmeans_risk_weights) instead of a flat 10.0 per iceberg.
    """
    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    risk = np.where(resized > 0, 10.0, 0.0)

    if icebergs_df is not None and len(icebergs_df) > 0 \
            and {'lat', 'lon'}.issubset(icebergs_df.columns):
        # Vectorized form of latlon_to_grid() (kept in sync with it manually) —
        # avoids a per-row iterrows() loop over the iceberg table.
        lats = pd.to_numeric(icebergs_df['lat'], errors='coerce').to_numpy()
        lons = pd.to_numeric(icebergs_df['lon'], errors='coerce').to_numpy()
        valid = ~(np.isnan(lats) | np.isnan(lons))
        if valid.any():
            lon_span = (LON_MAX - LON_MIN) or 1.0
            rs = np.clip(np.round(size * (1.0 - (lats[valid] - LAT_MIN))), 0, size - 1).astype(int)
            cs = np.clip(np.round(size * (lons[valid] - LON_MIN) / lon_span), 0, size - 1).astype(int)
            weights = _kmeans_risk_weights(icebergs_df)[valid] if use_kmeans else np.full(valid.sum(), 10.0)
            risk[rs, cs] = weights

    risk = cv2.GaussianBlur(risk, (5, 5), 0)
    risk = np.clip(risk, 0.0, 10.0)
    return risk


# ----------------------------------------------------------------------
# ML core: Ridge drift model (optional). The physics formula below is the
# permanent fallback — never removed, always reachable.
DRIFT_MODEL_PATH = "drift_model.joblib"
_drift_model_cache: Dict[str, Any] = {}


def load_drift_model(model_path: str = DRIFT_MODEL_PATH):
    """
    Load a trained scikit-learn Ridge drift model. Returns None (never
    raises) when scikit-learn/joblib isn't installed, the model file is
    absent, or loading fails — callers must fall back to the physics
    formula in predict_drift().
    """
    if joblib is None or not os.path.exists(model_path):
        return None
    if model_path in _drift_model_cache:
        return _drift_model_cache[model_path]
    try:
        model = joblib.load(model_path)
        _drift_model_cache[model_path] = model
        return model
    except Exception:
        return None


def _ridge_predict_drift(lat: float, lon: float, uc: float, vc: float,
                          uw: float, vw: float, hours: float, model) -> Tuple[float, float]:
    """Ridge model predicts (dlat, dlon) directly from the drift features."""
    features = np.array([[uc, vc, uw, vw, hours]], dtype=np.float64)
    dlat, dlon = model.predict(features)[0]
    return lat + dlat, lon + dlon


# ----------------------------------------------------------------------
# 4. Predict drift due to current and wind (single point)
def predict_drift(lat: float, lon: float, uc: float, vc: float,
                   uw: float, vw: float, hours: float = 24.0,
                   drift_model=None) -> Tuple[float, float]:
    """
    Compute new position after drifting with ocean current (uc,vc) and wind
    (uw,vw). Current and wind in m/s.

    If drift_model is given (a trained Ridge model, see load_drift_model),
    tries it first; any failure falls through to the physics formula below
    (displacement = (current + 0.03*wind) * hours * 3600, converted to
    lat/lon using 1° ~ 111 km) — Ridge is never allowed to crash the demo.
    """
    if drift_model is not None:
        try:
            return _ridge_predict_drift(lat, lon, uc, vc, uw, vw, hours, drift_model)
        except Exception:
            pass  # fall through to the physics formula

    u_total = uc + 0.03 * uw
    v_total = vc + 0.03 * vw

    dx = u_total * hours * 3600.0
    dy = v_total * hours * 3600.0

    dlat = dy / 111000.0
    cos_lat = math.cos(math.radians(lat))
    # Defensive only: Antarctic latitudes here (~-67° to -68°) give cos≈0.39,
    # nowhere near zero — this guard just protects against a stray/bad lat.
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6 if cos_lat >= 0 else -1e-6
    dlon = dx / (111000.0 * cos_lat)

    return lat + dlat, lon + dlon


# ----------------------------------------------------------------------
# 4b. Batch drift prediction for the Streamlit UI
def predict_iceberg_drift(icebergs_df: Optional[pd.DataFrame],
                           wind_df: Optional[pd.DataFrame],
                           hours: float = 24.0, use_ridge: bool = True) -> pd.DataFrame:
    """
    Vectorized-by-row wrapper around predict_drift() for a table of icebergs.

    wind_current.csv only carries a single (u, v) vector per grid point, so
    it is treated here as the net drift-driving field rather than splitting
    it into a separate current + wind (there's no second field to split).

    If use_ridge and a trained drift_model.joblib is available, every row
    uses it (with a per-row fallback to the physics formula on failure — see
    predict_drift). Never raises: a missing/empty icebergs_df returns an
    empty frame with pred_lat/pred_lon columns; a missing/empty wind_df
    falls back to zero drift velocity (icebergs stay put) instead of
    crashing.
    """
    base_cols = ["id", "lat", "lon", "mass_kt", "freeboard_m"]
    if icebergs_df is None:
        icebergs_df = pd.DataFrame(columns=base_cols)

    if len(icebergs_df) == 0 or not {'lat', 'lon'}.issubset(icebergs_df.columns):
        out = icebergs_df.copy()
        out['pred_lat'] = pd.Series(dtype=float)
        out['pred_lon'] = pd.Series(dtype=float)
        return out

    have_wind = (wind_df is not None and len(wind_df) > 0
                 and {'lat', 'lon', 'u_current', 'v_current', 'u_wind', 'v_wind'}.issubset(wind_df.columns))

    drift_model = load_drift_model() if use_ridge else None

    pred_lats: List[Optional[float]] = []
    pred_lons: List[Optional[float]] = []

    # Wind-grid columns hoisted to numpy arrays once, outside the loop, so the
    # per-iceberg nearest-point lookup below is plain array math instead of
    # repeated pandas Series arithmetic + idxmin() (both real per-call overhead).
    if have_wind:
        wind_lat = pd.to_numeric(wind_df['lat'], errors='coerce').to_numpy()
        wind_lon = pd.to_numeric(wind_df['lon'], errors='coerce').to_numpy()
        wind_uc = pd.to_numeric(wind_df['u_current'], errors='coerce').to_numpy()
        wind_vc = pd.to_numeric(wind_df['v_current'], errors='coerce').to_numpy()
        wind_uw = pd.to_numeric(wind_df['u_wind'], errors='coerce').to_numpy()
        wind_vw = pd.to_numeric(wind_df['v_wind'], errors='coerce').to_numpy()

    for _, row in icebergs_df.iterrows():
        try:
            lat, lon = float(row['lat']), float(row['lon'])
        except (KeyError, ValueError, TypeError):
            pred_lats.append(None)
            pred_lons.append(None)
            continue

        uc, vc, uw, vw = 0.0, 0.0, 0.0, 0.0
        if have_wind:
            d2 = (wind_lat - lat) ** 2 + (wind_lon - lon) ** 2
            if np.any(~np.isnan(d2)):
                i = np.nanargmin(d2)
                uc, vc, uw, vw = float(wind_uc[i]), float(wind_vc[i]), float(wind_uw[i]), float(wind_vw[i])
                if any(math.isnan(v) for v in (uc, vc, uw, vw)):
                    uc, vc, uw, vw = 0.0, 0.0, 0.0, 0.0

        new_lat, new_lon = predict_drift(lat, lon, uc, vc, uw, vw, hours, drift_model=drift_model)
        pred_lats.append(new_lat)
        pred_lons.append(new_lon)

    out = icebergs_df.copy()
    out['pred_lat'] = pred_lats
    out['pred_lon'] = pred_lons
    return out


# ----------------------------------------------------------------------
# 5. A* path planning on risk grid
def astar(risk_grid: Optional[np.ndarray], start: Tuple[int, int], goal: Tuple[int, int],
          risk_weight: float = 50.0) -> Optional[List[Tuple[int, int]]]:
    """
    8-directional A*. Cost per step = Euclidean distance (1 or sqrt(2)) +
    risk_weight * (risk/10). Returns list of (r,c) cells from start to goal,
    or None if no path (missing grid, out-of-bounds start/goal, or a
    genuinely unreachable goal) — callers should fall back to direct_path.
    """
    if risk_grid is None:
        return None

    rows, cols = risk_grid.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols and
            0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return None

    # Per-cell penalty, precomputed once. Indexing the numpy grid inside the
    # expansion loop was the hottest line in the whole engine; .tolist() hands
    # back plain Python floats holding the exact same values.
    pen = (risk_grid * (risk_weight / 10.0)).tolist()

    # (dr, dc, step_cost) — hypot(dr, dc) is constant per direction, so hoist it
    # instead of recomputing it on every expansion.
    dirs = [(-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
            (0, -1, 1.0),                   (0, 1, 1.0),
            (1, -1, _SQRT2),  (1, 0, 1.0),  (1, 1, _SQRT2)]

    gr, gc = goal
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()          # finalized cells

    while open_set:
        _, current = heapq.heappop(open_set)
        # A cell can sit in the heap several times with different f-scores. The
        # first pop is its best one; later pops are stale, and re-expanding them
        # was pure wasted work.
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        cr, cc = current
        base_g = g_score[current]
        for dr, dc, step in dirs:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            neighbor = (nr, nc)
            if neighbor in closed:      # already finalized, cannot improve
                continue

            tentative_g = base_g + step + pen[nr][nc]
            if tentative_g < g_score.get(neighbor, _INF):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set,
                               (tentative_g + math.hypot(nr - gr, nc - gc), neighbor))

    return None  # exhausted the open set without reaching goal — genuinely unreachable


# ----------------------------------------------------------------------
# 6. Direct (straight-line) path – sampled cells
def direct_path(risk_grid: Optional[np.ndarray], start: Tuple[int, int],
                 goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Return a list of grid cells along the straight line from start to goal.
    Uses linear interpolation and rounding. Always returns a non-empty list
    (never None) — this is the guaranteed fallback for astar().
    """
    r0, c0 = start
    r1, c1 = goal
    steps = max(abs(r1 - r0), abs(c1 - c0)) + 1
    rs = np.linspace(r0, r1, steps)
    cs = np.linspace(c0, c1, steps)

    if risk_grid is not None:
        max_r, max_c = risk_grid.shape[0] - 1, risk_grid.shape[1] - 1
    else:
        max_r, max_c = GRID - 1, GRID - 1  # no grid to clamp against — fall back to default size

    path = []
    for i in range(steps):
        r = max(0, min(max_r, int(round(rs[i]))))
        c = max(0, min(max_c, int(round(cs[i]))))
        if not path or (r, c) != path[-1]:
            path.append((r, c))
    return path


# ----------------------------------------------------------------------
# 7. Route metrics
def route_metrics(risk_grid: Optional[np.ndarray], path: List[Tuple[int, int]],
                   direct: List[Tuple[int, int]]) -> Dict[str, Any]:
    """
    Compute distance/risk/crossings for both `path` and `direct`, plus
    risk_reduction_pct and fuel_penalty_pct. Divisions by the direct-path
    baseline are guarded (both already were), and a missing risk_grid is
    now tolerated (treated as zero risk everywhere) instead of crashing.
    """
    def compute(path_list):
        if not path_list:
            return {'distance_km': 0.0, 'risk_score': 0.0, 'crossings': 0}

        dist_cells = 0.0
        for i in range(1, len(path_list)):
            dr = path_list[i][0] - path_list[i - 1][0]
            dc = path_list[i][1] - path_list[i - 1][1]
            dist_cells += math.hypot(dr, dc)
        distance_km = dist_cells * KM_PER_CELL

        if risk_grid is not None:
            risks = [float(risk_grid[r, c]) for (r, c) in path_list]
        else:
            risks = [0.0 for _ in path_list]
        risk_score = float(np.mean(risks)) if risks else 0.0
        crossings = sum(1 for r in risks if r > 5.0)
        return {'distance_km': distance_km, 'risk_score': risk_score, 'crossings': crossings}

    path_metrics = compute(path)
    direct_metrics = compute(direct)

    if direct_metrics['risk_score'] > 0:
        risk_reduction_pct = (direct_metrics['risk_score'] - path_metrics['risk_score']) \
            / direct_metrics['risk_score'] * 100.0
    else:
        risk_reduction_pct = 0.0

    if direct_metrics['distance_km'] > 0:
        fuel_penalty_pct = (path_metrics['distance_km'] - direct_metrics['distance_km']) \
            / direct_metrics['distance_km'] * 100.0
    else:
        fuel_penalty_pct = 0.0

    return {
        'path_distance_km': path_metrics['distance_km'],
        'direct_distance_km': direct_metrics['distance_km'],
        'path_risk_score': path_metrics['risk_score'],
        'direct_risk_score': direct_metrics['risk_score'],
        'path_crossings': path_metrics['crossings'],
        'direct_crossings': direct_metrics['crossings'],
        'risk_reduction_pct': risk_reduction_pct,
        'fuel_penalty_pct': fuel_penalty_pct
    }


# ----------------------------------------------------------------------
# 8. SQLite route storage
def save_route(db: str = "routes.db", **fields) -> bool:
    """
    Save a route to the database. Expected fields:
      start, goal (tuple of (lat,lon)), distance_km, risk_red, path_json
    Returns True on success, False on failure (e.g. a momentarily locked db
    from an overlapping Streamlit rerun or a second browser tab) instead of
    raising — a save hiccup should never crash the demo.
    """
    dirpath = os.path.dirname(db)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    conn = None
    try:
        conn = sqlite3.connect(db, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")  # readers/writers don't block each other across reruns
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                start TEXT,
                goal TEXT,
                distance_km REAL,
                risk_red REAL,
                path_json TEXT
            )
        ''')
        start = fields.get('start', (0, 0))
        goal = fields.get('goal', (0, 0))
        distance_km = fields.get('distance_km', 0.0)
        risk_red = fields.get('risk_red', 0.0)
        path_json = fields.get('path_json', [])
        ts = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        c.execute('''
            INSERT INTO routes (ts, start, goal, distance_km, risk_red, path_json)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ts, json.dumps(start), json.dumps(goal), distance_km, risk_red, json.dumps(path_json)))
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        if conn is not None:
            conn.close()


def load_routes(db: str = "routes.db") -> List[Dict[str, Any]]:
    """
    Load all routes from the database as a list of dicts. Never raises: a
    missing db file, a missing table (nothing saved yet), a locked db, or a
    corrupted row all degrade to an empty/partial list instead of an
    exception reaching the caller.
    """
    if not os.path.exists(db):
        return []

    conn = None
    try:
        conn = sqlite3.connect(db, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        c = conn.cursor()
        c.execute('SELECT id, ts, start, goal, distance_km, risk_red, path_json '
                   'FROM routes ORDER BY id DESC')
        rows = c.fetchall()
    except sqlite3.Error:
        return []
    finally:
        if conn is not None:
            conn.close()

    routes = []
    for row in rows:
        try:
            routes.append({
                'id': row[0],
                'ts': row[1],
                'start': json.loads(row[2]),
                'goal': json.loads(row[3]),
                'distance_km': row[4],
                'risk_red': row[5],
                'path_json': json.loads(row[6])
            })
        except (json.JSONDecodeError, TypeError, IndexError):
            continue  # skip a corrupted row rather than dropping the whole history
    return routes


# ----------------------------------------------------------------------
# 9. Strict JSON output
def strict_json(start_ll: Tuple[float, float], goal_ll: Tuple[float, float],
                 path_latlon: List[Tuple[float, float]], metrics: Optional[Dict[str, Any]],
                 drift_list: Optional[List[List[float]]] = None) -> Dict[str, Any]:
    """
    Return a dictionary with the exact keys required by the NCPOR vessel
    API. Callers must json.dumps() this before displaying/sending it —
    this function returns a dict, not a JSON string.
    """
    metrics = metrics or {}
    return {
        "status": "OK",
        "mode": "OFFLINE",
        "route_waypoints": path_latlon,      # list of [lat, lon]
        "total_distance_km": metrics.get('path_distance_km', 0.0),
        "risk_exposure": metrics.get('path_risk_score', 0.0),
        "drift_predictions": drift_list or [],  # list of [lat, lon]
        "generated_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }


# ----------------------------------------------------------------------
# Self-test when run as main
if __name__ == "__main__":
    print("=== Running offline self-test ===")

    sar_path = "data/sar_sample.png"
    print("Generating synthetic SAR...")
    make_synthetic_sar(sar_path, size=400)
    print(f"SAR saved to {sar_path}")

    print("Detecting ice...")
    resolved = resolve_sar_path(sar_path)
    print("Source: real Sentinel-1 crop" if resolved != sar_path else "Source: synthetic sample")
    orig_img, mask, ice_cells = detect_ice(sar_path)
    print(f"Ice pixels: {ice_cells}")

    print("Building risk grid...")
    risk_grid = build_risk_grid(mask, GRID)
    print(f"Risk grid shape: {risk_grid.shape}, min={risk_grid.min():.2f}, max={risk_grid.max():.2f}")

    start = DEMO_START
    goal = DEMO_GOAL
    print(f"Start: {start}, Goal: {goal}")

    print("Running A*...")
    astar_path = astar(risk_grid, start, goal, risk_weight=50.0)
    if astar_path is None:
        print("A* found no path! Falling back to direct path.")
        astar_path = direct_path(risk_grid, start, goal)
    else:
        print(f"A* path length: {len(astar_path)}")

    print("Computing direct path...")
    direct = direct_path(risk_grid, start, goal)
    print(f"Direct path length: {len(direct)}")

    print("Computing metrics...")
    metrics = route_metrics(risk_grid, astar_path, direct)
    print(json.dumps(metrics, indent=2))

    # Scenario check: the demo only tells its story if the direct route actually
    # runs through ice (risk>5 crossings) and A* buys a 50-90% risk reduction.
    # Printed, not raised: a real sar_real.png crop legitimately shifts these.
    crossings = metrics['direct_crossings']
    reduction = metrics['risk_reduction_pct']
    ok = crossings > 0 and 50.0 <= reduction <= 90.0
    print(f"Scenario check: direct_crossings={crossings} "
          f"risk_reduction={reduction:.1f}% -> {'PASS' if ok else 'FAIL'} "
          f"(target 50-90, crossings>0)")

    def path_to_latlon(path):
        return [[grid_to_latlon(r, c)[0], grid_to_latlon(r, c)[1]] for (r, c) in path]

    path_ll = path_to_latlon(astar_path)
    start_ll = grid_to_latlon(start[0], start[1])
    goal_ll = grid_to_latlon(goal[0], goal[1])

    print("Predicting batch iceberg drift...")
    sample_icebergs = pd.DataFrame([
        {"id": 1, "lat": -67.3, "lon": 60.1, "mass_kt": 12.0, "freeboard_m": 5.0},
        {"id": 2, "lat": -67.6, "lon": 60.5, "mass_kt": 8.0, "freeboard_m": 3.0},
    ])
    sample_wind = pd.DataFrame([
        {"lat": -67.3, "lon": 60.1, "u_current": 0.1, "v_current": 0.08, "u_wind": 3.0, "v_wind": 2.0},
        {"lat": -67.6, "lon": 60.5, "u_current": 0.12, "v_current": 0.05, "u_wind": 4.0, "v_wind": 2.5},
    ])
    drift_df = predict_iceberg_drift(sample_icebergs, sample_wind, hours=24.0)
    drift_list = drift_df[['pred_lat', 'pred_lon']].values.tolist()
    print(f"Drift predictions: {drift_list}")

    print("Generating strict JSON...")
    output = strict_json(start_ll, goal_ll, path_ll, metrics, drift_list)
    print(json.dumps(output, indent=2))

    print("Saving route to database...")
    ok = save_route(
        db="routes.db",
        start=start_ll,
        goal=goal_ll,
        distance_km=metrics['path_distance_km'],
        risk_red=metrics['risk_reduction_pct'],
        path_json=path_ll
    )
    print(f"Save succeeded: {ok}")

    print("Loading routes from database...")
    routes = load_routes("routes.db")
    print(f"Found {len(routes)} route(s) in DB.")

    print("=== Self-test complete ===")