"""
engine.py - Offline Antarctic navigation prototype (SIH26059)
Uses only numpy, opencv-python, pandas, and Python standard library.
No network calls, no ML frameworks, no API keys.
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

# Constants
GRID = 40
KM_PER_CELL = 111.0 / 40.0          # ~2.775 km per cell at 1° latitude
LAT_MIN, LAT_MAX = -68.0, -67.0
LON_MIN, LON_MAX = 59.5, 61.0

# Demo scenario: pinned ice field + the start/goal the app defaults to, tuned so
# the direct route runs through ice and A* buys a visible risk reduction.
DEMO_SEED = 13
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
# 2. Ice detection
def detect_ice(image_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load image, apply Gaussian blur, Otsu threshold, morphological opening.
    Returns (original_image, binary_mask, ice_pixel_count) — a 3-tuple, so
    the caller can display both the source SAR image and the detected mask.
    """
    image_path = resolve_sar_path(image_path)  # real crop if present, else the synthetic sample
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    ice_cells = int(np.count_nonzero(mask))
    return img, mask, ice_cells


# ----------------------------------------------------------------------
# 3. Build risk grid (0..10)
def build_risk_grid(mask: np.ndarray, size: int = GRID,
                     icebergs_df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Resize mask to (size,size), convert to float, set >0 to 10, optionally
    stamp known iceberg positions in as hard-risk cells, blur, clip to 0..10.

    icebergs_df is optional and defensive: a missing df, an empty df, a df
    without lat/lon columns, or a row with a bad value are all tolerated —
    none of them should crash a first-run demo.
    """
    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    risk = np.where(resized > 0, 10.0, 0.0)

    if icebergs_df is not None and len(icebergs_df) > 0 \
            and {'lat', 'lon'}.issubset(icebergs_df.columns):
        for _, row in icebergs_df.iterrows():
            try:
                r, c = latlon_to_grid(float(row['lat']), float(row['lon']), size)
                risk[r, c] = 10.0
            except (ValueError, TypeError):
                continue  # skip a malformed row rather than crash the whole grid build

    risk = cv2.GaussianBlur(risk, (5, 5), 0)
    risk = np.clip(risk, 0.0, 10.0)
    return risk


# ----------------------------------------------------------------------
# 4. Predict drift due to current and wind (single point)
def predict_drift(lat: float, lon: float, uc: float, vc: float,
                   uw: float, vw: float, hours: float = 24.0) -> Tuple[float, float]:
    """
    Compute new position after drifting with ocean current (uc,vc) and wind
    (uw,vw). Current and wind in m/s. Displacement = (current + 0.03*wind)
    * hours * 3600. Convert to lat/lon using 1° ~ 111 km.
    """
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
                           hours: float = 24.0) -> pd.DataFrame:
    """
    Vectorized-by-row wrapper around predict_drift() for a table of icebergs.

    wind_current.csv only carries a single (u, v) vector per grid point, so
    it is treated here as the net drift-driving field rather than splitting
    it into a separate current + wind (there's no second field to split).

    Never raises: a missing/empty icebergs_df returns an empty frame with
    pred_lat/pred_lon columns; a missing/empty wind_df falls back to zero
    drift velocity (icebergs stay put) instead of crashing.
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

    pred_lats: List[Optional[float]] = []
    pred_lons: List[Optional[float]] = []

    for _, row in icebergs_df.iterrows():
        try:
            lat, lon = float(row['lat']), float(row['lon'])
        except (KeyError, ValueError, TypeError):
            pred_lats.append(None)
            pred_lons.append(None)
            continue

        uc, vc, uw, vw = 0.0, 0.0, 0.0, 0.0
        if have_wind:
            d2 = (wind_df['lat'] - lat) ** 2 + (wind_df['lon'] - lon) ** 2
            if d2.notna().any():
                nearest = wind_df.loc[d2.idxmin()]
                try:
                    uc = float(nearest['u_current'])
                    vc = float(nearest['v_current'])
                    uw = float(nearest['u_wind'])
                    vw = float(nearest['v_wind'])
                except (ValueError, TypeError):
                    uc, vc, uw, vw = 0.0, 0.0, 0.0, 0.0

        new_lat, new_lon = predict_drift(lat, lon, uc, vc, uw, vw, hours)
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

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    dirs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dr, dc in dirs:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            move_cost = math.hypot(dr, dc)
            risk_penalty = risk_weight * (risk_grid[nr, nc] / 10.0)
            tentative_g = g_score[current] + move_cost + risk_penalty

            neighbor = (nr, nc)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

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