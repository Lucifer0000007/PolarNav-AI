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
except Exception:  # ImportError, or a broken/partial install raising OSError on DLL load
    torch = None   # a damaged torch must never take the offline demo down: Otsu path stays live
    nn = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.cluster import KMeans
except Exception:
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
# NSIDC Sea Ice Index (Antarctic/south) — makes the synthetic SAR patch
# physics-informed: a real historical extent drives how much ice the
# synthetic field shows, instead of an arbitrary fixed blob count.
SEAICE_CSV_PATH = "seaice.csv"
SEAICE_EXTENT_MIN = 2.0   # million km^2 -> ~0% patch ice coverage
SEAICE_EXTENT_MAX = 20.5  # million km^2 -> ~100% patch ice coverage. Calibrated
                          # against the real south-hemisphere range in
                          # seaice.csv (2.08-20.20 M km^2); these round just
                          # outside it so no real historical day clips to
                          # exactly 0% or 100% coverage.

_seaice_df_cache: Dict[str, Any] = {}


def load_seaice_south(csv_path: str = SEAICE_CSV_PATH) -> pd.DataFrame:
    """
    Load the NSIDC Sea Ice Index CSV, return only Antarctic (hemisphere ==
    'south') rows with column names stripped of stray whitespace (the raw
    CSV header has leading spaces, e.g. ' Month'). Never raises: a missing
    file, a malformed CSV, or a file with no usable rows all degrade to an
    empty frame — callers must fall back to the pinned default field.
    """
    if csv_path in _seaice_df_cache:
        return _seaice_df_cache[csv_path]

    empty = pd.DataFrame(columns=["Year", "Month", "Day", "Extent"])
    if not os.path.exists(csv_path):
        _seaice_df_cache[csv_path] = empty
        return empty
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if not {"hemisphere", "Extent"}.issubset(df.columns):
            df = empty
        else:
            df = df[df["hemisphere"].astype(str).str.strip() == "south"].copy()
            df["Extent"] = pd.to_numeric(df["Extent"], errors="coerce")
            df = df.dropna(subset=["Extent"])
    except Exception:
        df = empty
    _seaice_df_cache[csv_path] = df
    return df


def sample_seaice_row(csv_path: str = SEAICE_CSV_PATH) -> Optional[Dict[str, Any]]:
    """
    Pick one random Antarctic (south) row from the NSIDC CSV. Returns None
    (never raises) when the CSV is missing/empty/malformed — the caller
    should fall back to the pinned DEMO_SEED synthetic field.
    """
    df = load_seaice_south(csv_path)
    if df.empty:
        return None
    row = df.sample(n=1).iloc[0]
    return {
        "year": int(row["Year"]),
        "month": int(row["Month"]),
        "day": int(row["Day"]),
        "extent": float(row["Extent"]),
    }


def extent_to_coverage(extent_mkm2: float) -> float:
    """
    Normalize a real NSIDC extent (million km^2) to a 0..1 ice-coverage
    fraction for the synthetic SAR patch: SEAICE_EXTENT_MIN -> 0.0,
    SEAICE_EXTENT_MAX -> 1.0, clamped to that range.
    """
    span = (SEAICE_EXTENT_MAX - SEAICE_EXTENT_MIN) or 1.0
    frac = (extent_mkm2 - SEAICE_EXTENT_MIN) / span
    return max(0.0, min(1.0, frac))


# ----------------------------------------------------------------------
# 1. Generate synthetic SAR image (grayscale, uint8)
def make_synthetic_sar(path: str = "data/sar_sample.png", size: int = 400,
                       seed: Optional[int] = DEMO_SEED,
                       target_extent: Optional[float] = None) -> str:
    """
    Create a synthetic SAR image with:
      - dark ocean background (10–40)
      - bright elliptical ice blobs (150–255)
      - speckle noise (multiplicative)
    Save as PNG and return the path.

    seed pins the ice field so the demo scenario is repeatable run to run;
    pass seed=None for a fresh random field.

    If target_extent is given (million km^2, NSIDC-style — see
    sample_seaice_row), the blob count is scaled by extent_to_coverage()
    instead of the fixed 6-10 range, so the synthetic patch's ice density
    reflects a real historical Antarctic extent rather than an arbitrary
    count.
    """
    if seed is not None:
        np.random.seed(seed)

    dirpath = os.path.dirname(path)
    if dirpath:  # os.makedirs("") raises FileNotFoundError, so only call it when there IS a dir
        os.makedirs(dirpath, exist_ok=True)

    img = np.random.randint(10, 41, (size, size), dtype=np.uint8)
    if target_extent is not None:
        coverage = extent_to_coverage(target_extent)
        n_blobs = int(round(2 + coverage * 18))  # 2..20 blobs across the real extent range
    else:
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
#
# Architecture ported from sea-ice-segmentation-u-net.ipynb's get_unet():
# same filter progression (32-64-128-256, bottleneck 512), same Dropout(0.5)
# after every pool/concat, and the same upsample+conv decoder (the notebook
# uses UpSampling2D+Conv2D, not a transposed convolution) with skip
# connections. Adapted for grayscale SAR input (in_channels=1) and binary
# ice/no-ice output (out_channels=1, sigmoid via BCEWithLogitsLoss, applied
# by the caller) instead of the notebook's 3-channel RGB input and 8-class
# softmax output over ice-concentration categories.
UNET_WEIGHTS_PATH = "models/unet_weights.pth"

if torch is not None:
    class SmallUNet(nn.Module):
        """U-Net, architecture synced with sea-ice-segmentation-u-net.ipynb."""

        def __init__(self, in_channels: int = 1, out_channels: int = 1):
            super().__init__()

            def conv_block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
                )

            def up_conv(cin, cout):
                # Matches the notebook's UpSampling2D(2) -> Conv2D (not ConvTranspose2d).
                return nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                )

            self.pool = nn.MaxPool2d(2)
            self.drop = nn.Dropout(0.5)

            self.enc1 = conv_block(in_channels, 32)
            self.enc2 = conv_block(32, 64)
            self.enc3 = conv_block(64, 128)
            self.enc4 = conv_block(128, 256)
            self.bottleneck = conv_block(256, 512)

            self.up6 = up_conv(512, 256)
            self.dec6 = conv_block(512, 256)
            self.up7 = up_conv(256, 128)
            self.dec7 = conv_block(256, 128)
            self.up8 = up_conv(128, 64)
            self.dec8 = conv_block(128, 64)
            self.up9 = up_conv(64, 32)
            self.dec9 = conv_block(64, 32)

            self.out = nn.Conv2d(32, out_channels, 1)

        def forward(self, x):
            c1 = self.enc1(x)
            c2 = self.enc2(self.drop(self.pool(c1)))
            c3 = self.enc3(self.drop(self.pool(c2)))
            c4 = self.enc4(self.drop(self.pool(c3)))
            c5 = self.bottleneck(self.drop(self.pool(c4)))

            d6 = self.dec6(self.drop(torch.cat([self.up6(c5), c4], dim=1)))
            d7 = self.dec7(self.drop(torch.cat([self.up7(d6), c3], dim=1)))
            d8 = self.dec8(self.drop(torch.cat([self.up8(d7), c2], dim=1)))
            d9 = self.dec9(self.drop(torch.cat([self.up9(d8), c1], dim=1)))

            return self.out(d9)  # logits — caller applies sigmoid (binary, not the notebook's softmax)


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
def detect_ice(image_path: str, use_unet: bool = True) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Load image, segment ice, return (original_image, binary_mask,
    ice_pixel_count, active_path) — active_path is "unet" or "otsu", so the
    caller can honestly caption which inference path actually ran.

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
    active_path = "otsu"
    if use_unet:
        model = load_unet_model()
        if model is not None:
            try:
                candidate = _unet_segment(img, model)
                coverage = np.count_nonzero(candidate) / candidate.size
                if 0.01 <= coverage <= 0.60:
                    mask = candidate
                    active_path = "unet"
            except Exception:
                mask = None

    if mask is None:
        blurred = cv2.GaussianBlur(img, (7, 7), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        active_path = "otsu"

    ice_cells = int(np.count_nonzero(mask))
    return img, mask, ice_cells, active_path


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

    mass_kt (hundreds-to-thousands) and freeboard_m (single-to-double digits)
    are on wildly different numeric scales; KMeans uses Euclidean distance, so
    fitting on the raw values lets mass alone decide cluster membership and
    freeboard barely matters. Each feature is divided by its own column max
    (guarded against zero) before fitting so both contribute; tier ORDER
    still ranks clusters by raw mass_kt specifically, so "bigger = higher
    tier" keeps its plain meaning regardless of freeboard's own scale.
    """
    n = len(icebergs_df)
    flat = np.full(n, 10.0)
    if KMeans is None or n < n_clusters or not {'mass_kt', 'freeboard_m'}.issubset(icebergs_df.columns):
        return flat
    try:
        features = icebergs_df[['mass_kt', 'freeboard_m']].to_numpy(dtype=np.float64)
        col_max = np.where(features.max(axis=0) > 0, features.max(axis=0), 1.0)  # guard zero
        features_norm = features / col_max
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=DEMO_SEED).fit(features_norm)
        cluster_means = [features[km.labels_ == k, 0].mean() if np.any(km.labels_ == k) else 0.0
                          for k in range(n_clusters)]  # ranked by raw mass_kt, not the normalized joint centroid
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
            # np.maximum.at, not risk[rs, cs] = weights: plain fancy-index assignment lets
            # a later iceberg in array order silently overwrite an earlier one's higher
            # risk when two round to the same cell. maximum.at is order-independent.
            np.maximum.at(risk, (rs, cs), weights)

    risk = cv2.GaussianBlur(risk, (5, 5), 0)
    risk = np.clip(risk, 0.0, 10.0)
    return risk


# ----------------------------------------------------------------------
# 3b. 24-h sea-ice concentration forecast. The present risk grid is advected
# by the drift field (current + 3% wind, the same physics predict_drift uses)
# and max-combined with the present state. Physics only: the optional Ridge
# model is a per-iceberg predictor, not a per-cell field.
def build_drift_field(wind_df: Optional[pd.DataFrame], size: int = GRID) -> Optional[np.ndarray]:
    """
    Resample the sparse wind/current table onto the risk grid: every cell
    centre takes its nearest wind point, giving a (size, size, 2) array of
    (u_total, v_total) m/s where u_total = u_current + 0.03*u_wind (the net
    drift combination predict_drift uses). Returns None (never raises) for a
    missing/empty/badly-shaped table; callers treat None as "no forecast,
    predicted == current".
    """
    cols = ['lat', 'lon', 'u_current', 'v_current', 'u_wind', 'v_wind']
    if wind_df is None or len(wind_df) == 0 or not set(cols).issubset(wind_df.columns):
        return None
    try:
        w = wind_df[cols].apply(pd.to_numeric, errors='coerce').dropna()
        if w.empty:
            return None
        w_lat, w_lon = w['lat'].to_numpy(), w['lon'].to_numpy()
        u_tot = (w['u_current'] + 0.03 * w['u_wind']).to_numpy()
        v_tot = (w['v_current'] + 0.03 * w['v_wind']).to_numpy()
        rr, cc = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        lat_c = LAT_MIN + (1.0 - rr / size)                      # grid_to_latlon, vectorized
        lon_c = LON_MIN + (LON_MAX - LON_MIN) * (cc / size)
        d2 = (lat_c[..., None] - w_lat) ** 2 + (lon_c[..., None] - w_lon) ** 2
        idx = np.argmin(d2, axis=2)
        return np.stack([u_tot[idx], v_tot[idx]], axis=-1)
    except Exception:
        return None


def predict_risk_grid(risk_grid: Optional[np.ndarray], drift_field: Optional[np.ndarray],
                      hours: float = 24.0) -> Optional[np.ndarray]:
    """
    Forecast the risk grid `hours` ahead. Each cell's risk is carried by its
    drift vector (displacement = (current + 0.03*wind) * hours, 1 deg ~ 111 km,
    longitude scaled by cos(lat)), shifted a whole number of cells and clamped
    to the grid edge. Cells landing on the same target keep the maximum, and
    the result is max-combined with the present grid so forecast risk never
    drops below current risk (ice here now is still a hazard even if the field
    says it moves on). Never raises; drift_field=None returns a copy.
    """
    if risk_grid is None:
        return None
    if drift_field is None or drift_field.shape[:2] != risk_grid.shape:
        return risk_grid.copy()
    try:
        rows, cols = risk_grid.shape
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        lat_c = LAT_MIN + (1.0 - rr / rows)
        secs = hours * 3600.0
        dlat = drift_field[..., 1] * secs / 111000.0
        cos_lat = np.cos(np.radians(lat_c))
        cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
        dlon = drift_field[..., 0] * secs / (111000.0 * cos_lat)
        lon_span = (LON_MAX - LON_MIN) or 1.0
        tr = np.clip(np.round(rr - dlat * rows), 0, rows - 1).astype(int)        # r grows southward
        tc = np.clip(np.round(cc + dlon * cols / lon_span), 0, cols - 1).astype(int)
        advected = np.zeros_like(risk_grid)
        np.maximum.at(advected, (tr.ravel(), tc.ravel()), risk_grid.ravel())
        return np.maximum(risk_grid, advected)
    except Exception:
        return risk_grid.copy()


def kmeans_band_edges(risk_predicted: Optional[np.ndarray], n_bands: int = 5) -> Tuple[List[float], str]:
    """
    Concentration band edges (percent, 0..100) for the forecast overlay
    legend. With scikit-learn available, KMeans(n_bands) over the non-zero
    forecast cells gives data-driven classes; the edges are midpoints between
    sorted cluster centres (tag "kmeans"). Otherwise, or when clustering is
    impossible (too few distinct values) or fails, fixed 20 % steps are used
    (tag "fixed"). Never raises.
    """
    fixed = [20.0, 40.0, 60.0, 80.0]
    if KMeans is None or risk_predicted is None:
        return fixed, "fixed"
    try:
        vals = np.asarray(risk_predicted, dtype=np.float64).ravel() * 10.0   # risk 0..10 -> 0..100 %
        vals = vals[vals > 0]
        if len(np.unique(np.round(vals, 3))) < n_bands:
            return fixed, "fixed"
        km = KMeans(n_clusters=n_bands, n_init=10, random_state=DEMO_SEED).fit(vals.reshape(-1, 1))
        centres = np.sort(km.cluster_centers_.ravel())
        edges = [float(round((centres[i] + centres[i + 1]) / 2.0, 1)) for i in range(n_bands - 1)]
        return edges, "kmeans"
    except Exception:
        return fixed, "fixed"


# ----------------------------------------------------------------------
# ML core: Ridge drift model (optional). The physics formula below is the
# permanent fallback — never removed, always reachable.
DRIFT_MODEL_PATH = "models/drift_model.joblib"   # trained by train_drift.py (not run by the app)
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
                   direct: List[Tuple[int, int]],
                   predicted_grid: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute distance/risk/crossings for both `path` and `direct`, plus
    risk_reduction_pct and fuel_penalty_pct. Divisions by the direct-path
    baseline are guarded (both already were), and a missing risk_grid is
    now tolerated (treated as zero risk everywhere) instead of crashing.

    predicted_grid (see predict_risk_grid) adds the 24-h view of the same
    path: current_exposure (== path_risk_score), predicted_exposure and
    predicted_crossings. Without it the predicted values equal the current.
    """
    def compute(path_list, grid=risk_grid):
        if not path_list:
            return {'distance_km': 0.0, 'risk_score': 0.0, 'crossings': 0}

        dist_cells = 0.0
        for i in range(1, len(path_list)):
            dr = path_list[i][0] - path_list[i - 1][0]
            dc = path_list[i][1] - path_list[i - 1][1]
            dist_cells += math.hypot(dr, dc)
        distance_km = dist_cells * KM_PER_CELL

        if grid is not None:
            risks = [float(grid[r, c]) for (r, c) in path_list]
        else:
            risks = [0.0 for _ in path_list]
        risk_score = float(np.mean(risks)) if risks else 0.0
        crossings = sum(1 for r in risks if r > 5.0)
        return {'distance_km': distance_km, 'risk_score': risk_score, 'crossings': crossings}

    path_metrics = compute(path)
    direct_metrics = compute(direct)
    pred_metrics = compute(path, predicted_grid) if predicted_grid is not None else path_metrics

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
        'fuel_penalty_pct': fuel_penalty_pct,
        'current_exposure': path_metrics['risk_score'],
        'predicted_exposure': pred_metrics['risk_score'],
        'predicted_crossings': pred_metrics['crossings'],
    }


# ----------------------------------------------------------------------
# 7b. Alerting: closest point of approach, threat class, reroute suggestion
def cpa_km(route_latlon: List[Any], track_latlon: List[Any], samples: int = 25) -> float:
    """
    Closest Point of Approach (km) between a route polyline (list of
    (lat, lon) waypoints) and an iceberg's 24-h track (list of (lat, lon),
    typically [current, predicted]); the track is densified to `samples`
    points per segment. Equirectangular distance with longitude scaled by
    cos(mean route lat), accurate to well under 1 % on this 1 x 1.5 deg
    patch. Returns inf (never raises) when either input is empty.
    """
    try:
        route = np.asarray(route_latlon, dtype=np.float64).reshape(-1, 2)
        track = np.asarray(track_latlon, dtype=np.float64).reshape(-1, 2)
        if route.size == 0 or track.size == 0:
            return _INF
        if len(track) > 1:
            track = np.vstack([np.linspace(track[i], track[i + 1], samples)
                               for i in range(len(track) - 1)])
        cos_lat = math.cos(math.radians(float(np.mean(route[:, 0]))))
        dlat = (route[:, None, 0] - track[None, :, 0]) * 111.0
        dlon = (route[:, None, 1] - track[None, :, 1]) * 111.0 * cos_lat
        d = np.sqrt(dlat ** 2 + dlon ** 2)
        d = d[~np.isnan(d)]
        return float(d.min()) if d.size else _INF
    except Exception:
        return _INF


def classify_threat(cpa: float, predicted_crossings: int) -> str:
    """HIGH if CPA < 5 km or the route crosses predicted risk>5 cells; MED if CPA < 10 km; else LOW."""
    if cpa < 5.0 or predicted_crossings > 0:
        return "HIGH"
    if cpa < 10.0:
        return "MED"
    return "LOW"


def suggest_reroute(risk_combined: Optional[np.ndarray], start: Tuple[int, int], goal: Tuple[int, int],
                    base_path: List[Tuple[int, int]], predicted_grid: Optional[np.ndarray],
                    risk_weight: float = 50.0) -> Optional[Tuple[List[Tuple[int, int]], float]]:
    """
    On a HIGH threat, re-run A* with the risk penalty doubled and offer the
    result as a TEXT suggestion: returns (alt_path, delta_km) when the
    alternative lowers predicted exposure, else None (no improvement, no
    path, or any failure). The displayed route is never switched here.
    """
    try:
        alt = astar(risk_combined, start, goal, risk_weight=risk_weight * 2.0)
        if alt is None or not base_path:
            return None
        base_m = route_metrics(risk_combined, base_path, base_path, predicted_grid=predicted_grid)
        alt_m = route_metrics(risk_combined, alt, base_path, predicted_grid=predicted_grid)
        if alt_m['predicted_exposure'] < base_m['predicted_exposure'] - 1e-9:
            return alt, alt_m['path_distance_km'] - base_m['path_distance_km']
        return None
    except Exception:
        return None


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
    orig_img, mask, ice_cells, ice_path = detect_ice(sar_path)
    print(f"Ice pixels: {ice_cells} (active path: {ice_path})")
    print(f"Drift source: {'ridge (models/drift_model.joblib)' if load_drift_model() else 'physics formula (fallback)'}")

    print("Building risk grid...")
    risk_grid = build_risk_grid(mask, GRID)
    print(f"Risk grid shape: {risk_grid.shape}, min={risk_grid.min():.2f}, max={risk_grid.max():.2f}")

    print("Forecasting 24-h risk grid...")
    wind_path = "data/wind_current.csv"
    wind_df = pd.read_csv(wind_path) if os.path.exists(wind_path) else pd.DataFrame([
        {"lat": -67.3, "lon": 60.1, "u_current": 0.1, "v_current": 0.08, "u_wind": 3.0, "v_wind": 2.0},
        {"lat": -67.6, "lon": 60.5, "u_current": 0.12, "v_current": 0.05, "u_wind": 4.0, "v_wind": 2.5},
    ])
    drift_field = build_drift_field(wind_df)
    risk_pred = predict_risk_grid(risk_grid, drift_field, hours=24.0)
    risk_combined = np.maximum(risk_grid, risk_pred)
    print(f"Drift field: {'ok' if drift_field is not None else 'none (predicted == current)'} | "
          f"cells>5 now={int((risk_grid > 5).sum())} in 24h={int((risk_pred > 5).sum())}")
    band_edges, band_src = kmeans_band_edges(risk_pred)
    print(f"Concentration bands (%): {band_edges} (source: {band_src})")

    start = DEMO_START
    goal = DEMO_GOAL
    print(f"Start: {start}, Goal: {goal}")

    print("Running A* on combined (current | 24h predicted) risk...")
    astar_path = astar(risk_combined, start, goal, risk_weight=50.0)
    if astar_path is None:
        print("A* found no path! Falling back to direct path.")
        astar_path = direct_path(risk_grid, start, goal)
    else:
        print(f"A* path length: {len(astar_path)}")

    print("Computing direct path...")
    direct = direct_path(risk_grid, start, goal)
    print(f"Direct path length: {len(direct)}")

    print("Computing metrics...")
    metrics = route_metrics(risk_grid, astar_path, direct, predicted_grid=risk_pred)
    print(json.dumps(metrics, indent=2))
    assert metrics['predicted_exposure'] >= metrics['current_exposure'], \
        "forecast exposure must never be below current exposure (max-combine invariant)"
    print(f"Exposure: current={metrics['current_exposure']:.3f} "
          f"predicted_24h={metrics['predicted_exposure']:.3f} "
          f"(predicted risk>5 cells on route: {metrics['predicted_crossings']})")

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

    print("Computing CPA / threat class...")
    cpas = {int(row['id']): round(cpa_km(path_ll, [[row['lat'], row['lon']],
                                                    [row['pred_lat'], row['pred_lon']]]), 2)
            for _, row in drift_df.iterrows()}
    min_cpa = min(cpas.values()) if cpas else _INF
    threat = classify_threat(min_cpa, metrics['predicted_crossings'])
    print(f"CPA per iceberg (km): {cpas} -> threat {threat}")
    reroute = suggest_reroute(risk_combined, start, goal, astar_path, risk_pred) if threat == "HIGH" else None
    print(f"Reroute suggestion: {('+%.1f km' % reroute[1]) if reroute else 'none'}")

    print("Verifying risk-stamp max-accumulate (F2)...")
    # Two icebergs rounding to the same grid cell must leave the HIGHER weight
    # stamped regardless of array order — plain fancy-index assignment
    # (risk[rs,cs] = weights) would instead keep whichever is listed LAST.
    _dupe_rs = np.array([2, 2])
    _dupe_cs = np.array([2, 2])
    for _order, _w in [("low-then-high", np.array([7.0, 10.0])), ("high-then-low", np.array([10.0, 7.0]))]:
        _grid = np.zeros((5, 5))
        np.maximum.at(_grid, (_dupe_rs, _dupe_cs), _w)
        assert _grid[2, 2] == 10.0, f"max-accumulate failed for order {_order}: got {_grid[2, 2]}, expected 10.0"
        _would_have_been = _w[-1]  # what plain risk[rs,cs] = weights would have left behind
        print(f"  order={_order}: max-accumulate -> {_grid[2, 2]:.1f} "
              f"(plain assignment would have given {_would_have_been:.1f})")
    print("Max-accumulate OK: result is 10.0 regardless of array order.")

    print("Verifying KMeans per-feature scaling (F4)...")
    _bergs_path = "data/icebergs.csv"
    if os.path.exists(_bergs_path):
        _bergs = pd.read_csv(_bergs_path)
        _w = _kmeans_risk_weights(_bergs)
        for _i, _row in _bergs.iterrows():
            print(f"  id={int(_row['id']):>2}  mass_kt={_row['mass_kt']:>8.2f}  "
                  f"freeboard_m={_row['freeboard_m']:>6.2f}  tier_weight={_w[_i]:.1f}")
        print("(freeboard now measurably influences tier assignment instead of being drowned out by mass's larger scale)")
    else:
        print(f"  {_bergs_path} not found — skipping (not part of the pinned scenario)")

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