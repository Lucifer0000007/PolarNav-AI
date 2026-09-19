"""
train_drift.py - External training script for the Ridge drift model (SIH26059).

NOT run by the app. Run once, offline, to produce models/drift_model.joblib,
which engine.predict_drift() then uses ahead of the physics formula (the
formula stays as the permanent fallback).

DATA PROVENANCE - read before quoting any number from this script:
  * If data/features.csv exists with columns
        u_current, v_current, u_wind, v_wind, hours, dlat, dlon
    it is used as OBSERVED training data (one row = one iceberg displacement).
  * Otherwise (the shipped default) the training set is SYNTHETIC: drift
    features are sampled around the ranges in data/wind_current.csv and the
    targets come from engine.predict_drift()'s physics formula at the patch's
    mid-latitude plus small Gaussian noise. The Ridge model therefore learns
    to reproduce the physics - it demonstrates the ML pipeline (train ->
    serialize -> load -> infer with fallback), NOT skill on observed drift.
    The report file states which source was used; captions must stay honest.

Usage:
    python train_drift.py                # synthetic (or features.csv if present)
    python train_drift.py --n 8000 --alpha 0.5
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
except ImportError as e:
    raise SystemExit(
        "scikit-learn + joblib are required to run train_drift.py (the app "
        "itself falls back to the physics formula without them). Install: "
        "pip install scikit-learn joblib"
    ) from e

from engine import predict_drift, DRIFT_MODEL_PATH, DEMO_SEED, LAT_MIN, LAT_MAX

FEATURES_CSV = "data/features.csv"
WIND_CSV = "data/wind_current.csv"
REPORT_PATH = "drift_training_report.json"
FEATURE_COLS = ["u_current", "v_current", "u_wind", "v_wind", "hours"]
TARGET_COLS = ["dlat", "dlon"]
MID_LAT = (LAT_MIN + LAT_MAX) / 2.0   # -67.5: the patch mid-latitude the synthetic targets assume
MAX_MODEL_MB = 10.0


def load_observed():
    """OBSERVED rows from data/features.csv, or None if absent/malformed."""
    if not os.path.exists(FEATURES_CSV):
        return None
    df = pd.read_csv(FEATURES_CSV)
    if not set(FEATURE_COLS + TARGET_COLS).issubset(df.columns):
        print(f"{FEATURES_CSV} lacks the required columns - ignoring it.")
        return None
    df = df[FEATURE_COLS + TARGET_COLS].apply(pd.to_numeric, errors="coerce").dropna()
    return df if len(df) >= 50 else None


def make_synthetic(n: int, seed: int, noise_deg: float = 0.002) -> pd.DataFrame:
    """SYNTHETIC rows: features sampled around the wind CSV's ranges, targets
    from the physics formula at MID_LAT with Gaussian noise (sigma noise_deg)."""
    rng = np.random.default_rng(seed)
    if os.path.exists(WIND_CSV):
        w = pd.read_csv(WIND_CSV)
        cur_scale = float(np.nanmax(np.abs(w[["u_current", "v_current"]].to_numpy()))) * 2.0
        wind_scale = float(np.nanmax(np.abs(w[["u_wind", "v_wind"]].to_numpy()))) * 2.0
    else:
        cur_scale, wind_scale = 0.3, 10.0
    uc = rng.uniform(-cur_scale, cur_scale, n)
    vc = rng.uniform(-cur_scale, cur_scale, n)
    uw = rng.uniform(-wind_scale, wind_scale, n)
    vw = rng.uniform(-wind_scale, wind_scale, n)
    hours = rng.uniform(1.0, 48.0, n)
    dlat, dlon = np.empty(n), np.empty(n)
    for i in range(n):
        lat2, lon2 = predict_drift(MID_LAT, 60.0, uc[i], vc[i], uw[i], vw[i], hours[i])
        dlat[i], dlon[i] = lat2 - MID_LAT, lon2 - 60.0
    dlat += rng.normal(0.0, noise_deg, n)
    dlon += rng.normal(0.0, noise_deg, n)
    return pd.DataFrame({"u_current": uc, "v_current": vc, "u_wind": uw, "v_wind": vw,
                         "hours": hours, "dlat": dlat, "dlon": dlon})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="synthetic sample count")
    ap.add_argument("--alpha", type=float, default=1e-3, help="Ridge regularisation")
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--out", default=DRIFT_MODEL_PATH)
    args = ap.parse_args()

    df = load_observed()
    source = "OBSERVED (data/features.csv)"
    if df is None:
        df = make_synthetic(args.n, args.seed)
        source = "SYNTHETIC (physics formula at lat -67.5 + Gaussian noise; see module docstring)"
    print(f"Training source: {source} - {len(df)} rows")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    n_hold = max(1, int(0.2 * len(df)))
    hold, train = idx[:n_hold], idx[n_hold:]
    X, Y = df[FEATURE_COLS].to_numpy(), df[TARGET_COLS].to_numpy()

    # Displacement is velocity x time, so give Ridge the pairwise products
    # (u*hours, ...) via PolynomialFeatures. The saved object is one sklearn
    # Pipeline: engine._ridge_predict_drift() still calls .predict() on the raw
    # 5-feature row, unchanged.
    model = make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                          Ridge(alpha=args.alpha)).fit(X[train], Y[train])
    pred = model.predict(X[hold])
    r2 = float(r2_score(Y[hold], pred))
    rmse_km = float(np.sqrt(np.mean((pred - Y[hold]) ** 2)) * 111.0)

    # Physics baseline on the same held-out rows, for an honest comparison.
    phys = np.array([[predict_drift(MID_LAT, 60.0, *row)[0] - MID_LAT,
                      predict_drift(MID_LAT, 60.0, *row)[1] - 60.0] for row in X[hold]])
    r2_phys = float(r2_score(Y[hold], phys))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(model, args.out)
    size_mb = os.path.getsize(args.out) / 1e6

    report = {
        "source": source,
        "rows": int(len(df)),
        "holdout_rows": int(n_hold),
        "features": FEATURE_COLS,
        "targets": TARGET_COLS,
        "ridge_alpha": args.alpha,
        "model": "Pipeline(PolynomialFeatures(degree=2, interaction_only) -> Ridge)",
        "heldout_r2_ridge": r2,
        "heldout_rmse_km_ridge": rmse_km,
        "heldout_r2_physics_baseline": r2_phys,
        "model_path": args.out,
        "model_size_mb": round(size_mb, 4),
        "size_ok_lt_10mb": size_mb < MAX_MODEL_MB,
        "note": ("Ridge reproduces the physics formula from synthetic samples - pipeline "
                 "demonstration, not observed-drift skill." if source.startswith("SYNTHETIC")
                 else "Trained on observed displacements from data/features.csv."),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Held-out R2  - Ridge: {r2:.4f}   physics baseline: {r2_phys:.4f}")
    print(f"Held-out RMSE - Ridge: {rmse_km:.3f} km")
    print(f"Saved {args.out} ({size_mb:.3f} MB, {'OK' if size_mb < MAX_MODEL_MB else 'TOO BIG'} vs <{MAX_MODEL_MB:.0f} MB)")
    print(f"Report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
