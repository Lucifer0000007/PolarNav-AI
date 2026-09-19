"""
build_drops_stock.py - One-time authoring tool for drops_stock/*.zip
(SIH26059, Realtime Core M1).

NOT run by the app, NOT run by satcom_sim.py -- run manually whenever the
committed stock drops need to be (re)generated. Reuses engine.py's own
synthetic-SAR generator and the project's existing data/ CSVs, so the
stock drops are consistent with everything else the demo already ships.

Produces drops_stock/drop_00{1,2,3}.zip, each containing:
    sar.png            - synthetic SAR patch (engine.make_synthetic_sar, a fixed seed)
    icebergs.csv       - copy of data/icebergs.csv
    wind_current.csv   - copy of data/wind_current.csv
    manifest.sha256     - sha256sum-style per-file checksums of the 3 files above

drop_watcher.py verifies every extracted file against manifest.sha256
before accepting a delivered drop as valid.

Usage:
    python build_drops_stock.py
"""
import hashlib
import os
import shutil
import zipfile

from engine import make_synthetic_sar

STOCK_DIR = "drops_stock"
SEEDS = [101, 202, 303]  # 3 distinct, deterministic synthetic SAR fields


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_one(index: int, seed: int, tmp_dir: str) -> str:
    sar_path = os.path.join(tmp_dir, "sar.png")
    make_synthetic_sar(sar_path, size=400, seed=seed)

    icebergs_path = os.path.join(tmp_dir, "icebergs.csv")
    wind_path = os.path.join(tmp_dir, "wind_current.csv")
    shutil.copy("data/icebergs.csv", icebergs_path)
    shutil.copy("data/wind_current.csv", wind_path)

    manifest_path = os.path.join(tmp_dir, "manifest.sha256")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for name in ("sar.png", "icebergs.csv", "wind_current.csv"):
            f.write(f"{_sha256(os.path.join(tmp_dir, name))}  {name}\n")

    zip_path = os.path.join(STOCK_DIR, f"drop_{index:03d}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("sar.png", "icebergs.csv", "wind_current.csv", "manifest.sha256"):
            zf.write(os.path.join(tmp_dir, name), arcname=name)
    return zip_path


def main():
    os.makedirs(STOCK_DIR, exist_ok=True)
    tmp_dir = os.path.join(STOCK_DIR, "_build_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        zip_paths = [_build_one(i, seed, tmp_dir) for i, seed in enumerate(SEEDS, start=1)]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total = 0
    for zp in zip_paths:
        size = os.path.getsize(zp)
        total += size
        print(f"built {zp} ({size} bytes)")
    print(f"total drops_stock/ size: {total} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
