"""
drop_watcher.py - Validates simulated satellite drops (SIH26059, Realtime
Core M1).

Run as a SEPARATE process alongside demo.bat and satcom_sim.py:
    python drop_watcher.py

NOT imported by the app. Stdlib polling only (no watchdog dependency).
Polls drops_in/ for new zips; verifies each extracted file's sha256
against the zip's own manifest.sha256 sidecar, plus basic lat/lon and
wind-speed range checks reusing engine.py's existing AOI constants
(read-only import -- engine.py itself is never modified).

Pass -> files moved into drops_done/<batch_id>/, drops_done/latest_receipt.json
        written, bus.publish("drops.validated", ...).
Fail -> zip's contents moved into drops_quarantine/<batch_id>/ with a
        reason.txt, bus.publish("drop.rejected", ...).

Never crashes on one bad zip -- any exception while processing a single
drop is treated as a rejection of that drop; the watcher loop continues.
bus.py may not exist yet (it's a separate, later step in this build) --
the import is optional, exactly like every other optional dependency in
this project; publish() calls simply no-op until bus.py is present.
"""
import csv
import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone

import cv2

from engine import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

try:
    import bus
except Exception:
    bus = None

IN_DIR = "drops_in"
DONE_DIR = "drops_done"
QUARANTINE_DIR = "drops_quarantine"
POLL_SECONDS = 1.0
AOI_MARGIN_DEG = 0.5   # generous sanity margin, not exact-AOI enforcement
WIND_SPEED_MAX_MS = 60.0  # m/s -- catches garbage data, not a real physical model


def _publish(topic: str, payload: dict):
    if bus is None:
        return
    try:
        bus.publish(topic, payload)
    except Exception:
        pass  # publish must never take the watcher down


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksums(extract_dir: str):
    manifest_path = os.path.join(extract_dir, "manifest.sha256")
    if not os.path.exists(manifest_path):
        raise ValueError("manifest.sha256 missing from drop")
    with open(manifest_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        raise ValueError("manifest.sha256 is empty")
    for line in lines:
        expected, name = line.split("  ", 1)
        path = os.path.join(extract_dir, name)
        if not os.path.exists(path):
            raise ValueError(f"{name} listed in manifest but missing from zip")
        if _sha256(path) != expected:
            raise ValueError(f"{name} checksum mismatch")


def _range_check(extract_dir: str):
    icebergs_path = os.path.join(extract_dir, "icebergs.csv")
    with open(icebergs_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lat, lon = float(row["lat"]), float(row["lon"])
            if not (LAT_MIN - AOI_MARGIN_DEG <= lat <= LAT_MAX + AOI_MARGIN_DEG
                    and LON_MIN - AOI_MARGIN_DEG <= lon <= LON_MAX + AOI_MARGIN_DEG):
                raise ValueError(f"iceberg {row.get('id')} lat/lon ({lat},{lon}) far outside AOI")

    wind_path = os.path.join(extract_dir, "wind_current.csv")
    with open(wind_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("u_wind", "v_wind", "u_current", "v_current"):
                if key in row and row[key] not in (None, "") and abs(float(row[key])) > WIND_SPEED_MAX_MS:
                    raise ValueError(f"{key}={row[key]} exceeds sanity bound ({WIND_SPEED_MAX_MS} m/s)")

    sar_path = os.path.join(extract_dir, "sar.png")
    img = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise ValueError("sar.png did not decode as a valid image")


def _process_one(zip_path: str):
    batch_id = os.path.splitext(os.path.basename(zip_path))[0]
    extract_dir = os.path.join(IN_DIR, f"_extract_{batch_id}")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        _verify_checksums(extract_dir)
        _range_check(extract_dir)

        os.makedirs(DONE_DIR, exist_ok=True)
        dest_dir = os.path.join(DONE_DIR, batch_id)
        if os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.move(extract_dir, dest_dir)

        receipt = {
            "batch_id": batch_id,
            "validated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "sar_path": os.path.join(dest_dir, "sar.png"),
            "icebergs_csv": os.path.join(dest_dir, "icebergs.csv"),
            "wind_csv": os.path.join(dest_dir, "wind_current.csv"),
        }
        with open(os.path.join(DONE_DIR, "latest_receipt.json"), "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        _publish("drops.validated", receipt)
        print(f"[drop_watcher] PASS {batch_id} -> {dest_dir}")
    except Exception as e:
        os.makedirs(QUARANTINE_DIR, exist_ok=True)
        q_dir = os.path.join(QUARANTINE_DIR, batch_id)
        if os.path.isdir(q_dir):
            shutil.rmtree(q_dir)
        if os.path.isdir(extract_dir):
            shutil.move(extract_dir, q_dir)
        else:
            os.makedirs(q_dir, exist_ok=True)
        with open(os.path.join(q_dir, "reason.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))
        _publish("drop.rejected", {"batch_id": batch_id, "reason": str(e)})
        print(f"[drop_watcher] REJECT {batch_id}: {e}")
    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass


def main():
    os.makedirs(IN_DIR, exist_ok=True)
    print(f"[drop_watcher] watching {IN_DIR}/ every {POLL_SECONDS:.0f}s "
          f"(bus: {'available' if bus is not None else 'not yet available -- publishes will no-op'})")
    while True:
        try:
            for name in sorted(os.listdir(IN_DIR)):
                if name.endswith(".zip"):
                    _process_one(os.path.join(IN_DIR, name))
        except Exception as e:
            print(f"[drop_watcher] loop error (continuing): {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[drop_watcher] stopped.")
