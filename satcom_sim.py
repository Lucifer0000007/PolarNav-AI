"""
satcom_sim.py - Stage-safe satellite-pass mimic (SIH26059, Realtime Core M1).

Run as a SEPARATE process alongside demo.bat, e.g. in another terminal:
    python satcom_sim.py

NOT imported by the app. Every --interval seconds (default 20), copies the
next drops_stock/*.zip into drops_in/ under a fresh timestamped name,
cycling through the stock drops forever. Stdlib only.

Delivery is atomic (copy to a .part temp name, then os.rename into place)
so drop_watcher.py never sees a partially-written zip.

If drops_stock/ is missing or empty, prints a message and idles rather
than crashing -- absence of stock data must never raise.
"""
import argparse
import itertools
import os
import shutil
import sys
import time
from datetime import datetime, timezone

STOCK_DIR = "drops_stock"
IN_DIR = "drops_in"


def _idle_forever(reason: str):
    print(f"[satcom_sim] {reason} Idling (Ctrl+C to stop).")
    while True:
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=20.0, help="seconds between deliveries")
    args = ap.parse_args()

    if not os.path.isdir(STOCK_DIR):
        _idle_forever(f"{STOCK_DIR}/ not found -- run build_drops_stock.py first.")
        return

    stock = sorted(f for f in os.listdir(STOCK_DIR) if f.endswith(".zip"))
    if not stock:
        _idle_forever(f"{STOCK_DIR}/ has no .zip files -- run build_drops_stock.py first.")
        return

    os.makedirs(IN_DIR, exist_ok=True)
    print(f"[satcom_sim] cycling {len(stock)} stock drop(s) every {args.interval:.0f}s -> {IN_DIR}/")
    for name in itertools.cycle(stock):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(IN_DIR, f"{ts}_{name}")
        tmp_dest = dest + ".part"
        try:
            shutil.copy(os.path.join(STOCK_DIR, name), tmp_dest)
            os.rename(tmp_dest, dest)  # atomic on the same filesystem -- watcher never sees a partial file
            print(f"[satcom_sim] delivered {name} -> {dest}")
        except Exception as e:
            print(f"[satcom_sim] delivery failed ({name}): {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[satcom_sim] stopped.")
        sys.exit(0)
