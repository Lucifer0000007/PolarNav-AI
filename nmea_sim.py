"""
nmea_sim.py - Stage-safe GPS telemetry mimic (SIH26059, Realtime Core M2).

Run as a SEPARATE process alongside demo.bat:
    python nmea_sim.py

NOT imported by the app. Stdlib TCP server on 127.0.0.1:10110, emitting
$GPGGA and $GPRMC sentences at 1 Hz along a canned track that walks
DEMO_START -> DEMO_GOAL and back (a ping-pong loop, so motion is always
continuous, never a discontinuous jump) -- reusing engine.py's own
grid_to_latlon so the track always sits inside the same AOI the rest of
the demo uses. engine.py itself is only read from, never modified.

Loopback-only by construction (127.0.0.1) -- this never counts as an
"external request" for the project's offline/Wi-Fi-off guarantee.

Accepts one client at a time; a client that reads a bit and disconnects
(the app's usual pattern -- a short probe-and-read, not a held-open
connection) just causes the next accept() to fire immediately. The
internal position index is server-lifetime, not per-connection, so the
simulated vessel keeps moving regardless of how many short connections
tap into the feed.
"""
import socket
import sys
import time
from datetime import datetime, timezone

from engine import DEMO_START, DEMO_GOAL, grid_to_latlon

HOST, PORT = "127.0.0.1", 10110
RATE_HZ = 1.0
# DEMO_START->DEMO_GOAL is ~117.7 km one-way; 600 points gives ~0.2 km per
# 1 Hz tick, so the app's 2 km drift-hysteresis fires roughly every 10s of
# continuous motion -- fast enough to see within a short demo, slow enough
# that the hysteresis visibly throttles replans instead of firing on every
# single tick (the original 30-point track moved ~4 km/tick, which defeated
# the hysteresis entirely -- verified live before this fix).
N_WAYPOINTS = 600


def _build_track():
    lat0, lon0 = grid_to_latlon(*DEMO_START)
    lat1, lon1 = grid_to_latlon(*DEMO_GOAL)
    forward = [
        (lat0 + (lat1 - lat0) * i / (N_WAYPOINTS - 1), lon0 + (lon1 - lon0) * i / (N_WAYPOINTS - 1))
        for i in range(N_WAYPOINTS)
    ]
    return forward + forward[-2:0:-1]  # ping-pong: forward then back, no duplicated endpoints


TRACK = _build_track()


def _checksum(body: str) -> str:
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"{cs:02X}"


def _nmea_lat(lat: float):
    hemi = "N" if lat >= 0 else "S"
    lat = abs(lat)
    deg = int(lat)
    minutes = (lat - deg) * 60.0
    return f"{deg:02d}{minutes:07.4f}", hemi


def _nmea_lon(lon: float):
    hemi = "E" if lon >= 0 else "W"
    lon = abs(lon)
    deg = int(lon)
    minutes = (lon - deg) * 60.0
    return f"{deg:03d}{minutes:07.4f}", hemi


def build_gpgga(t: datetime, lat: float, lon: float) -> str:
    lat_s, lat_h = _nmea_lat(lat)
    lon_s, lon_h = _nmea_lon(lon)
    body = (f"GPGGA,{t.strftime('%H%M%S.00')},{lat_s},{lat_h},{lon_s},{lon_h},"
            f"1,08,0.9,10.0,M,30.0,M,,")
    return f"${body}*{_checksum(body)}\r\n"


def build_rmc(t: datetime, lat: float, lon: float, speed_kn: float = 8.0, course: float = 45.0) -> str:
    lat_s, lat_h = _nmea_lat(lat)
    lon_s, lon_h = _nmea_lon(lon)
    body = (f"GPRMC,{t.strftime('%H%M%S.00')},A,{lat_s},{lat_h},{lon_s},{lon_h},"
            f"{speed_kn:.1f},{course:.1f},{t.strftime('%d%m%y')},,,A")
    return f"${body}*{_checksum(body)}\r\n"


def serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[nmea_sim] listening on {HOST}:{PORT}, {len(TRACK)}-point ping-pong track")
    idx = 0
    while True:
        conn, addr = srv.accept()
        conn.settimeout(5.0)
        try:
            while True:
                lat, lon = TRACK[idx % len(TRACK)]
                idx += 1
                t = datetime.now(timezone.utc)
                conn.sendall(build_gpgga(t, lat, lon).encode("ascii"))
                conn.sendall(build_rmc(t, lat, lon).encode("ascii"))
                time.sleep(1.0 / RATE_HZ)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass  # client disconnected or went quiet -- normal for a short probe-and-read client
        except Exception as e:
            print(f"[nmea_sim] client loop error: {e}")
        finally:
            try:
                conn.close()
            except OSError:
                pass


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        print("\n[nmea_sim] stopped.")
        sys.exit(0)
