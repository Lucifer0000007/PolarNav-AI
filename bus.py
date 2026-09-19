"""
bus.py - Stage-safe event bus (SIH26059, Realtime Core M3).

publish(topic, payload) tries Kafka ONLY if kafka-python-ng is importable
AND a broker is reachable (a fast loopback-only probe, never a public
host); otherwise, and ALWAYS in addition when Kafka does work, appends
one JSON line to events.log. This is a deliberate dual-write, not a
fallback-only design: the local event tail always works for the UI
regardless of which transport happens to be live, so judges never see an
empty "Event Stream" expander just because Kafka was up that run.

Never raises. The Kafka attempt is tried exactly once per process (the
result is cached), so a missing broker costs one fast timeout, not a
retry on every single publish call. kafka-python-ng is never added to
requirements.txt -- this entire module works with zero extra packages
installed.
"""
import json
import os
import socket
import time

EVENTS_LOG = "events.log"
KAFKA_HOST = "localhost"   # loopback-only by design -- never a public broker
KAFKA_PORT = 9092
KAFKA_TOPIC_PREFIX = "polarnav."
PROBE_TIMEOUT = 0.2

_mode = None          # "kafka" | "file" | None (not yet determined)
_producer = None       # cached KafkaProducer, or None


def _try_kafka_once():
    """Attempt to import kafka-python-ng and reach a broker exactly once
    per process. Returns True (and caches a producer) on success, False
    otherwise. Any exception at any stage is treated as 'use file mode'."""
    global _mode, _producer
    try:
        with socket.create_connection((KAFKA_HOST, KAFKA_PORT), timeout=PROBE_TIMEOUT):
            pass
    except Exception:
        _mode = "file"
        return False
    try:
        from kafka import KafkaProducer
        _producer = KafkaProducer(
            bootstrap_servers=f"{KAFKA_HOST}:{KAFKA_PORT}",
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=1000,
        )
        _mode = "kafka"
        return True
    except Exception:
        _mode = "file"
        _producer = None
        return False


def _append_to_file(topic: str, payload: dict):
    try:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "topic": topic, "payload": payload}
        with open(EVENTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # ultimate last resort: publish silently no-ops rather than raising


def transport_mode() -> str:
    """Read-only status for M4's Production Mimic dots. Never probes;
    just reports whatever the first publish() call (if any) determined.
    Returns 'file' if no publish has happened yet this process (the
    honest default -- file mode is always what's actually available)."""
    return _mode or "file"


def publish(topic: str, payload: dict) -> None:
    """Never raises. Always leaves a trace in events.log; additionally
    sends to Kafka when a broker is reachable and kafka-python-ng is
    installed."""
    global _mode
    if _mode is None:
        _try_kafka_once()

    if _mode == "kafka" and _producer is not None:
        try:
            _producer.send(f"{KAFKA_TOPIC_PREFIX}{topic}", payload)
            _producer.flush(timeout=1.0)
        except Exception:
            # A broker that was up at probe time but drops mid-session must
            # never take the caller down -- fall through to the file write,
            # which happens unconditionally below regardless of this branch.
            pass

    _append_to_file(topic, payload)


def tail(n: int = 20):
    """Last n JSON-decoded events.log lines, oldest first. Returns [] if
    the file doesn't exist yet or any line fails to parse -- never
    raises. Used by app.py's Event Stream expander."""
    if not os.path.exists(EVENTS_LOG):
        return []
    try:
        with open(EVENTS_LOG, encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
    except Exception:
        return []
