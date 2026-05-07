from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .state import iso_now


@dataclass(frozen=True)
class ModuleProbe:
    node_id: str
    url: str
    timeout_seconds: float = 1.2
    ttl_ms: int = 5000


class ModuleStatusPoller:
    def __init__(
        self,
        probes: list[ModuleProbe],
        on_status: Callable[[dict], None],
        *,
        interval_seconds: float = 2.0,
    ) -> None:
        self.probes = list(probes)
        self.on_status = on_status
        self.interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="module-status-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for probe in self.probes:
                if self._stop_event.is_set():
                    break
                self.on_status(_probe_status(probe))
            self._stop_event.wait(self.interval_seconds)


def _probe_status(probe: ModuleProbe) -> dict:
    started = time.monotonic()
    observed_at = iso_now()
    try:
        request = urllib.request.Request(probe.url, headers={"Cache-Control": "no-store"})
        with urllib.request.urlopen(request, timeout=probe.timeout_seconds) as response:
            body = response.read(64 * 1024).decode("utf-8", errors="replace")
            latency_ms = int((time.monotonic() - started) * 1000)
            parsed = _try_parse_json(body)
            ok = 200 <= response.status < 400
            if isinstance(parsed, dict) and "ok" in parsed:
                ok = bool(parsed.get("ok"))
            return {
                "schema_version": 1,
                "node_id": probe.node_id,
                "status": "ok" if ok else "error",
                "phase": _phase_from_body(parsed) if ok else "http_error",
                "observed_at": observed_at,
                "ttl_ms": probe.ttl_ms,
                "detail": _detail_from_body(parsed, fallback=f"HTTP {response.status}"),
                "metrics": {
                    **_metrics_from_body(parsed),
                    "latency_ms": latency_ms,
                    "http_status": response.status,
                },
                "last_event": _last_event_from_body(parsed),
                "last_error": _last_error_from_body(parsed),
            }
    except urllib.error.HTTPError as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        return _error_status(probe, observed_at, latency_ms, f"HTTP {exc.code}", "http_error")
    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        return _error_status(probe, observed_at, latency_ms, str(exc), "unreachable")


def _error_status(
    probe: ModuleProbe,
    observed_at: str,
    latency_ms: int,
    detail: str,
    phase: str,
) -> dict:
    return {
        "schema_version": 1,
        "node_id": probe.node_id,
        "status": "offline",
        "phase": phase,
        "observed_at": observed_at,
        "ttl_ms": probe.ttl_ms,
        "detail": detail,
        "metrics": {
            "latency_ms": latency_ms,
            "http_status": None,
        },
        "last_event": None,
        "last_error": detail,
    }


def _try_parse_json(body: str) -> object:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _phase_from_body(body: object) -> str:
    if isinstance(body, dict):
        for key in ("phase", "status", "state"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    return "ready"


def _detail_from_body(body: object, *, fallback: str) -> str:
    if isinstance(body, dict):
        for key in ("detail", "message", "status"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    return fallback


def _metrics_from_body(body: object) -> dict:
    if isinstance(body, dict) and isinstance(body.get("metrics"), dict):
        return dict(body["metrics"])
    metrics: dict[str, object] = {}
    if isinstance(body, dict):
        for key in ("queue_depth", "queued", "uptime_s", "active_transcriptions"):
            if key in body:
                metrics[key] = body[key]
    return metrics


def _last_event_from_body(body: object) -> dict | None:
    if isinstance(body, dict) and isinstance(body.get("last_event"), dict):
        return dict(body["last_event"])
    return None


def _last_error_from_body(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    for key in ("last_error", "error"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    return None
