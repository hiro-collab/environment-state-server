from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .state import iso_now


@dataclass(frozen=True)
class HomeControlActionStateProbe:
    action_ids: tuple[str, ...]
    base_url: str
    api_token: str
    timeout_seconds: float = 1.2


class HomeControlActionStatePoller:
    def __init__(
        self,
        probe: HomeControlActionStateProbe,
        on_action_state: Callable[[dict], None],
        *,
        interval_seconds: float = 2.0,
    ) -> None:
        self.probe = probe
        self.on_action_state = on_action_state
        self.interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="home-control-state-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            state = probe_home_control_action_states(self.probe)
            if state is not None:
                self.on_action_state(state)
            self._stop_event.wait(self.interval_seconds)


def probe_home_control_action_states(probe: HomeControlActionStateProbe) -> dict | None:
    candidates: list[dict] = []
    for action_id in probe.action_ids:
        response = _probe_home_control_action_state(probe, action_id)
        if response is None:
            continue
        candidates.append(response)
        if response.get("status") == "matched" and response.get("actual_state"):
            break
    return _best_action_state(candidates)


def _probe_home_control_action_state(
    probe: HomeControlActionStateProbe,
    action_id: str,
) -> dict | None:
    if not action_id:
        return None
    url = _action_state_url(probe.base_url, action_id)
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {probe.api_token}",
            "Cache-Control": "no-store",
        },
    )
    observed_at = iso_now()
    try:
        with urllib.request.urlopen(request, timeout=probe.timeout_seconds) as response:
            body = response.read(64 * 1024).decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                return None
            parsed["observed_at"] = observed_at
            return parsed
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {
            "action_id": action_id,
            "status": "poll_error",
            "state_tracking": "unknown",
            "verification_mode": "unknown",
            "state_authority": "unknown",
            "observed_at": observed_at,
        }


def _best_action_state(candidates: list[dict]) -> dict | None:
    readable = [item for item in candidates if str(item.get("actual_state") or "").strip()]
    for item in readable:
        if item.get("status") == "matched":
            return item
    if readable:
        return readable[0]
    return candidates[0] if candidates else None


def _action_state_url(base_url: str, action_id: str) -> str:
    base = base_url.rstrip("/") + "/"
    quoted = urllib.parse.quote(action_id, safe="")
    return urllib.parse.urljoin(base, f"actions/{quoted}/state")
