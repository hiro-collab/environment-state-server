from __future__ import annotations

import hashlib
import json
import math
import threading
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = 1
HOME_ASSISTANT_SOURCE = "home_assistant"
HOME_ASSISTANT_EVENTS_SOURCE = "home_assistant_events"
HOME_ASSISTANT_BRIDGE_SOURCE = "home_assistant_bridge"
HOME_CONTROL_READONLY_SOURCE = "home_control_readonly"
CAMERA_HUB_SOURCE = "camera_hub"
VISION_SNAPSHOT_PROCESSOR_SOURCE = "vision_snapshot_processor"
ROOM_LIGHT_TOPIC = "/vision/room_light/observation"
ROOM_LIGHT_MSG_TYPE = "vision_snapshot_processor/RoomLightObservation"
ROOM_LIGHT_MODEL_NAME = "room-light-heuristic-snapshot-v3"
ROOM_LIGHT_MODEL_KIND = "heuristic"
ROOM_LIGHT_SOURCE_CLASS = "camera_environment_estimate"
ROOM_LIGHT_PROOF_CEILING = "camera_environment_estimate_only"
ROOM_LIGHT_DOES_NOT_PROVE = (
    "physical_room_light_state",
    "home_assistant_light_state",
)
ROOM_LIGHT_IDENTIFIER_MAX_LENGTH = 160
ROOM_LIGHT_HEADER_SEQUENCE_MAX = 2**63 - 1
AIRCON_STATUS_QUERY_ID = "aircon_current_status"
AIRCON_STATUS_SOURCE_CLASS = "Home_Control_HA_visible_tracked_climate"
AIRCON_STATUS_AUTHORITY = "home_control_ha_readonly_climate"
AIRCON_STATUS_PROOF_CEILING = "HA_visible_climate_state_only"
CAPABILITIES = {
    "relations": True,
    "ready": True,
    "indicators": True,
    "camera_hub_snapshots": True,
    "vision_topic_snapshots": True,
    "freshness": True,
    AIRCON_STATUS_QUERY_ID: True,
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_timestamp(value: object, *, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return fallback or utc_now()


def parse_optional_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    sentinel = datetime(1970, 1, 1, tzinfo=UTC)
    parsed = parse_timestamp(value, fallback=sentinel)
    return None if parsed == sentinel else parsed


def to_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class EnvironmentStateStore:
    def __init__(self, *, ttl_ms: int = 5000, max_events: int = 20) -> None:
        if ttl_ms <= 0:
            raise ValueError("ttl_ms must be greater than 0")
        if max_events <= 0:
            raise ValueError("max_events must be greater than 0")
        self.ttl_ms = int(ttl_ms)
        self.max_events = int(max_events)
        self._lock = threading.RLock()
        self._appliances: dict[str, dict[str, Any]] = {}
        self._vision: dict[str, dict[str, Any]] = {}
        self._nodes: dict[str, dict[str, Any]] = {}
        self._last_home_assistant_events: list[dict[str, Any]] = []
        self._source_updates: dict[str, datetime] = {}
        self._source_errors: dict[str, str | None] = {}
        self._snapshot_sequence = 0
        self._relations: dict[str, object] = {
            "ha_request_id": None,
            "ha_execution_id": None,
            "snapshot_id": None,
        }

    def ingest_home_assistant_event(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        observed = parse_timestamp(event.get("timestamp") or event.get("issued_at"))
        summary = _home_assistant_event_summary(event, observed)
        appliance = _appliance_from_home_assistant_event(event, observed)

        with self._lock:
            self._source_updates[HOME_ASSISTANT_EVENTS_SOURCE] = observed
            self._source_errors[HOME_ASSISTANT_EVENTS_SOURCE] = None
            self._last_home_assistant_events.append(summary)
            self._last_home_assistant_events = self._last_home_assistant_events[-self.max_events :]
            if appliance is not None:
                key, payload = appliance
                self._appliances[key] = payload

    def ingest_home_control_action_state(self, action_state: dict[str, Any]) -> None:
        if not isinstance(action_state, dict):
            return
        observed = parse_timestamp(
            action_state.get("observed_at")
            or action_state.get("checked_at")
            or action_state.get("timestamp")
        )
        appliance = _appliance_from_home_control_action_state(action_state, observed)
        last_error = _home_control_action_state_error(action_state, appliance is not None)

        with self._lock:
            self._source_updates[HOME_CONTROL_READONLY_SOURCE] = observed
            self._source_errors[HOME_CONTROL_READONLY_SOURCE] = last_error
            if appliance is not None:
                key, payload = appliance
                self._appliances[key] = payload

    def ingest_node_status(self, status: dict[str, Any]) -> None:
        if not isinstance(status, dict):
            return
        node_id = str(status.get("node_id") or "").strip()
        if not node_id:
            return
        observed = parse_timestamp(status.get("observed_at"))
        payload = {
            "schema_version": _optional_int(status.get("schema_version")) or SCHEMA_VERSION,
            "node_id": node_id,
            "status": str(status.get("status") or "unknown"),
            "phase": status.get("phase"),
            "observed_at": to_iso(observed),
            "ttl_ms": _optional_int(status.get("ttl_ms")) or self.ttl_ms,
            "detail": status.get("detail"),
            "metrics": status.get("metrics") if isinstance(status.get("metrics"), dict) else {},
            "last_event": status.get("last_event") if isinstance(status.get("last_event"), dict) else None,
            "last_error": status.get("last_error"),
        }
        with self._lock:
            self._nodes[node_id] = payload
            if node_id == HOME_ASSISTANT_BRIDGE_SOURCE:
                self._source_updates[HOME_ASSISTANT_BRIDGE_SOURCE] = observed
                self._source_errors[HOME_ASSISTANT_BRIDGE_SOURCE] = (
                    None
                    if payload["status"] == "ok"
                    else str(payload.get("last_error") or payload.get("detail") or "unhealthy")
                )

    def ingest_camera_hub_envelope(self, envelope: dict[str, Any]) -> None:
        self.ingest_vision_envelope(envelope, source=CAMERA_HUB_SOURCE)

    def ingest_vision_envelope(
        self,
        envelope: dict[str, Any],
        *,
        source: str = CAMERA_HUB_SOURCE,
    ) -> None:
        if not isinstance(envelope, dict):
            return
        topic = str(envelope.get("topic") or "")
        header = envelope.get("header") if isinstance(envelope.get("header"), dict) else {}
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
        if topic == ROOM_LIGHT_TOPIC:
            room_light = _canonical_room_light_observation(envelope, payload, source=source)
            if room_light is None:
                return
            observed, source_id, value = room_light
            update = "room_light", value
        else:
            observed = parse_timestamp(header.get("stamp") if isinstance(header, dict) else None)
            if isinstance(header, dict) and isinstance(header.get("stamp"), (int, float)):
                observed = datetime.fromtimestamp(float(header["stamp"]), tz=UTC)

            source_id = _source_id(source)
            update = _vision_update_from_topic(topic, payload, header, observed, source=source_id)
        if update is None:
            return
        key, value = update
        with self._lock:
            self._source_updates[source_id] = observed
            self._source_errors[source_id] = None
            self._vision[key] = value

    def update_relations(self, relations: dict[str, Any]) -> dict[str, object]:
        allowed = {"ha_request_id", "ha_execution_id", "snapshot_id"}
        with self._lock:
            for key in allowed:
                if key in relations:
                    value = relations[key]
                    self._relations[key] = None if value is None else str(value)
            self._relations["updated_at"] = iso_now()
            return deepcopy(self._relations)

    def current(self, *, now: datetime | None = None) -> dict[str, Any]:
        current_time = now or utc_now()
        ttl_seconds = self.ttl_ms / 1000.0
        with self._lock:
            self._snapshot_sequence += 1
            sequence = self._snapshot_sequence
            appliances = deepcopy(self._appliances)
            vision = deepcopy(self._vision)
            events = deepcopy(self._last_home_assistant_events)
            source_updates = dict(self._source_updates)
            source_errors = dict(self._source_errors)
            nodes = deepcopy(self._nodes)
            relations = deepcopy(self._relations)

        observed_at = _latest_timestamp(
            [
                *(
                    parsed
                    for item in appliances.values()
                    if (parsed := parse_optional_timestamp(item.get("updated_at"))) is not None
                ),
                *(
                    parsed
                    for item in vision.values()
                    if (parsed := parse_optional_timestamp(item.get("updated_at"))) is not None
                ),
                *source_updates.values(),
            ]
        )
        stale = observed_at is None or _is_stale(observed_at, current_time, ttl_seconds)
        observed_iso = to_iso(observed_at) if observed_at is not None else None
        age_ms = _age_ms(observed_at, current_time)
        snapshot_id = _snapshot_id(current_time, sequence)

        for item in appliances.values():
            updated = parse_optional_timestamp(item.get("updated_at"))
            item["stale"] = True if updated is None else _is_stale(updated, current_time, ttl_seconds)
            item["freshness"] = _freshness(updated, current_time, ttl_seconds)
        for item in vision.values():
            updated = parse_optional_timestamp(item.get("observed_at") or item.get("updated_at"))
            item["stale"] = True if updated is None else _is_stale(updated, current_time, ttl_seconds)
            item["freshness"] = _freshness(updated, current_time, ttl_seconds)

        state_queries = _state_queries_from_environment(appliances)
        sources = {
            HOME_ASSISTANT_SOURCE: _node_source_state(
                nodes.get(HOME_ASSISTANT_BRIDGE_SOURCE),
                current_time,
                ttl_seconds,
                fallback=_source_state(
                    source_updates.get(HOME_ASSISTANT_EVENTS_SOURCE),
                    current_time,
                    ttl_seconds,
                    last_error=source_errors.get(HOME_ASSISTANT_EVENTS_SOURCE),
                ),
            ),
            HOME_ASSISTANT_EVENTS_SOURCE: _source_state(
                source_updates.get(HOME_ASSISTANT_EVENTS_SOURCE),
                current_time,
                ttl_seconds,
                last_error=source_errors.get(HOME_ASSISTANT_EVENTS_SOURCE),
            ),
            CAMERA_HUB_SOURCE: _source_state(
                source_updates.get(CAMERA_HUB_SOURCE),
                current_time,
                ttl_seconds,
                last_error=source_errors.get(CAMERA_HUB_SOURCE),
            ),
        }
        for source_id in sorted(source_updates):
            if source_id not in sources:
                sources[source_id] = _source_state(
                    source_updates.get(source_id),
                    current_time,
                    ttl_seconds,
                    last_error=source_errors.get(source_id),
                )

        return {
            "schema_version": SCHEMA_VERSION,
            "capabilities": deepcopy(CAPABILITIES),
            "snapshot_id": snapshot_id,
            "sequence": sequence,
            "observed_at": observed_iso,
            "age_ms": age_ms,
            "stale": stale,
            "freshness": _freshness(observed_at, current_time, ttl_seconds),
            "ttl_ms": self.ttl_ms,
            "appliances": appliances,
            "vision": vision,
            "state_queries": state_queries,
            "last_home_assistant_events": events,
            "relations": relations,
            "sources": sources,
        }

    def indicators_current(self, *, now: datetime | None = None) -> dict[str, Any]:
        current_time = now or utc_now()
        environment = self.current(now=current_time)
        with self._lock:
            nodes = deepcopy(self._nodes)

        nodes["environment_state_server"] = {
            "schema_version": SCHEMA_VERSION,
            "node_id": "environment_state_server",
            "status": "ok",
            "phase": "serving",
            "observed_at": to_iso(current_time),
            "ttl_ms": self.ttl_ms,
            "detail": "ready",
            "metrics": {},
            "last_event": None,
        }
        for node in nodes.values():
            updated = parse_optional_timestamp(node.get("observed_at"))
            ttl_ms = _optional_int(node.get("ttl_ms")) or self.ttl_ms
            node["stale"] = True if updated is None else _is_stale(updated, current_time, ttl_ms / 1000.0)
            if node["stale"] and node.get("status") == "ok":
                node["status"] = "stale"

        return {
            "schema_version": SCHEMA_VERSION,
            "capabilities": deepcopy(CAPABILITIES),
            "snapshot_id": environment.get("snapshot_id"),
            "sequence": environment.get("sequence"),
            "observed_at": environment.get("observed_at"),
            "age_ms": environment.get("age_ms"),
            "stale": bool(environment.get("stale", True)),
            "environment": {
                "ttl_ms": environment.get("ttl_ms"),
                "appliances": environment.get("appliances", {}),
                "vision": environment.get("vision", {}),
                "state_queries": environment.get("state_queries", {}),
                "sources": environment.get("sources", {}),
            },
            "nodes": nodes,
        }

    def health(self, *, now: datetime | None = None) -> dict[str, Any]:
        current_time = now or utc_now()
        current = self.current(now=current_time)
        return {
            "ok": True,
            "status": "ok",
            "checked_at": to_iso(current_time),
            "schema_version": SCHEMA_VERSION,
            "capabilities": deepcopy(CAPABILITIES),
            "snapshot_id": current.get("snapshot_id"),
            "sources": current.get("sources", {}),
            "nodes": self.indicators_current(now=current_time).get("nodes", {}),
        }

    def ready(self, *, now: datetime | None = None) -> dict[str, Any]:
        current_time = now or utc_now()
        current = self.current(now=current_time)
        sources = current.get("sources", {})
        ha_source = sources.get(HOME_ASSISTANT_SOURCE, {})
        camera_hub_source = sources.get(CAMERA_HUB_SOURCE, {})
        reasons: list[str] = []
        if not (bool(ha_source.get("available")) and not bool(ha_source.get("stale"))):
            reasons.append("home_assistant_bridge_unavailable_or_stale")
        if not (bool(camera_hub_source.get("available")) and not bool(camera_hub_source.get("stale"))):
            reasons.append("camera_hub_stale")
        ready = not reasons
        return {
            "ok": ready,
            "ready": ready,
            "status": "ready" if ready else "not_ready",
            "checked_at": to_iso(current_time),
            "schema_version": SCHEMA_VERSION,
            "capabilities": deepcopy(CAPABILITIES),
            "snapshot_id": current.get("snapshot_id"),
            "reasons": reasons,
            "sources": sources,
        }


def _latest_timestamp(values: list[datetime]) -> datetime | None:
    parsed = [value for value in values if isinstance(value, datetime)]
    return max(parsed) if parsed else None


def _is_stale(updated_at: datetime, now: datetime, ttl_seconds: float) -> bool:
    return (now - updated_at).total_seconds() > ttl_seconds


def _age_ms(updated_at: datetime | None, now: datetime) -> int | None:
    if updated_at is None:
        return None
    return max(0, int((now - updated_at).total_seconds() * 1000))


def _freshness(
    updated_at: datetime | None,
    now: datetime,
    ttl_seconds: float,
) -> dict[str, Any]:
    ttl_ms = max(0, int(ttl_seconds * 1000))
    age_ms = _age_ms(updated_at, now)
    if updated_at is None or age_ms is None:
        return {
            "level": "stale",
            "age_ms": None,
            "ttl_ms": ttl_ms,
            "updated_at": None,
            "reason": "missing_timestamp",
        }
    if age_ms > ttl_ms:
        level = "stale"
        reason = "age_exceeds_ttl"
    elif age_ms <= max(1, int(ttl_ms * 0.4)):
        level = "fresh"
        reason = "age_within_fresh_window"
    elif age_ms <= max(1, int(ttl_ms * 0.8)):
        level = "recent"
        reason = "age_within_recent_window"
    else:
        level = "usable"
        reason = "age_within_usable_window"
    return {
        "level": level,
        "age_ms": age_ms,
        "ttl_ms": ttl_ms,
        "updated_at": to_iso(updated_at),
        "reason": reason,
    }


def _snapshot_id(now: datetime, sequence: int) -> str:
    return f"env_{now.strftime('%Y%m%d_%H%M%S')}_{sequence:06d}"


def _source_state(
    updated_at: datetime | None,
    now: datetime,
    ttl_seconds: float,
    *,
    last_error: str | None = None,
) -> dict[str, Any]:
    if updated_at is None:
        return {
            "available": False,
            "stale": True,
            "updated_at": None,
            "last_seen_at": None,
            "age_ms": None,
            "freshness": _freshness(None, now, ttl_seconds),
            "last_error": last_error,
        }
    stale = _is_stale(updated_at, now, ttl_seconds)
    return {
        "available": not stale,
        "stale": stale,
        "updated_at": to_iso(updated_at),
        "last_seen_at": to_iso(updated_at),
        "age_ms": _age_ms(updated_at, now),
        "freshness": _freshness(updated_at, now, ttl_seconds),
        "last_error": last_error,
    }


def _node_source_state(
    node: dict[str, Any] | None,
    now: datetime,
    ttl_seconds: float,
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if not node:
        return fallback
    updated_at = parse_optional_timestamp(node.get("observed_at"))
    ttl_ms = _optional_int(node.get("ttl_ms"))
    source_ttl_seconds = (ttl_ms / 1000.0) if ttl_ms else ttl_seconds
    stale = True if updated_at is None else _is_stale(updated_at, now, source_ttl_seconds)
    status = str(node.get("status") or "unknown")
    available = status == "ok" and not stale
    detail = node.get("detail")
    last_error = node.get("last_error")
    return {
        "available": available,
        "stale": stale,
        "updated_at": to_iso(updated_at) if updated_at is not None else None,
        "last_seen_at": to_iso(updated_at) if updated_at is not None else None,
        "age_ms": _age_ms(updated_at, now),
        "freshness": _freshness(updated_at, now, source_ttl_seconds),
        "last_error": None if available else str(last_error or detail or status),
        "status": status,
        "phase": node.get("phase"),
        "detail": detail,
    }


def _home_assistant_event_summary(event: dict[str, Any], observed: datetime) -> dict[str, Any]:
    event_id = event.get("event_id") or event.get("execution_id") or _stable_event_id(event)
    return {
        "ha_event_id": str(event_id),
        "action_id": event.get("action_id"),
        "execution_id": event.get("execution_id"),
        "request_id": event.get("request_id"),
        "event": event.get("event"),
        "status": _normalize_home_assistant_status(event),
        "message": event.get("message"),
        "occurred_at": to_iso(observed),
    }


def _stable_event_id(event: dict[str, Any]) -> str:
    digest = hashlib.sha1(
        json.dumps(event, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"ha_{digest}"


def _normalize_home_assistant_status(event: dict[str, Any]) -> str:
    status = event.get("status")
    if isinstance(status, str) and status:
        if event.get("event") == "execute_succeeded" and status == "submitted":
            return "succeeded"
        return status
    name = str(event.get("event") or "")
    if name == "execute_succeeded":
        return "succeeded"
    if name == "execute_failed":
        return "failed"
    if name == "execute_blocked_confirmation":
        return "confirmation_required"
    return "unknown"


def _appliance_from_home_assistant_event(
    event: dict[str, Any],
    observed: datetime,
) -> tuple[str, dict[str, Any]] | None:
    if event.get("executed") is not True:
        return None
    action_id = str(event.get("action_id") or "")
    effect = event.get("expected_effect") if isinstance(event.get("expected_effect"), dict) else None
    if effect is None:
        return None
    if not _effect_has_read_state_authority(effect):
        return None
    key = _appliance_key(action_id, effect)
    state = _appliance_state(action_id, effect)
    if key is None or state is None:
        return None

    payload: dict[str, Any] = {
        "state": state,
        "updated_at": to_iso(observed),
        "source": HOME_ASSISTANT_SOURCE,
        "action_id": action_id or None,
        "execution_id": event.get("execution_id"),
        "request_id": event.get("request_id"),
    }
    if effect is not None:
        payload["domain"] = effect.get("domain")
        payload["entity_id"] = effect.get("entity_id")
        payload["expected_state"] = effect.get("expected_state")
    return key, payload


def _appliance_from_home_control_action_state(
    action_state: dict[str, Any],
    observed: datetime,
) -> tuple[str, dict[str, Any]] | None:
    action_id = str(action_state.get("action_id") or "").strip()
    if not action_id:
        return None
    if not action_id.startswith("aircon_"):
        return None
    if str(action_state.get("state_tracking") or "").strip() != "tracked":
        return None
    if str(action_state.get("verification_mode") or "").strip() != "ha_state":
        return None
    if str(action_state.get("state_authority") or "").strip() != "ha_entity":
        return None

    actual_state = _aircon_state(action_state.get("actual_state"))
    if actual_state in {"unknown", "unavailable"}:
        return None

    expected_states = action_state.get("expected_states")
    safe_expected_states = [
        _aircon_state(item)
        for item in expected_states
        if _aircon_state(item) not in {"unknown", "unavailable"}
    ] if isinstance(expected_states, list) else []

    payload: dict[str, Any] = {
        "state": actual_state,
        "updated_at": to_iso(observed),
        "source": HOME_CONTROL_READONLY_SOURCE,
        "source_class": AIRCON_STATUS_SOURCE_CLASS,
        "action_id": action_id,
        "checkstate_status": _home_control_checkstate_status(action_state.get("status")),
        "expected_state": _aircon_state(action_state.get("expected_state")),
        "expected_states": safe_expected_states,
        "control_type": str(action_state.get("control_type") or "mode_command"),
        "state_authority": "ha_entity",
        "verification_mode": "ha_state",
        "state_tracking": "tracked",
        "proof_ceiling": str(action_state.get("proof_ceiling") or AIRCON_STATUS_PROOF_CEILING),
    }
    return "aircon", payload


def _home_control_action_state_error(
    action_state: dict[str, Any],
    registered: bool,
) -> str | None:
    if registered:
        return None
    status = _home_control_checkstate_status(action_state.get("status"))
    if status in {"poll_error", "unavailable", "unsupported", "ack_only", "external_required"}:
        return f"aircon_current_status_{status}"
    if not str(action_state.get("actual_state") or "").strip():
        return "aircon_current_status_missing_actual_state"
    return None


def _home_control_checkstate_status(value: object) -> str:
    status = str(value or "").strip().lower()
    return status if status else "unknown"


def _effect_has_read_state_authority(effect: dict[str, Any] | None) -> bool:
    if not effect:
        return False
    verification_mode = str(effect.get("verification_mode") or "").strip()
    state_authority = str(effect.get("state_authority") or "").strip()
    return verification_mode == "ha_state" and state_authority == "ha_entity"


def _appliance_key(action_id: str, effect: dict[str, Any] | None) -> str | None:
    lowered = action_id.lower()
    for prefix, key in (
        ("light_", "light"),
        ("fan_", "fan"),
        ("door_", "door"),
        ("curtain_", "curtain"),
        ("aircon_", "aircon"),
        ("vacuum_", "vacuum"),
    ):
        if lowered.startswith(prefix):
            return key
    if effect is not None:
        entity_id = str(effect.get("entity_id") or "")
        if "." in entity_id:
            return entity_id.split(".", 1)[1].replace("-", "_")
        domain = str(effect.get("domain") or "")
        return domain or None
    return None


def _appliance_state(action_id: str, effect: dict[str, Any] | None) -> str | None:
    if effect is not None:
        expected_state = str(effect.get("expected_state") or "").strip()
        if expected_state:
            return expected_state

    lowered = action_id.lower()
    if lowered.startswith("door_"):
        if lowered.endswith("_open"):
            return "open"
        if lowered.endswith("_close"):
            return "closed"
        if lowered.endswith("_stop"):
            return "stopped"
        return "unknown"
    if lowered.startswith("vacuum_"):
        if lowered.endswith("_start"):
            return "cleaning"
        if lowered.endswith("_return"):
            return "returning"
        if lowered.endswith("_pause"):
            return "paused"
        return "unknown"
    if lowered.endswith("_on"):
        return "on"
    if lowered.endswith("_off"):
        return "off"
    if lowered.endswith("_start"):
        return "on"
    if lowered.endswith("_pause"):
        return "stopped"
    if lowered.endswith("_return"):
        return "returning"
    if effect is not None and effect.get("expected_state") is not None:
        return str(effect["expected_state"])
    return None


def _state_queries_from_environment(
    appliances: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {AIRCON_STATUS_QUERY_ID: _aircon_status_query(appliances.get("aircon"))}


def _aircon_status_query(aircon: object) -> dict[str, Any]:
    base = {
        "schema_version": "environment_state_aircon_current_status.v0",
        "authority": AIRCON_STATUS_AUTHORITY,
        "projected_by": "environment_state_server",
        "source_class": AIRCON_STATUS_SOURCE_CLASS,
        "proof_ceiling": AIRCON_STATUS_PROOF_CEILING,
        "does_not_prove": [
            "physical_hvac_comfort",
            "physical_HVAC_cooling_or_comfort",
            "physical_cooling_or_airflow",
            "home_control_action_success",
            "command_acceptance",
            "home_control_pass",
            "rr003_review_ready_or_pass",
        ],
    }
    if not isinstance(aircon, dict):
        return {
            **base,
            "available": False,
            "stale": True,
            "stale_reason": "aircon_current_status_missing",
            "state": "unknown",
            "current_status": "unknown",
            "state_is_last_known": False,
            "appliance_family": "aircon",
            "status_availability_class": "unavailable",
            "status_match_class": "unknown",
            "freshness_class": "unavailable",
            "safe_wording_class": "current_ac_status_unavailable_must_revalidate_current_state",
            "status_label": "unavailable",
            "answer_hint": "現在のEnvironment Stateではエアコンの現在状態を確認できない。再取得が必要です。",
            "observed_at": None,
            "updated_at": None,
            "freshness": {
                "level": "stale",
                "age_ms": None,
                "ttl_ms": 0,
                "updated_at": None,
                "reason": "aircon_current_status_missing",
            },
            "evidence": {
                "reason": "aircon_current_status_missing",
                "source_class": AIRCON_STATUS_SOURCE_CLASS,
                "state_authority": "ha_entity",
                "verification_mode": "ha_state",
                "state_tracking": "tracked",
                "checkstate_class": "not_registered",
            },
        }

    stale = bool(aircon.get("stale", True))
    state = _aircon_state(aircon.get("state"))
    freshness = aircon.get("freshness") if isinstance(aircon.get("freshness"), dict) else {}
    checkstate_status = _home_control_checkstate_status(aircon.get("checkstate_status"))
    available = not stale and state not in {"unknown", "unavailable"}
    stale_reason = "aircon_current_status_stale" if stale else ""
    if state in {"unknown", "unavailable"}:
        available = False
        stale_reason = stale_reason or "aircon_current_status_unknown"
    status_availability_class = (
        "available_fresh"
        if available
        else "available_stale"
        if state not in {"unknown", "unavailable"}
        else "unavailable"
    )

    return {
        **base,
        "available": available,
        "stale": stale,
        "stale_reason": stale_reason,
        "state": state,
        "current_status": state,
        "state_is_last_known": bool(stale and state not in {"unknown", "unavailable"}),
        "appliance_family": "aircon",
        "status_availability_class": status_availability_class,
        "status_match_class": _aircon_status_match_class(
            action_id=str(aircon.get("action_id") or ""),
            state=state,
            checkstate_status=checkstate_status,
        ),
        "freshness_class": _aircon_freshness_class(freshness, available=available, stale=stale),
        "safe_wording_class": _aircon_safe_wording_class(
            available=available,
            stale=stale,
            state=state,
            checkstate_status=checkstate_status,
        ),
        "status_label": _aircon_status_label(state),
        "answer_hint": _aircon_status_answer_hint(
            available=available,
            stale=stale,
            state=state,
        ),
        "observed_at": aircon.get("updated_at"),
        "updated_at": aircon.get("updated_at"),
        "freshness": deepcopy(freshness),
        "evidence": {
            "source": str(aircon.get("source") or HOME_CONTROL_READONLY_SOURCE),
            "source_class": str(aircon.get("source_class") or AIRCON_STATUS_SOURCE_CLASS),
            "checkstate_status": checkstate_status,
            "expected_state": _aircon_state(aircon.get("expected_state")),
            "expected_states": [
                _aircon_state(item)
                for item in aircon.get("expected_states", [])
                if _aircon_state(item) not in {"unknown", "unavailable"}
            ] if isinstance(aircon.get("expected_states"), list) else [],
            "control_type": str(aircon.get("control_type") or "mode_command"),
            "state_authority": "ha_entity",
            "verification_mode": "ha_state",
            "state_tracking": "tracked",
            "proof_ceiling": str(aircon.get("proof_ceiling") or AIRCON_STATUS_PROOF_CEILING),
        },
    }


def _aircon_state(value: object) -> str:
    state = str(value or "").strip().lower()
    aliases = {
        "オフ": "off",
        "冷房": "cool",
        "暖房": "heat",
        "送風": "fan_only",
        "除湿": "dry",
        "unavailable": "unavailable",
        "unknown": "unknown",
    }
    state = aliases.get(state, state)
    allowed = {"off", "cool", "heat", "dry", "fan_only", "auto", "heat_cool", "unknown", "unavailable"}
    return state if state in allowed else "unknown"


def _aircon_status_label(state: str) -> str:
    return {
        "off": "off",
        "cool": "cooling_mode",
        "heat": "heating_mode",
        "dry": "dry_mode",
        "fan_only": "fan_only_mode",
        "auto": "auto_mode",
        "heat_cool": "heat_cool_mode",
        "unavailable": "unavailable",
        "unknown": "unknown",
    }.get(state, "unknown")


def _aircon_status_match_class(
    *,
    action_id: str,
    state: str,
    checkstate_status: str,
) -> str:
    if checkstate_status == "mismatch":
        return "mismatch"
    if checkstate_status != "matched":
        return "unknown"
    if action_id == "aircon_restore_original":
        return "restore_original_matched"
    if state == "off":
        return "off_matched"
    if state == "cool":
        return "cool_matched"
    return "unknown"


def _aircon_freshness_class(
    freshness: dict[str, Any],
    *,
    available: bool,
    stale: bool,
) -> str:
    if available:
        return "fresh"
    if stale and freshness:
        return "stale"
    return "unavailable"


def _aircon_safe_wording_class(
    *,
    available: bool,
    stale: bool,
    state: str,
    checkstate_status: str,
) -> str:
    if available:
        return "HA_visible_aircon_status_class_available"
    if stale and state not in {"unknown", "unavailable"}:
        return "last_known_aircon_status_only_must_revalidate"
    return "current_ac_status_unavailable_must_revalidate_current_state"


def _aircon_status_answer_hint(
    *,
    available: bool,
    stale: bool,
    state: str,
) -> str:
    if stale and state not in {"unknown", "unavailable"}:
        return "最後に分かっている範囲ではエアコン状態を読めるが、現在状態としては古いため再取得が必要です。"
    if not available:
        return "現在のEnvironment Stateではエアコンの現在状態を確認できない。再取得が必要です。"
    if state == "off":
        return "Home Control/HAの読み取り専用状態では、エアコンはオフ扱いです。"
    if state == "cool":
        return "Home Control/HAの読み取り専用状態では、エアコンは冷房扱いです。"
    if state == "heat":
        return "Home Control/HAの読み取り専用状態では、エアコンは暖房扱いです。"
    return "Home Control/HAの読み取り専用状態でエアコン状態を要約しています。"


def _canonical_room_light_observation(
    envelope: dict[str, Any],
    payload: dict[str, Any],
    *,
    source: object,
) -> tuple[datetime, str, dict[str, Any]] | None:
    if type(envelope.get("schema_version")) is not int or envelope["schema_version"] != SCHEMA_VERSION:
        return None
    if envelope.get("topic") != ROOM_LIGHT_TOPIC:
        return None
    if envelope.get("msg_type") != ROOM_LIGHT_MSG_TYPE:
        return None
    header = envelope.get("header")
    if not isinstance(header, dict):
        return None
    header_sequence = header.get("seq")
    if (
        type(header_sequence) is not int
        or header_sequence < 0
        or header_sequence > ROOM_LIGHT_HEADER_SEQUENCE_MAX
    ):
        return None
    received_at = _room_light_numeric_timestamp(header.get("stamp"))
    camera_frame_id = _canonical_room_light_text(header.get("frame_id"))
    if received_at is None or camera_frame_id is None:
        return None
    source_id = source if source in {CAMERA_HUB_SOURCE, VISION_SNAPSHOT_PROCESSOR_SOURCE} else None
    if source_id is None:
        return None

    if payload.get("type") != "room_light_observation":
        return None
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != SCHEMA_VERSION:
        return None
    if payload.get("observation_bucket") not in {"dark", "dim", "balanced", "bright"}:
        return None
    if not _is_unit_interval_number(payload.get("confidence")):
        return None
    if payload.get("daylight_ambiguity") not in {"low", "medium", "high"}:
        return None
    cues = payload.get("cue_likelihoods")
    if not isinstance(cues, dict):
        return None
    if not all(
        cue in cues and _is_unit_interval_number(cues[cue])
        for cue in ("warm_light", "daylight", "darkness")
    ):
        return None
    if payload.get("source") != VISION_SNAPSHOT_PROCESSOR_SOURCE:
        return None
    if payload.get("source_class") != ROOM_LIGHT_SOURCE_CLASS:
        return None
    observed_at = _room_light_observed_at(payload.get("observed_at"))
    observation_id = _canonical_room_light_text(payload.get("observation_id"))
    if observed_at is None or observation_id is None:
        return None

    sequence = payload.get("sequence")
    if not isinstance(sequence, dict):
        return None
    frame_count = sequence.get("frame_count")
    first_frame_id = sequence.get("first_frame_id")
    last_frame_id = sequence.get("last_frame_id")
    temporal_window_ms = sequence.get("temporal_window_ms")
    if type(frame_count) is not int or frame_count <= 0:
        return None
    if type(first_frame_id) is not int or first_frame_id < 0:
        return None
    if type(last_frame_id) is not int or last_frame_id < 0:
        return None
    if type(temporal_window_ms) is not int or temporal_window_ms < 0:
        return None
    if first_frame_id > last_frame_id or frame_count > last_frame_id - first_frame_id + 1:
        return None

    model = payload.get("model")
    if not isinstance(model, dict):
        return None
    if model.get("name") != ROOM_LIGHT_MODEL_NAME or model.get("kind") != ROOM_LIGHT_MODEL_KIND:
        return None
    if payload.get("proof_ceiling") != ROOM_LIGHT_PROOF_CEILING:
        return None
    if payload.get("does_not_prove") != list(ROOM_LIGHT_DOES_NOT_PROVE):
        return None

    source_snapshot_id = observation_id
    return received_at, source_id, {
        "updated_at": to_iso(received_at),
        "source": source_id,
        "topic": ROOM_LIGHT_TOPIC,
        "frame_id": header_sequence,
        "camera_frame_id": camera_frame_id,
        "schema_version": SCHEMA_VERSION,
        "type": "room_light_observation",
        "observation_bucket": payload["observation_bucket"],
        "confidence": payload["confidence"],
        "daylight_ambiguity": payload["daylight_ambiguity"],
        "cue_likelihoods": {
            "warm_light": cues["warm_light"],
            "daylight": cues["daylight"],
            "darkness": cues["darkness"],
        },
        "source_class": ROOM_LIGHT_SOURCE_CLASS,
        "observed_at": to_iso(observed_at),
        "observation_id": observation_id,
        "source_snapshot_id": source_snapshot_id,
        "sequence": {
            "frame_count": frame_count,
            "first_frame_id": first_frame_id,
            "last_frame_id": last_frame_id,
            "temporal_window_ms": temporal_window_ms,
        },
        "model": {
            "name": ROOM_LIGHT_MODEL_NAME,
            "kind": ROOM_LIGHT_MODEL_KIND,
        },
        "provenance": {
            "source": source_id,
            "source_class": ROOM_LIGHT_SOURCE_CLASS,
            "producer": VISION_SNAPSHOT_PROCESSOR_SOURCE,
            "topic": ROOM_LIGHT_TOPIC,
            "source_snapshot_id": source_snapshot_id,
        },
        "proof_ceiling": ROOM_LIGHT_PROOF_CEILING,
        "does_not_prove": list(ROOM_LIGHT_DOES_NOT_PROVE),
    }


def _is_unit_interval_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0 <= value <= 1 and math.isfinite(value)


def _room_light_numeric_timestamp(value: object) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        stamp = float(value)
        if not math.isfinite(stamp):
            return None
        return datetime.fromtimestamp(stamp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _room_light_observed_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (OverflowError, ValueError):
        return None


def _canonical_room_light_text(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > ROOM_LIGHT_IDENTIFIER_MAX_LENGTH:
        return None
    text = value.strip()
    if not text or not all(char.isprintable() for char in text):
        return None
    return text


def _vision_update_from_topic(
    topic: str,
    payload: dict[str, Any],
    header: dict[str, Any],
    observed: datetime,
    *,
    source: str = CAMERA_HUB_SOURCE,
) -> tuple[str, dict[str, Any]] | None:
    base = {
        "updated_at": to_iso(observed),
        "source": _source_id(source),
        "topic": topic,
        "frame_id": header.get("seq"),
        "camera_frame_id": header.get("frame_id"),
    }
    if topic == "/vision/sword_sign/state":
        gestures = payload.get("gestures", {})
        if not isinstance(gestures, dict):
            gestures = {}
        signal = gestures.get("sword_sign")
        stable = payload.get("stable") if isinstance(payload.get("stable"), dict) else {}
        best_gesture = _best_gesture(gestures)
        return "sword_sign", {
            **base,
            "hand_detected": bool(payload.get("hand_detected")),
            "primary_gesture": payload.get("primary") or payload.get("primary_gesture"),
            "best_gesture": best_gesture,
            "gestures": gestures,
            "active": signal.get("active") if isinstance(signal, dict) else None,
            "confidence": signal.get("confidence") if isinstance(signal, dict) else None,
            "stable": stable,
        }
    if topic == "/camera/status":
        return "camera", {
            **base,
            "state": "available" if payload.get("camera", {}).get("opened") else "unavailable",
            "fps": payload.get("fps"),
            "camera": payload.get("camera") if isinstance(payload.get("camera"), dict) else {},
            "capture": payload.get("capture") if isinstance(payload.get("capture"), dict) else {},
        }
    return None


def _source_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return CAMERA_HUB_SOURCE
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)


def _best_gesture(gestures: dict[str, Any]) -> dict[str, Any] | None:
    best_name = None
    best_payload: dict[str, Any] | None = None
    best_confidence = -1.0
    for name, value in gestures.items():
        if not isinstance(value, dict):
            continue
        confidence = _optional_float(value.get("confidence"))
        if confidence is None:
            continue
        if confidence > best_confidence:
            best_name = str(name)
            best_payload = value
            best_confidence = confidence
    if best_name is None or best_payload is None or best_confidence <= 0.0:
        return None
    return {
        "name": best_name,
        "label": best_payload.get("label"),
        "active": bool(best_payload.get("active")),
        "confidence": best_confidence,
    }


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
