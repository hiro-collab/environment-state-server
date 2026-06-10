from __future__ import annotations

import hashlib
import json
import threading
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from .actions import build_action_registry


SCHEMA_VERSION = 1
HOME_ASSISTANT_SOURCE = "home_assistant"
HOME_ASSISTANT_EVENTS_SOURCE = "home_assistant_events"
HOME_ASSISTANT_BRIDGE_SOURCE = "home_assistant_bridge"
CAMERA_HUB_SOURCE = "camera_hub"
VISION_SNAPSHOT_PROCESSOR_SOURCE = "vision_snapshot_processor"
CAPABILITIES = {
    "actions": True,
    "relations": True,
    "ready": True,
    "indicators": True,
    "camera_hub_snapshots": True,
    "vision_topic_snapshots": True,
    "freshness": True,
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
            "dify_issue_id": None,
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
        allowed = {"dify_issue_id", "ha_request_id", "ha_execution_id", "snapshot_id"}
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
            updated = parse_optional_timestamp(item.get("updated_at"))
            item["stale"] = True if updated is None else _is_stale(updated, current_time, ttl_seconds)
            item["freshness"] = _freshness(updated, current_time, ttl_seconds)

        state_queries = _state_queries_from_vision(vision)
        actions = build_action_registry(appliances)
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
            "actions": actions,
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
                "actions": environment.get("actions", []),
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
        actions = current.get("actions", [])
        reasons: list[str] = []
        if not (bool(ha_source.get("available")) and not bool(ha_source.get("stale"))):
            reasons.append("home_assistant_bridge_unavailable_or_stale")
        if not actions:
            reasons.append("action_registry_empty")
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
    if _effect_requires_external_observation(effect):
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


def _effect_requires_external_observation(effect: dict[str, Any] | None) -> bool:
    if not effect:
        return False
    physical_state_source = str(effect.get("physical_state_source") or "").strip()
    verification_mode = str(effect.get("verification_mode") or "").strip()
    state_authority = str(effect.get("state_authority") or "").strip()
    evidence_class = str(effect.get("evidence_class") or "").strip()
    return (
        physical_state_source == "not_supported"
        or verification_mode == "external_observation"
        or verification_mode == "command_ack_only"
        or state_authority == "open_loop"
        or state_authority == "submitted_only"
        or evidence_class in {"action_event_only", "command_ack_only"}
    )


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


def _state_queries_from_vision(vision: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "room_light": _room_light_state_query(vision.get("room_light")),
    }


def _room_light_state_query(room_light: object) -> dict[str, Any]:
    if not isinstance(room_light, dict):
        return {
            "available": False,
            "stale": True,
            "stale_reason": "room_light_missing",
            "state": "unknown",
            "confidence_label": "none",
            "answer_hint": "現在のセンサー情報では確認できない。",
            "authority": VISION_SNAPSHOT_PROCESSOR_SOURCE,
            "projected_by": "environment_state_server",
            "observed_at": None,
            "updated_at": None,
            "source_snapshot_id": "",
            "freshness": {
                "level": "stale",
                "age_ms": None,
                "ttl_ms": 0,
                "updated_at": None,
                "reason": "room_light_missing",
            },
            "evidence": {
                "reason": "room_light_missing",
                "topic": "/vision/room_light/state",
            },
        }

    stale = bool(room_light.get("stale", True))
    state = _room_light_state(room_light.get("state"))
    confidence = _optional_float(room_light.get("confidence"))
    lighting_type = _room_light_text(room_light.get("lighting_type"), default="unknown")
    daylight_state = _room_light_text(room_light.get("daylight_state"), default="unknown")
    electric_on_probability = _room_light_probability(room_light, "electric_on", "electric_light")
    daylight_present_probability = _room_light_probability(room_light, "daylight_present", "daylight")
    dark_probability = _room_light_probability(room_light, "dark", None)

    evidence = {
        "source": _room_light_text(room_light.get("source"), default="unknown"),
        "topic": _room_light_text(room_light.get("topic"), default="/vision/room_light/state"),
        "lighting_type": lighting_type,
        "daylight_state": daylight_state,
        "electric_on_probability": electric_on_probability,
        "daylight_present_probability": daylight_present_probability,
        "dark_probability": dark_probability,
        "confidence": confidence,
        "observed_at": room_light.get("observed_at"),
        "updated_at": room_light.get("updated_at"),
        "model": room_light.get("model") if isinstance(room_light.get("model"), dict) else {},
        "sequence": room_light.get("sequence") if isinstance(room_light.get("sequence"), dict) else {},
    }
    freshness = room_light.get("freshness") if isinstance(room_light.get("freshness"), dict) else {}

    return {
        "available": not stale,
        "stale": stale,
        "stale_reason": "room_light_stale" if stale else "",
        "state": state,
        "confidence_label": _room_light_confidence_label(confidence, available=not stale),
        "answer_hint": _room_light_answer_hint(
            available=not stale,
            stale=stale,
            state=state,
            lighting_type=lighting_type,
            daylight_state=daylight_state,
        ),
        "authority": VISION_SNAPSHOT_PROCESSOR_SOURCE,
        "projected_by": "environment_state_server",
        "observed_at": room_light.get("observed_at"),
        "updated_at": room_light.get("updated_at"),
        "source_snapshot_id": _room_light_source_snapshot_id(room_light),
        "freshness": deepcopy(freshness),
        "evidence": evidence,
    }


def _room_light_state(value: object) -> str:
    state = str(value or "").strip().lower()
    return state if state in {"on", "off", "unknown"} else "unknown"


def _room_light_text(value: object, *, default: str) -> str:
    text = str(value or "").strip()
    return text if text else default


def _room_light_probability(
    room_light: dict[str, Any],
    probability_key: str,
    fallback_group: str | None,
) -> float | None:
    probabilities = room_light.get("probabilities") if isinstance(room_light.get("probabilities"), dict) else {}
    probability = _optional_float(probabilities.get(probability_key))
    if probability is not None or fallback_group is None:
        return probability
    group = room_light.get(fallback_group) if isinstance(room_light.get(fallback_group), dict) else {}
    return _optional_float(group.get("probability"))


def _room_light_source_snapshot_id(room_light: dict[str, Any]) -> str:
    observation_id = str(room_light.get("observation_id") or "").strip()
    if observation_id:
        return observation_id
    camera_frame_id = str(room_light.get("camera_frame_id") or "").strip()
    frame_id = room_light.get("frame_id")
    if camera_frame_id and frame_id is not None:
        return f"{camera_frame_id}:{frame_id}"
    sequence = room_light.get("sequence") if isinstance(room_light.get("sequence"), dict) else {}
    last_frame_id = sequence.get("last_frame_id")
    if last_frame_id is not None:
        return f"frame:{last_frame_id}"
    updated_at = str(room_light.get("updated_at") or "").strip()
    return updated_at


def _room_light_confidence_label(confidence: float | None, *, available: bool) -> str:
    if not available:
        return "none"
    value = confidence or 0.0
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def _room_light_answer_hint(
    *,
    available: bool,
    stale: bool,
    state: str,
    lighting_type: str,
    daylight_state: str,
) -> str:
    if not available:
        if stale:
            return "現在のセンサー情報では確認できない。直近の映像推定が古くなっています。"
        return "現在のセンサー情報では確認できない。"
    if state == "on":
        return "照明が点いている可能性が高い。"
    if state == "off":
        return "照明が消えている可能性が高い。"

    lighting = lighting_type.lower()
    daylight = daylight_state.lower()
    if lighting == "daylight" or daylight == "present":
        return "映像推定では断定できない。日光の影響が強そう。"
    if lighting == "mixed":
        return "映像推定では断定できない。日光と照明が混ざっている可能性がある。"
    if lighting == "dark":
        return "映像推定では断定できないが、暗い状態として観測されている。"
    if lighting == "electric":
        return "映像推定では照明らしい成分はあるが、断定できない。"
    return "映像推定では断定できない。"


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
    if topic == "/vision/room_light/state":
        electric = payload.get("electric_light") if isinstance(payload.get("electric_light"), dict) else {}
        daylight = payload.get("daylight") if isinstance(payload.get("daylight"), dict) else {}
        sequence = payload.get("sequence") if isinstance(payload.get("sequence"), dict) else {}
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
        return "room_light", {
            **base,
            "state": electric.get("state") or payload.get("state") or payload.get("label") or "unknown",
            "label": payload.get("label") or "unknown",
            "confidence": _optional_float(payload.get("confidence")),
            "lighting_type": payload.get("lighting_type") or "unknown",
            "probabilities": payload.get("probabilities") if isinstance(payload.get("probabilities"), dict) else {},
            "electric_light": electric,
            "daylight": daylight,
            "daylight_state": daylight.get("state") or "unknown",
            "observed_at": payload.get("observed_at") or to_iso(observed),
            "observation_id": payload.get("observation_id"),
            "sequence": {
                "frame_count": sequence.get("frame_count"),
                "first_frame_id": sequence.get("first_frame_id"),
                "last_frame_id": sequence.get("last_frame_id"),
                "temporal_window_ms": sequence.get("temporal_window_ms"),
            },
            "model": payload.get("model") if isinstance(payload.get("model"), dict) else evidence.get("model", {}),
            "evidence": evidence,
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
