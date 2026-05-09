from __future__ import annotations

import hmac
import json
import ipaddress
import sys
import time
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .feedback import StateQueryFeedbackStore
from .state import EnvironmentStateStore

INDICATOR_CORS_ORIGINS = {
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:3002",
    "http://localhost:3002",
}


class EnvironmentHttpServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        store: EnvironmentStateStore,
        api_token: str,
        feedback_store: StateQueryFeedbackStore | None = None,
    ) -> None:
        super().__init__(server_address, EnvironmentRequestHandler)
        self.store = store
        self.api_token = api_token
        self.feedback_store = feedback_store

    def handle_error(self, request: object, client_address: object) -> None:
        _exc_type, exc, _traceback = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
            return
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) in {10053, 10054}:
            return
        super().handle_error(request, client_address)


class EnvironmentRequestHandler(BaseHTTPRequestHandler):
    server: EnvironmentHttpServer

    def do_OPTIONS(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/indicators/current" and self._local_request():
            self._send_json(
                HTTPStatus.NO_CONTENT,
                {},
                cors=True,
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        if path == "/health":
            self._send_json(HTTPStatus.OK, self.server.store.health())
            return
        if path == "/ready":
            payload = self.server.store.ready()
            self._send_json(
                HTTPStatus.OK if payload.get("ready") else HTTPStatus.SERVICE_UNAVAILABLE,
                payload,
            )
            return
        if path == "/environment/current":
            if not self._authorized():
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {
                        "ok": False,
                        "error": "unauthorized",
                    },
                )
                return
            query = parse_qs(parsed_url.query)
            try:
                payload = self._environment_current(query)
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                )
                return
            self._send_json(HTTPStatus.OK, payload)
            return
        if path == "/indicators/current":
            if not self._local_request():
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {
                        "ok": False,
                        "error": "local_access_required",
                    },
                )
                return
            self._send_json(HTTPStatus.OK, self.server.store.indicators_current(), cors=True)
            return
        if path == "/feedback/state-query/recent":
            if not self._authorized():
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {
                        "ok": False,
                        "error": "unauthorized",
                    },
                )
                return
            if self.server.feedback_store is None:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "ok": False,
                        "error": "feedback_store_unavailable",
                    },
                )
                return
            query = parse_qs(parsed_url.query)
            target = _first_query_value(query, "target")
            limit = _parse_limit(_first_query_value(query, "limit"))
            try:
                items = self.server.feedback_store.recent(target=target, limit=limit)
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "items": items,
                },
            )
            return
        if path == "/feedback/state-query/summary":
            if not self._authorized():
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {
                        "ok": False,
                        "error": "unauthorized",
                    },
                )
                return
            if self.server.feedback_store is None:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "ok": False,
                        "error": "feedback_store_unavailable",
                    },
                )
                return
            query = parse_qs(parsed_url.query)
            target = _first_query_value(query, "target")
            try:
                summary = self.server.feedback_store.summary(target=target)
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "summary": summary,
                },
            )
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {
                "ok": False,
                "error": "not_found",
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/feedback/state-query":
            if not self._authorized():
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {
                        "ok": False,
                        "error": "unauthorized",
                    },
                )
                return
            if self.server.feedback_store is None:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "ok": False,
                        "error": "feedback_store_unavailable",
                    },
                )
                return
            payload = self._read_json_body()
            if payload is None:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": "invalid_json",
                    },
                )
                return
            current = self.server.store.current()
            received_snapshot_id = str(current.get("snapshot_id") or "")
            try:
                record, duplicate = self.server.feedback_store.append(
                    payload,
                    received_snapshot_id=received_snapshot_id,
                )
            except ValueError as exc:
                self.server.feedback_store.record_rejected(payload)
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "feedback_id": record["feedback_id"],
                    "received_snapshot_id": record["received_snapshot_id"],
                    "duplicate": duplicate,
                    "status": record.get("status", "accepted"),
                    "warnings": record.get("warnings", []),
                },
            )
            return
        if path == "/environment/relations":
            if not self._authorized():
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {
                        "ok": False,
                        "error": "unauthorized",
                    },
                )
                return
            payload = self._read_json_body()
            if payload is None:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": "invalid_json",
                    },
                )
                return
            relations = self.server.store.update_relations(payload)
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "relations": relations,
                },
            )
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {
                "ok": False,
                "error": "not_found",
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _environment_current(self, query: dict[str, list[str]]) -> dict[str, Any]:
        wait_for = (_first_query_value(query, "wait_for") or "").strip()
        if not wait_for:
            return _attach_state_query_learning(
                self.server.store.current(),
                self.server.feedback_store,
            )
        if wait_for != "room_light":
            raise ValueError("unsupported_wait_for")

        after_text = (_first_query_value(query, "after") or "").strip()
        after = _parse_timestamp(after_text)
        timeout_ms = _parse_timeout_ms(_first_query_value(query, "timeout_ms"))
        started = time.monotonic()
        reason = ""

        while True:
            current = _attach_state_query_learning(
                self.server.store.current(),
                self.server.feedback_store,
            )
            matched, observed_at = _room_light_matches_after(current, after)
            if matched:
                reason = "matched"
                break
            if after is None:
                reason = "missing_or_invalid_after"
                break
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if elapsed_ms >= timeout_ms:
                reason = "timeout"
                break
            time.sleep(min(0.05, max(0.0, (timeout_ms - elapsed_ms) / 1000.0)))

        current["wait_result"] = {
            "target": "room_light",
            "after": after_text,
            "timeout_ms": timeout_ms,
            "matched": matched,
            "reason": reason,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "observed_at": observed_at,
        }
        return current

    def _authorized(self) -> bool:
        expected = self.server.api_token
        authorization = self.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer "):
            return hmac.compare_digest(authorization[7:].strip(), expected)
        header_token = self.headers.get("X-API-Token", "")
        return hmac.compare_digest(header_token.strip(), expected)

    def _local_request(self) -> bool:
        host, _port = self.client_address
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return host in {"localhost", ""}

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0 or length > 64 * 1024:
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _cors_headers(self) -> dict[str, str]:
        origin = self.headers.get("Origin", "")
        if origin not in INDICATOR_CORS_ORIGINS:
            return {}
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Vary": "Origin",
        }

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any], *, cors: bool = False) -> None:
        body = (
            b""
            if status == HTTPStatus.NO_CONTENT
            else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        if cors:
            for key, value in self._cors_headers().items():
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if status != HTTPStatus.NO_CONTENT:
            self.wfile.write(body)


def _first_query_value(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    if not values:
        return None
    return values[0]


def _parse_limit(value: str | None) -> int:
    if value is None:
        return 20
    try:
        return int(value)
    except ValueError:
        return 20


def _parse_timeout_ms(value: str | None) -> int:
    if value is None:
        return 1500
    try:
        parsed = int(value)
    except ValueError:
        return 1500
    return max(0, min(parsed, 3000))


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _room_light_matches_after(current: dict[str, Any], after: datetime | None) -> tuple[bool, str]:
    room_light = current.get("state_queries", {}).get("room_light")
    if not isinstance(room_light, dict):
        return False, ""
    observed_text = str(room_light.get("observed_at") or "")
    source_snapshot_id = str(room_light.get("source_snapshot_id") or "").strip()
    observed = _parse_timestamp(observed_text)
    if observed is None:
        return False, observed_text
    if after is None:
        return False, observed_text
    if not source_snapshot_id:
        return False, observed_text
    return observed > after, observed_text


def _attach_state_query_learning(
    current: dict[str, Any],
    feedback_store: StateQueryFeedbackStore | None,
) -> dict[str, Any]:
    if feedback_store is None:
        return current
    state_queries = current.get("state_queries")
    if not isinstance(state_queries, dict):
        return current
    room_light = state_queries.get("room_light")
    if not isinstance(room_light, dict):
        return current
    try:
        summary = feedback_store.summary(target="room_light")
        learning = summary.get("learning")
        recent_feedback = feedback_store.recent(target="room_light", limit=20)
    except Exception:
        return current
    if isinstance(learning, dict):
        room_light["learning"] = learning
        _attach_room_light_calibration(
            current=current,
            room_light=room_light,
            learning=learning,
            recent_feedback=recent_feedback,
        )
    return current


def _attach_room_light_calibration(
    *,
    current: dict[str, Any],
    room_light: dict[str, Any],
    learning: dict[str, Any],
    recent_feedback: list[dict[str, Any]],
) -> None:
    calibration = _room_light_calibration(
        current=current,
        room_light=room_light,
        learning=learning,
        recent_feedback=recent_feedback,
    )
    room_light["calibration"] = calibration
    if calibration.get("applied") is True:
        room_light["effective_state"] = calibration.get("state")
        room_light["effective_confidence_label"] = calibration.get("confidence_label")
        room_light["effective_authority"] = calibration.get("authority")
        room_light["effective_answer_hint"] = calibration.get("answer_hint")
    else:
        room_light["effective_state"] = room_light.get("state")
        room_light["effective_confidence_label"] = room_light.get("confidence_label")
        room_light["effective_authority"] = room_light.get("authority")
        room_light["effective_answer_hint"] = room_light.get("answer_hint")


def _room_light_calibration(
    *,
    current: dict[str, Any],
    room_light: dict[str, Any],
    learning: dict[str, Any],
    recent_feedback: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_state = _room_light_state(room_light.get("state"))
    raw_confidence = _confidence_label(room_light.get("confidence_label"))
    level_index = _int_value(learning.get("level_index"))
    base = {
        "applied": False,
        "state": raw_state,
        "confidence_label": raw_confidence,
        "authority": room_light.get("authority") or "vision_snapshot_processor",
        "reason": "",
        "raw_state": raw_state,
        "raw_confidence_label": raw_confidence,
        "learning_level": learning.get("level") or "none",
        "learning_level_index": level_index,
    }
    if room_light.get("available") is False or room_light.get("stale") is True:
        return {**base, "reason": "room_light_unavailable_or_stale"}
    if learning.get("ok") is False or level_index < 3:
        return {**base, "reason": "learning_not_usable"}
    if raw_state in {"on", "off"} and raw_confidence in {"medium", "high"}:
        quality = learning.get("prediction_quality")
        high_conflicts = 0
        if isinstance(quality, dict):
            high_conflicts = _int_value(quality.get("high_confidence_conflict_count"))
        reason = (
            "vision_confident_with_learning_conflicts"
            if high_conflicts > 0
            else "vision_confident"
        )
        return {**base, "reason": reason, "high_confidence_conflict_count": high_conflicts}

    snapshot_feedback = _matching_snapshot_feedback(room_light, recent_feedback)
    if snapshot_feedback:
        state = _room_light_state(snapshot_feedback.get("user_label"))
        if state in {"on", "off"}:
            return {
                **base,
                "applied": True,
                "state": state,
                "confidence_label": "high",
                "authority": "environment_state_server.calibration.user_feedback",
                "reason": "matching_snapshot_feedback",
                "feedback_id": snapshot_feedback.get("feedback_id") or "",
                "answer_hint": _calibrated_room_light_hint(state, "直近のユーザー訂正"),
            }

    appliance = _fresh_light_appliance(current.get("appliances"))
    if appliance:
        state = _room_light_state(appliance.get("state"))
        if state in {"on", "off"}:
            confidence = "high" if level_index >= 4 else "medium"
            return {
                **base,
                "applied": True,
                "state": state,
                "confidence_label": confidence,
                "authority": "environment_state_server.calibration.home_assistant",
                "reason": "fresh_home_assistant_light_state",
                "appliance_source": appliance.get("source") or "environment.appliances",
                "appliance_updated_at": appliance.get("updated_at"),
                "answer_hint": _calibrated_room_light_hint(state, "学習済みの操作履歴と Home Assistant の直近状態"),
            }

    return {**base, "reason": "no_supporting_calibration_signal"}


def _matching_snapshot_feedback(
    room_light: dict[str, Any],
    recent_feedback: list[dict[str, Any]],
) -> dict[str, Any] | None:
    snapshot_ids = {
        str(room_light.get("source_snapshot_id") or "").strip(),
        str(room_light.get("observed_at") or "").strip(),
        str(room_light.get("updated_at") or "").strip(),
    }
    snapshot_ids.discard("")
    if not snapshot_ids:
        return None
    for item in recent_feedback:
        if item.get("status") not in {"accepted", "accepted_with_warning"}:
            continue
        candidates = {
            str(item.get("snapshot_id") or "").strip(),
            str(item.get("current_snapshot_id") or "").strip(),
            str(item.get("received_snapshot_id") or "").strip(),
        }
        pending = item.get("pending") if isinstance(item.get("pending"), dict) else {}
        candidates.add(str(pending.get("source_snapshot_id") or "").strip())
        candidates.add(str(pending.get("observed_at") or "").strip())
        candidates.discard("")
        if snapshot_ids.intersection(candidates):
            return item
    return None


def _fresh_light_appliance(appliances: object) -> dict[str, Any] | None:
    if not isinstance(appliances, dict):
        return None
    for key in ("light", "living_room_light"):
        item = appliances.get(key)
        if not isinstance(item, dict):
            continue
        if item.get("stale") is True:
            continue
        if _room_light_state(item.get("state")) in {"on", "off"}:
            return item
    return None


def _calibrated_room_light_hint(state: str, source_label: str) -> str:
    if state == "on":
        return f"{source_label}では、照明は点いている可能性が高い。映像の生判定は補助情報として扱う。"
    if state == "off":
        return f"{source_label}では、照明は消えている可能性が高い。映像の生判定は補助情報として扱う。"
    return f"{source_label}では、照明状態を補正できない。"


def _room_light_state(value: object) -> str:
    state = str(value or "").strip().lower()
    return state if state in {"on", "off", "unknown"} else "unknown"


def _confidence_label(value: object) -> str:
    label = str(value or "").strip().lower()
    return label if label in {"none", "low", "medium", "high"} else "none"


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
