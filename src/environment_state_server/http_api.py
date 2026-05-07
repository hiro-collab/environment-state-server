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
            return self.server.store.current()
        if wait_for != "room_light":
            raise ValueError("unsupported_wait_for")

        after_text = (_first_query_value(query, "after") or "").strip()
        after = _parse_timestamp(after_text)
        timeout_ms = _parse_timeout_ms(_first_query_value(query, "timeout_ms"))
        started = time.monotonic()
        reason = ""

        while True:
            current = self.server.store.current()
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
