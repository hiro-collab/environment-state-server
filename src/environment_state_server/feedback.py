from __future__ import annotations

import json
import re
import threading
import unicodedata
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ALLOWED_STATE_QUERY_TARGETS = {"room_light"}
ALLOWED_USER_LABELS = {"on", "off", "daylight", "unknown"}
FEEDBACK_RESULT_STATUSES = ("accepted", "accepted_with_warning", "duplicate", "rejected")
MAX_USER_TEXT_LENGTH = 500
MAX_STRING_LENGTH = 2000
MAX_RECENT_LIMIT = 100
MAX_PENDING_AGE_SECONDS = 120
LEARNING_LEVELS = (
    "none",
    "collecting",
    "seeded",
    "usable",
    "reinforced",
)


class StateQueryFeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._runtime_counts: dict[str, dict[str, int]] = {}

    def append(
        self,
        payload: dict[str, Any],
        *,
        received_snapshot_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        record = self._normalize(payload, received_snapshot_id=received_snapshot_id)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            idempotency_key = record.get("idempotency_key")
            if isinstance(idempotency_key, str) and idempotency_key:
                existing = self._find_by_idempotency_key_unlocked(idempotency_key)
                if existing is not None:
                    self._increment_runtime_count_unlocked("duplicate", str(existing.get("target") or ""))
                    return existing, True
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(line + "\n")
        return deepcopy(record), False

    def record_rejected(self, payload: dict[str, Any] | None = None) -> None:
        target = ""
        if isinstance(payload, dict):
            target = _optional_identifier(payload.get("target") or payload.get("state_query_id")) or ""
            if target not in ALLOWED_STATE_QUERY_TARGETS:
                target = ""
        with self._lock:
            self._increment_runtime_count_unlocked("rejected", target)

    def recent(self, *, target: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        target_id = _optional_identifier(target)
        if target_id is not None and target_id not in ALLOWED_STATE_QUERY_TARGETS:
            raise ValueError("unsupported_target")
        count = max(1, min(int(limit), MAX_RECENT_LIMIT))
        if not self.path.exists():
            return []

        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()

        items: list[dict[str, Any]] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if target_id is not None and item.get("target") != target_id:
                continue
            items.append(item)
            if len(items) >= count:
                break
        return items

    def summary(self, *, target: str | None = None) -> dict[str, Any]:
        target_id = _optional_identifier(target)
        if target_id is not None and target_id not in ALLOWED_STATE_QUERY_TARGETS:
            raise ValueError("unsupported_target")
        items = self._read_records(target=target_id)
        recent_items = items[-20:]
        label_counts = _count_by_key(items, "user_label")
        reason_counts = _count_by_key(items, "feedback_reason")
        source_context_counts = _count_by_key(items, "source_context")
        action_counts = _count_by_key(items, "action_id")
        expected_state_counts = _count_by_key(items, "expected_state")
        status_counts = _with_status_keys(_count_by_key(items, "status"))
        runtime_counts = self._runtime_counts_for(target_id)
        for status, count in runtime_counts.items():
            status_counts[status] = status_counts.get(status, 0) + count
        recent_status_counts = _with_status_keys(_count_by_key(recent_items, "status"))
        latest = items[-1] if items else {}
        learning = _learning_summary(
            target=target_id or "",
            items=items,
            label_counts=label_counts,
            reason_counts=reason_counts,
            source_context_counts=source_context_counts,
            status_counts=status_counts,
        )
        return {
            "target": target_id or "",
            "total_count": len(items),
            "label_counts": label_counts,
            "reason_counts": reason_counts,
            "source_context_counts": source_context_counts,
            "action_counts": action_counts,
            "expected_state_counts": expected_state_counts,
            "status_counts": status_counts,
            "runtime_status_counts": runtime_counts,
            "recent_window": len(recent_items),
            "recent_status_counts": recent_status_counts,
            "latest_received_at": latest.get("received_at", ""),
            "latest_feedback_id": latest.get("feedback_id", ""),
            "learning": learning,
        }

    def _normalize(
        self,
        payload: dict[str, Any],
        *,
        received_snapshot_id: str | None,
    ) -> dict[str, Any]:
        target = _required_identifier(payload.get("target") or payload.get("state_query_id"), "target")
        if target not in ALLOWED_STATE_QUERY_TARGETS:
            raise ValueError("unsupported_target")

        state_query_id = _optional_identifier(payload.get("state_query_id")) or target
        if state_query_id != target:
            raise ValueError("state_query_id_mismatch")

        user_label = _required_identifier(payload.get("user_label"), "user_label")
        if user_label not in ALLOWED_USER_LABELS:
            raise ValueError("unsupported_user_label")

        now = datetime.now(UTC)
        received_at = now.isoformat()
        feedback_id = f"sqf_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        pending = payload.get("pending") if isinstance(payload.get("pending"), dict) else {}
        warnings = _feedback_warnings(payload, pending=pending, received_at=now)
        status = "accepted_with_warning" if warnings else "accepted"

        return {
            "schema_version": 1,
            "feedback_id": feedback_id,
            "received_at": received_at,
            "status": status,
            "warnings": warnings,
            "type": "state_query_feedback",
            "target": target,
            "state_query_id": state_query_id,
            "idempotency_key": _short_text(payload.get("idempotency_key"), max_length=200),
            "snapshot_id": _short_text(payload.get("snapshot_id"), max_length=160),
            "current_snapshot_id": _short_text(payload.get("current_snapshot_id"), max_length=160),
            "received_snapshot_id": _short_text(received_snapshot_id, max_length=160),
            "predicted_state": _room_light_state(payload.get("predicted_state")),
            "predicted_confidence_label": _confidence_label(payload.get("predicted_confidence_label")),
            "user_label": user_label,
            "user_text": _sanitize_user_text(payload.get("user_text")),
            "authority": "user_feedback",
            "source": _short_text(payload.get("source") or "dify", max_length=80),
            "workflow_version": _short_text(payload.get("workflow_version") or "unknown", max_length=160),
            "feedback_reason": _short_text(payload.get("feedback_reason"), max_length=160),
            "source_context": _short_text(payload.get("source_context"), max_length=120),
            "action_id": _short_text(payload.get("action_id"), max_length=120),
            "issue_id": _short_text(payload.get("issue_id"), max_length=160),
            "expected_state": _room_light_state(payload.get("expected_state")),
            "pending": _json_safe(pending, max_depth=8),
        }

    def _read_records(self, *, target: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if target is not None and item.get("target") != target:
                continue
            records.append(item)
        return records

    def _find_by_idempotency_key_unlocked(self, idempotency_key: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        for line in reversed(self.path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("idempotency_key") == idempotency_key:
                return deepcopy(item)
        return None

    def _increment_runtime_count_unlocked(self, status: str, target: str) -> None:
        for key in {"", target}:
            counts = self._runtime_counts.setdefault(key, {})
            counts[status] = counts.get(status, 0) + 1

    def _runtime_counts_for(self, target: str | None) -> dict[str, int]:
        key = target or ""
        with self._lock:
            counts = dict(self._runtime_counts.get(key, {}))
        return _with_status_keys(counts)


def _required_identifier(value: object, field_name: str) -> str:
    identifier = _optional_identifier(value)
    if identifier is None:
        raise ValueError(f"missing_{field_name}")
    return identifier


def _optional_identifier(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return re.sub(r"[^a-z0-9_-]+", "_", text)[:80]


def _room_light_state(value: object) -> str:
    state = _optional_identifier(value) or "unknown"
    return state if state in {"on", "off", "unknown"} else "unknown"


def _confidence_label(value: object) -> str:
    label = _optional_identifier(value) or "none"
    return label if label in {"none", "low", "medium", "high"} else "none"


def _learning_summary(
    *,
    target: str,
    items: list[dict[str, Any]],
    label_counts: dict[str, int],
    reason_counts: dict[str, int],
    source_context_counts: dict[str, int],
    status_counts: dict[str, int],
) -> dict[str, Any]:
    accepted_count = len(items)
    warning_count = sum(1 for item in items if item.get("status") == "accepted_with_warning")
    on_count = int(label_counts.get("on", 0))
    off_count = int(label_counts.get("off", 0))
    daylight_count = int(label_counts.get("daylight", 0))
    unknown_count = int(label_counts.get("unknown", 0))
    post_action_count = int(source_context_counts.get("post_light_action", 0))
    state_query_count = int(source_context_counts.get("state_query", 0))
    stale_ratio = (warning_count / accepted_count) if accepted_count else 0.0
    prediction_quality = _prediction_quality(items)

    level_index = 0
    if accepted_count >= 1:
        level_index = 1
    if accepted_count >= 3:
        level_index = 2
    if accepted_count >= 6 and on_count > 0 and off_count > 0:
        level_index = 3
    if accepted_count >= 12 and on_count > 0 and off_count > 0 and post_action_count >= 4:
        level_index = 4

    problems: list[dict[str, str]] = []
    if accepted_count == 0:
        problems.append(
            _learning_problem(
                "no_feedback",
                "error",
                "room_light のユーザー訂正 feedback がまだ保存されていません。",
            )
        )
    elif accepted_count < 3:
        problems.append(
            _learning_problem(
                "few_feedback",
                "warning",
                "feedback が少なく、学習材料としてはまだ初期段階です。",
            )
        )
    if accepted_count > 0 and on_count == 0:
        problems.append(
            _learning_problem(
                "missing_on_label",
                "warning",
                "点灯状態の user_label=on がまだありません。",
            )
        )
    if accepted_count > 0 and off_count == 0:
        problems.append(
            _learning_problem(
                "missing_off_label",
                "warning",
                "消灯状態の user_label=off がまだありません。",
            )
        )
    if accepted_count >= 3 and post_action_count == 0:
        problems.append(
            _learning_problem(
                "no_post_action_feedback",
                "warning",
                "操作後確認由来の source_context=post_light_action がまだありません。",
            )
        )
    if stale_ratio > 0.33:
        problems.append(
            _learning_problem(
                "stale_feedback_ratio_high",
                "warning",
                "古い pending に対する feedback の割合が高めです。",
            )
        )
    if int(status_counts.get("rejected", 0)) > 0:
        problems.append(
            _learning_problem(
                "rejected_feedback_seen",
                "error",
                "不正な feedback が reject されています。Dify の payload 正規化を確認してください。",
            )
        )
    if int(status_counts.get("duplicate", 0)) > 0:
        problems.append(
            _learning_problem(
                "duplicate_feedback_seen",
                "warning",
                "同じ idempotency_key の重複送信が検出されています。",
            )
        )
    if int(prediction_quality.get("high_confidence_conflict_count", 0)) > 0:
        problems.append(
            _learning_problem(
                "high_confidence_prediction_conflicts",
                "warning",
                "高確信のカメラ推定とユーザー訂正が衝突した記録があります。照明判定の閾値や撮影条件を見直してください。",
            )
        )

    return {
        "target": target,
        "level": LEARNING_LEVELS[level_index],
        "level_index": level_index,
        "accepted_count": accepted_count,
        "warning_count": warning_count,
        "stale_ratio": round(stale_ratio, 4),
        "post_action_count": post_action_count,
        "state_query_count": state_query_count,
        "label_balance": {
            "on": on_count,
            "off": off_count,
            "daylight": daylight_count,
            "unknown": unknown_count,
            "has_on": on_count > 0,
            "has_off": off_count > 0,
        },
        "reason_counts": dict(reason_counts),
        "source_context_counts": dict(source_context_counts),
        "prediction_quality": prediction_quality,
        "problems": problems,
        "ok": not any(problem["severity"] == "error" for problem in problems),
        "next_level_hint": _next_learning_level_hint(level_index),
    }


def _prediction_quality(items: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = 0
    match_count = 0
    conflict_count = 0
    high_confidence_count = 0
    high_confidence_conflict_count = 0
    recent_conflicts: list[dict[str, str]] = []
    for item in items:
        predicted = _room_light_state(item.get("predicted_state"))
        actual = _room_light_state(item.get("user_label"))
        confidence = _confidence_label(item.get("predicted_confidence_label"))
        if predicted not in {"on", "off"} or actual not in {"on", "off"}:
            continue
        comparable += 1
        high_confidence = confidence == "high"
        if high_confidence:
            high_confidence_count += 1
        if predicted == actual:
            match_count += 1
            continue
        conflict_count += 1
        if high_confidence:
            high_confidence_conflict_count += 1
        recent_conflicts.append(
            {
                "feedback_id": str(item.get("feedback_id") or ""),
                "received_at": str(item.get("received_at") or ""),
                "predicted_state": predicted,
                "predicted_confidence_label": confidence,
                "user_label": actual,
                "snapshot_id": str(item.get("snapshot_id") or ""),
            }
        )
    match_ratio = (match_count / comparable) if comparable else 0.0
    conflict_ratio = (conflict_count / comparable) if comparable else 0.0
    return {
        "comparable_count": comparable,
        "match_count": match_count,
        "conflict_count": conflict_count,
        "match_ratio": round(match_ratio, 4),
        "conflict_ratio": round(conflict_ratio, 4),
        "high_confidence_count": high_confidence_count,
        "high_confidence_conflict_count": high_confidence_conflict_count,
        "recent_conflicts": recent_conflicts[-5:],
    }


def _learning_problem(code: str, severity: str, message: str) -> dict[str, str]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
    }


def _next_learning_level_hint(level_index: int) -> str:
    if level_index <= 0:
        return "まず状態照会または操作後確認でユーザー訂正を1件保存してください。"
    if level_index == 1:
        return "最低3件の feedback を集めると seeded になります。"
    if level_index == 2:
        return "on/off の両方を含む6件以上の feedback で usable になります。"
    if level_index == 3:
        return "12件以上、かつ操作後確認由来が4件以上になると reinforced になります。"
    return "十分な初期学習材料があります。偏りや stale warning を継続監視してください。"


def _feedback_warnings(
    payload: dict[str, Any],
    *,
    pending: dict[str, Any],
    received_at: datetime,
) -> list[str]:
    reference = _parse_timestamp(pending.get("created_at"))
    if reference is None:
        reference = _parse_timestamp(pending.get("updated_at") or pending.get("observed_at"))
    if reference is None:
        reference = _snapshot_timestamp(payload.get("snapshot_id"))
    if reference is None:
        return []
    age = (received_at - reference).total_seconds()
    if age > MAX_PENDING_AGE_SECONDS:
        return ["pending_stale"]
    return []


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _snapshot_timestamp(value: object) -> datetime | None:
    text = str(value or "")
    match = re.search(r"env_(\d{8})_(\d{6})_", text)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _count_by_key(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _with_status_keys(counts: dict[str, int]) -> dict[str, int]:
    normalized = {status: int(counts.get(status, 0)) for status in FEEDBACK_RESULT_STATUSES}
    for status, count in counts.items():
        if status not in normalized:
            normalized[status] = int(count)
    return normalized


def _sanitize_user_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_USER_TEXT_LENGTH]


def _short_text(value: object, *, max_length: int) -> str:
    text = _sanitize_user_text(value)
    return text[:max_length]


def _json_safe(value: object, *, max_depth: int) -> object:
    if max_depth <= 0:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return _short_text(value, max_length=MAX_STRING_LENGTH)
    if isinstance(value, dict):
        safe: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 100:
                break
            safe_key = _short_text(key, max_length=120)
            if safe_key:
                safe[safe_key] = _json_safe(item, max_depth=max_depth - 1)
        return safe
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, max_depth=max_depth - 1) for item in value[:100]]
    return _short_text(value, max_length=MAX_STRING_LENGTH)
