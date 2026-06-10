from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionDefinition:
    action_id: str
    label: str
    appliance_id: str
    aliases: tuple[str, ...]
    target_label: str
    verb: str
    pre_action_phrase: str
    requires_confirmation: bool
    confirmation_reason: str | None
    risk_level: str
    expected_state: str
    expected_effect: dict[str, Any]


ACTION_DEFINITIONS: tuple[ActionDefinition, ...] = (
    ActionDefinition(
        action_id="door_open",
        label="中扉を開ける",
        appliance_id="door",
        aliases=(
            "中扉を開けて",
            "中扉開けて",
            "中扉を開いて",
            "ドアを開けて",
            "ドア開けて",
            "ドアを開いて",
            "扉を開けて",
            "扉開けて",
        ),
        target_label="中扉",
        verb="開ける",
        pre_action_phrase="中扉を開ける",
        requires_confirmation=True,
        confirmation_reason="door_motion",
        risk_level="medium",
        expected_state="open",
        expected_effect={"expected_state": "open"},
    ),
    ActionDefinition(
        action_id="door_close",
        label="中扉を閉める",
        appliance_id="door",
        aliases=(
            "中扉を閉めて",
            "中扉閉めて",
            "中扉を閉じて",
            "中扉閉じて",
            "ドアを閉めて",
            "ドア閉めて",
            "ドアを閉じて",
            "ドア閉じて",
            "扉を閉めて",
            "扉を閉じて",
        ),
        target_label="中扉",
        verb="閉める",
        pre_action_phrase="中扉を閉める",
        requires_confirmation=True,
        confirmation_reason="door_motion",
        risk_level="medium",
        expected_state="closed",
        expected_effect={"expected_state": "closed"},
    ),
    ActionDefinition(
        action_id="door_stop",
        label="中扉を止める",
        appliance_id="door",
        aliases=(
            "中扉を止めて",
            "中扉止めて",
            "ドアを止めて",
            "ドア止めて",
            "扉を止めて",
            "扉止めて",
        ),
        target_label="中扉",
        verb="止める",
        pre_action_phrase="中扉を止める",
        requires_confirmation=True,
        confirmation_reason="door_motion",
        risk_level="medium",
        expected_state="stopped",
        expected_effect={"expected_state": "stopped"},
    ),
    ActionDefinition(
        action_id="light_on",
        label="ライトをつける",
        appliance_id="light",
        aliases=(
            "ライトをつけて",
            "ライトつけて",
            "電気をつけて",
            "電気つけて",
            "照明をつけて",
            "照明つけて",
        ),
        target_label="電気",
        verb="つける",
        pre_action_phrase="電気をつける",
        requires_confirmation=False,
        confirmation_reason=None,
        risk_level="low",
        expected_state="on",
        expected_effect={
            "expected_state": "on",
            "control_type": "stateless_toggle",
            "state_authority": "open_loop",
            "verification_mode": "external_observation",
            "evidence_class": "external_observation_required",
            "physical_state_source": "not_supported",
            "unverified_state_label": "open_loop_toggle_submitted",
        },
    ),
    ActionDefinition(
        action_id="light_off",
        label="ライトを消す",
        appliance_id="light",
        aliases=(
            "ライトを消して",
            "ライト消して",
            "電気を消して",
            "電気消して",
            "照明を消して",
            "照明消して",
        ),
        target_label="電気",
        verb="消す",
        pre_action_phrase="電気を消す",
        requires_confirmation=False,
        confirmation_reason=None,
        risk_level="low",
        expected_state="off",
        expected_effect={
            "expected_state": "off",
            "control_type": "stateless_toggle",
            "state_authority": "open_loop",
            "verification_mode": "external_observation",
            "evidence_class": "external_observation_required",
            "physical_state_source": "not_supported",
            "unverified_state_label": "open_loop_toggle_submitted",
        },
    ),
    ActionDefinition(
        action_id="fan_on",
        label="扇風機をつける",
        appliance_id="fan",
        aliases=(
            "扇風機をつけて",
            "扇風機つけて",
            "ファンをつけて",
            "ファンつけて",
        ),
        target_label="扇風機",
        verb="つける",
        pre_action_phrase="扇風機をつける",
        requires_confirmation=False,
        confirmation_reason=None,
        risk_level="low",
        expected_state="on",
        expected_effect={"expected_state": "on"},
    ),
    ActionDefinition(
        action_id="fan_off",
        label="扇風機を消す",
        appliance_id="fan",
        aliases=(
            "扇風機を消して",
            "扇風機消して",
            "ファンを消して",
            "ファン消して",
            "扇風機を止めて",
        ),
        target_label="扇風機",
        verb="消す",
        pre_action_phrase="扇風機を消す",
        requires_confirmation=False,
        confirmation_reason=None,
        risk_level="low",
        expected_state="off",
        expected_effect={"expected_state": "off"},
    ),
    ActionDefinition(
        action_id="aircon_on",
        label="エアコンをつける",
        appliance_id="aircon",
        aliases=(
            "エアコンをつけて",
            "エアコンつけて",
            "エアコンを入れて",
            "エアコン入れて",
            "冷房をつけて",
            "暖房をつけて",
            "空調をつけて",
        ),
        target_label="エアコン",
        verb="つける",
        pre_action_phrase="エアコンをつける",
        requires_confirmation=True,
        confirmation_reason="climate_control",
        risk_level="medium",
        expected_state="on",
        expected_effect={
            "expected_state": "on",
            "evidence_class": "action_event_only",
            "physical_state_source": "not_supported",
            "unverified_state_label": "submitted_unverified",
        },
    ),
    ActionDefinition(
        action_id="aircon_off",
        label="エアコンを消す",
        appliance_id="aircon",
        aliases=(
            "エアコンを消して",
            "エアコン消して",
            "エアコンを切って",
            "エアコン切って",
            "エアコンを止めて",
            "エアコン止めて",
            "エアコンを停止して",
            "エアコン停止して",
            "冷房を消して",
            "冷房を止めて",
            "暖房を消して",
            "暖房を止めて",
            "空調を消して",
            "空調を止めて",
        ),
        target_label="エアコン",
        verb="消す",
        pre_action_phrase="エアコンを消す",
        requires_confirmation=True,
        confirmation_reason="climate_control",
        risk_level="medium",
        expected_state="off",
        expected_effect={
            "expected_state": "off",
            "evidence_class": "action_event_only",
            "physical_state_source": "not_supported",
            "unverified_state_label": "submitted_unverified",
        },
    ),
    ActionDefinition(
        action_id="vacuum_start",
        label="掃除機を開始する",
        appliance_id="vacuum",
        aliases=(
            "掃除機をかけて",
            "掃除機かけて",
            "掃除機を動かして",
            "掃除を始めて",
            "ロボット掃除機を動かして",
            "掃除機スタート",
        ),
        target_label="掃除機",
        verb="動かす",
        pre_action_phrase="掃除機を動かす",
        requires_confirmation=True,
        confirmation_reason="vacuum_motion",
        risk_level="medium",
        expected_state="cleaning",
        expected_effect={"expected_state": "cleaning"},
    ),
    ActionDefinition(
        action_id="vacuum_return",
        label="掃除機を戻す",
        appliance_id="vacuum",
        aliases=(
            "掃除機を戻して",
            "掃除機戻して",
            "掃除機を充電器に戻して",
            "ロボット掃除機を戻して",
            "掃除機を帰らせて",
            "掃除機帰って",
        ),
        target_label="掃除機",
        verb="戻す",
        pre_action_phrase="掃除機を戻す",
        requires_confirmation=True,
        confirmation_reason="vacuum_motion",
        risk_level="medium",
        expected_state="returning",
        expected_effect={"expected_state": "returning"},
    ),
    ActionDefinition(
        action_id="vacuum_pause",
        label="掃除機を一時停止する",
        appliance_id="vacuum",
        aliases=(
            "掃除機を止めて",
            "掃除機止めて",
            "掃除機を一時停止して",
            "掃除機一時停止",
            "ロボット掃除機を止めて",
            "掃除を止めて",
        ),
        target_label="掃除機",
        verb="一時停止する",
        pre_action_phrase="掃除機を一時停止する",
        requires_confirmation=True,
        confirmation_reason="vacuum_motion",
        risk_level="medium",
        expected_state="paused",
        expected_effect={"expected_state": "paused"},
    ),
)


def build_action_registry(appliances: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [_action_payload(action, appliances.get(action.appliance_id)) for action in ACTION_DEFINITIONS]


def _action_payload(
    action: ActionDefinition,
    appliance: dict[str, Any] | None,
) -> dict[str, Any]:
    current_state = _current_state(appliance)
    available, noop, reason = _availability(action, current_state)
    return {
        "action_id": action.action_id,
        "label": action.label,
        "appliance_id": action.appliance_id,
        "aliases": list(action.aliases),
        "target_label": action.target_label,
        "verb": action.verb,
        "pre_action_phrase": action.pre_action_phrase,
        "requires_confirmation": action.requires_confirmation,
        "confirmation_reason": action.confirmation_reason,
        "risk_level": action.risk_level,
        "confirmation_policy": {
            "requires_confirmation": action.requires_confirmation,
            "reason": action.confirmation_reason,
            "risk_level": action.risk_level,
        },
        "current_state": current_state,
        "expected_state": action.expected_state,
        "expected_effect": dict(action.expected_effect),
        "recheck_visibility": _recheck_visibility(action),
        "available": available,
        "noop": noop,
        "reason": reason,
        "reason_text": _reason_text(action, reason),
    }


def _recheck_visibility(action: ActionDefinition) -> dict[str, Any]:
    effect = action.expected_effect
    physical_state_source = str(effect.get("physical_state_source") or "").strip()
    evidence_class = str(effect.get("evidence_class") or "").strip()
    entity_id = str(effect.get("entity_id") or "").strip()
    domain = str(effect.get("domain") or "").strip()

    if physical_state_source == "not_supported":
        return {
            "status": "known_gap",
            "structured_signal": "action_registry.expected_effect",
            "evidence_class": evidence_class or "action_event_only",
            "physical_state_source": "not_supported",
            "unverified_state_label": effect.get("unverified_state_label") or "submitted_unverified",
            "review_note": "physical_state_not_observable_from_environment_state",
        }

    if entity_id:
        return {
            "status": "observable",
            "structured_signal": "home_assistant.entity_state",
            "evidence_class": evidence_class or "physical_state_supported",
            "physical_state_source": physical_state_source or "home_assistant",
            "entity_id": entity_id,
            "domain": domain,
        }

    return {
        "status": "unmapped",
        "structured_signal": "action_registry.expected_state",
        "evidence_class": evidence_class or "expected_state_only",
        "physical_state_source": physical_state_source or "not_declared",
    }


def _current_state(appliance: dict[str, Any] | None) -> dict[str, Any] | None:
    if not appliance:
        return None
    return {
        "state": appliance.get("state"),
        "updated_at": appliance.get("updated_at"),
        "stale": bool(appliance.get("stale", True)),
        "source": appliance.get("source"),
    }


def _availability(
    action: ActionDefinition,
    current_state: dict[str, Any] | None,
) -> tuple[bool, bool, str]:
    if current_state is None or not current_state.get("state"):
        return True, False, "current_state_unavailable"
    if current_state.get("stale"):
        return True, False, "current_state_stale"

    state = str(current_state.get("state"))
    if state == action.expected_state:
        return False, True, f"already_{state}"
    return True, False, "ready"


def _reason_text(action: ActionDefinition, reason: str) -> str:
    if reason == "ready":
        return f"{action.pre_action_phrase}操作を実行できます"
    if reason == "current_state_unavailable":
        return f"{action.target_label}の現在状態が取得できていません"
    if reason == "current_state_stale":
        return f"{action.target_label}の現在状態が古いため、必要なら{action.verb}操作を実行できます"
    if reason.startswith("already_"):
        state = reason.removeprefix("already_")
        return f"{action.target_label}はすでに{_state_text(state)}"
    return reason


def _state_text(state: str) -> str:
    return {
        "open": "開いています",
        "closed": "閉まっています",
        "stopped": "止まっています",
        "on": "ついています",
        "off": "消えています",
        "cleaning": "掃除中です",
        "returning": "帰還中です",
        "paused": "一時停止中です",
    }.get(state, state)
