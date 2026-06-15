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
    live_test_candidate: bool = False
    restore_action_id: str | None = None
    stop_action_id: str | None = None
    terminal_action: bool = False
    safety_requirements: tuple[str, ...] = ()
    proof_ceiling: str | None = None


def _command_only_effect(expected_state: str, *, control_type: str) -> dict[str, Any]:
    return {
        "expected_state": expected_state,
        "control_type": control_type,
        "state_authority": "submitted_only",
        "verification_mode": "command_ack_only",
        "evidence_class": "command_ack_only",
        "physical_state_source": "not_supported",
        "unverified_state_label": "submitted_unverified",
    }


def _open_loop_effect(expected_state: str) -> dict[str, Any]:
    return {
        "expected_state": expected_state,
        "control_type": "stateless_toggle",
        "state_authority": "open_loop",
        "verification_mode": "external_observation",
        "evidence_class": "external_observation_required",
        "physical_state_source": "not_supported",
        "unverified_state_label": "open_loop_toggle_submitted",
    }


def _ha_state_effect(
    expected_state: str,
    *,
    control_type: str,
    domain: str,
    service: str,
    entity_id: str,
) -> dict[str, Any]:
    return {
        "expected_state": expected_state,
        "control_type": control_type,
        "state_authority": "ha_entity",
        "verification_mode": "ha_state",
        "evidence_class": "ha_state",
        "physical_state_source": "home_assistant",
        "domain": domain,
        "service": service,
        "entity_id": entity_id,
    }


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
        expected_effect=_ha_state_effect(
            "open",
            control_type="position_command",
            domain="cover",
            service="open_cover",
            entity_id="cover.demo_curtain",
        ),
        live_test_candidate=True,
        restore_action_id="door_close",
        stop_action_id="door_stop",
        safety_requirements=("obstruction_clearance", "original_position_restore"),
        proof_ceiling="ha_visible_cover_position_checkstate_layer",
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
        expected_effect=_ha_state_effect(
            "closed",
            control_type="position_command",
            domain="cover",
            service="close_cover",
            entity_id="cover.demo_curtain",
        ),
        live_test_candidate=True,
        restore_action_id="door_open",
        stop_action_id="door_stop",
        safety_requirements=("obstruction_clearance", "original_position_restore"),
        proof_ceiling="ha_visible_cover_position_checkstate_layer",
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
        expected_effect=_command_only_effect("stopped", control_type="position_command"),
        proof_ceiling="command_ack_only",
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
        expected_effect=_open_loop_effect("on"),
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
        expected_effect=_open_loop_effect("off"),
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
        expected_effect=_command_only_effect("on", control_type="stateless_command"),
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
        expected_effect=_command_only_effect("off", control_type="stateless_command"),
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
        expected_effect=_command_only_effect("on", control_type="stateless_command"),
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
        expected_effect=_command_only_effect("off", control_type="stateless_command"),
    ),
    ActionDefinition(
        action_id="aircon_cool",
        label="エアコンを冷房にする",
        appliance_id="aircon",
        aliases=(
            "エアコンを冷房にして",
            "エアコン冷房にして",
            "冷房にして",
            "冷房をつけて",
        ),
        target_label="エアコン",
        verb="冷房にする",
        pre_action_phrase="エアコンを冷房にする",
        requires_confirmation=True,
        confirmation_reason="climate_control",
        risk_level="medium",
        expected_state="cool",
        expected_effect=_ha_state_effect(
            "cool",
            control_type="mode_command",
            domain="climate",
            service="set_hvac_mode",
            entity_id="climate.demo_aircon",
        ),
    ),
    ActionDefinition(
        action_id="aircon_hvac_off",
        label="エアコンを停止する",
        appliance_id="aircon",
        aliases=(
            "エアコンを停止して",
            "エアコン停止して",
        ),
        target_label="エアコン",
        verb="停止する",
        pre_action_phrase="エアコンを停止する",
        requires_confirmation=True,
        confirmation_reason="climate_control",
        risk_level="medium",
        expected_state="off",
        expected_effect=_ha_state_effect(
            "off",
            control_type="mode_command",
            domain="climate",
            service="set_hvac_mode",
            entity_id="climate.demo_aircon",
        ),
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
        expected_effect=_ha_state_effect(
            "cleaning",
            control_type="job_command",
            domain="vacuum",
            service="start",
            entity_id="vacuum.demo_cloud_target",
        ),
        live_test_candidate=True,
        restore_action_id="vacuum_return",
        safety_requirements=("path_floor_safety",),
        proof_ceiling="ha_visible_vacuum_state_checkstate_layer",
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
        expected_effect=_ha_state_effect(
            "docked",
            control_type="job_command",
            domain="vacuum",
            service="return_to_base",
            entity_id="vacuum.demo_cloud_target",
        ),
        live_test_candidate=True,
        terminal_action=True,
        proof_ceiling="ha_visible_vacuum_return_checkstate_layer",
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
        expected_effect=_ha_state_effect(
            "paused",
            control_type="job_command",
            domain="vacuum",
            service="pause",
            entity_id="vacuum.demo_cloud_target",
        ),
        live_test_candidate=True,
        restore_action_id="vacuum_return",
        safety_requirements=("active_task_context", "return_cleanup"),
        proof_ceiling="ha_visible_vacuum_state_checkstate_layer",
    ),
    ActionDefinition(
        action_id="projection_mode",
        label="プロジェクションモードにする",
        appliance_id="projection",
        aliases=(
            "プロジェクションモードにして",
            "プロジェクションにして",
            "投影モードにして",
        ),
        target_label="プロジェクション",
        verb="切り替える",
        pre_action_phrase="プロジェクションモードにする",
        requires_confirmation=True,
        confirmation_reason="display_mode",
        risk_level="medium",
        expected_state="projection_mode",
        expected_effect=_command_only_effect("projection_mode", control_type="stateless_command"),
        proof_ceiling="not_home_control_appliance_coverage_row",
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
    live_test_blockers = _live_test_blockers(action)
    live_test_readiness = (
        "test_now"
        if action.live_test_candidate and not live_test_blockers
        else "do_not_test_current_config"
        if action.live_test_candidate
        else "not_live_test_candidate"
    )
    proof_ceiling = _action_proof_ceiling(action)
    verification_mode = _verification_mode(action.expected_effect)
    state_tracking = _state_tracking(action.expected_effect)
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
        "control_type": action.expected_effect.get("control_type"),
        "state_authority": action.expected_effect.get("state_authority"),
        "verification_mode": verification_mode,
        "state_tracking": state_tracking,
        "proof_ceiling": proof_ceiling,
        "live_test_candidate": action.live_test_candidate,
        "live_test_readiness": live_test_readiness,
        "live_test_blockers": live_test_blockers,
        "restore_action_id": action.restore_action_id,
        "stop_action_id": action.stop_action_id,
        "terminal_action": action.terminal_action,
        "safety_requirements": list(action.safety_requirements),
        "recheck_visibility": _recheck_visibility(action),
        "available": available,
        "noop": noop,
        "reason": reason,
        "reason_text": _reason_text(action, reason),
    }


def _recheck_visibility(action: ActionDefinition) -> dict[str, Any]:
    effect = action.expected_effect
    proof_ceiling = _action_proof_ceiling(action)
    live_test_blockers = _live_test_blockers(action)
    live_test_readiness = (
        "test_now"
        if action.live_test_candidate and not live_test_blockers
        else "do_not_test_current_config"
        if action.live_test_candidate
        else "not_live_test_candidate"
    )
    live_fields = {
        "proof_ceiling": proof_ceiling,
        "live_test_candidate": action.live_test_candidate,
        "live_test_readiness": live_test_readiness,
        "live_test_blockers": live_test_blockers,
        "restore_action_id": action.restore_action_id,
        "stop_action_id": action.stop_action_id,
        "terminal_action": action.terminal_action,
        "safety_requirements": list(action.safety_requirements),
    }
    physical_state_source = str(effect.get("physical_state_source") or "").strip()
    evidence_class = str(effect.get("evidence_class") or "").strip()
    entity_id = str(effect.get("entity_id") or "").strip()
    domain = str(effect.get("domain") or "").strip()

    if physical_state_source == "not_supported":
        return {
            **live_fields,
            "status": "known_gap",
            "structured_signal": "action_registry.expected_effect",
            "evidence_class": evidence_class or "action_event_only",
            "physical_state_source": "not_supported",
            "unverified_state_label": effect.get("unverified_state_label") or "submitted_unverified",
            "review_note": "physical_state_not_observable_from_environment_state",
        }

    if entity_id:
        return {
            **live_fields,
            "status": "observable",
            "structured_signal": "home_assistant.entity_state",
            "evidence_class": evidence_class or "physical_state_supported",
            "physical_state_source": physical_state_source or "home_assistant",
            "entity_id": entity_id,
            "domain": domain,
        }

    return {
        **live_fields,
        "status": "unmapped",
        "structured_signal": "action_registry.expected_state",
        "evidence_class": evidence_class or "expected_state_only",
        "physical_state_source": physical_state_source or "not_declared",
    }


def _verification_mode(effect: dict[str, Any]) -> str:
    value = str(effect.get("verification_mode") or "").strip()
    if value:
        return value
    return "ha_state" if effect.get("entity_id") else "command_ack_only"


def _state_tracking(effect: dict[str, Any]) -> str:
    verification_mode = _verification_mode(effect)
    state_authority = str(effect.get("state_authority") or "").strip()
    if verification_mode == "ha_state" and state_authority in {"ha_entity", "ha_inferred"}:
        return "tracked"
    if verification_mode == "external_observation" or state_authority == "open_loop":
        return "external_required"
    if verification_mode == "manual_confirmation" or state_authority == "manual":
        return "manual_required"
    if verification_mode == "unsupported":
        return "unsupported"
    return "ack_only"


def _action_proof_ceiling(action: ActionDefinition) -> str:
    if action.proof_ceiling:
        return action.proof_ceiling
    effect = action.expected_effect
    if _state_tracking(effect) == "tracked":
        return "ha_visible_state_checkstate_layer"
    if _verification_mode(effect) == "external_observation":
        return "external_observation_required"
    return "command_ack_only"


def _live_test_blockers(action: ActionDefinition) -> list[str]:
    blockers: list[str] = []
    if not action.live_test_candidate:
        return ["not_marked_live_test_candidate"]
    if _state_tracking(action.expected_effect) != "tracked":
        blockers.append("missing_ha_visible_success_criterion")
    if not action.terminal_action and not action.restore_action_id and not action.stop_action_id:
        blockers.append("missing_restore_or_stop")
    for requirement in action.safety_requirements:
        blockers.append(f"safety_requirement:{requirement}")
    return blockers


def action_readiness_summary(actions: list[dict[str, Any]]) -> dict[str, Any]:
    by_readiness: dict[str, int] = {}
    proof_ceilings: dict[str, int] = {}
    candidate_ids: list[str] = []
    blocked_candidate_ids: list[str] = []
    for action in actions:
        readiness = str(action.get("live_test_readiness") or "unknown")
        proof_ceiling = str(action.get("proof_ceiling") or "unknown")
        by_readiness[readiness] = by_readiness.get(readiness, 0) + 1
        proof_ceilings[proof_ceiling] = proof_ceilings.get(proof_ceiling, 0) + 1
        action_id = str(action.get("action_id") or "")
        if action.get("live_test_candidate") is True and action_id:
            candidate_ids.append(action_id)
            if readiness != "test_now":
                blocked_candidate_ids.append(action_id)
    return {
        "schema_version": "home_control_action_readiness.v0",
        "by_readiness": by_readiness,
        "proof_ceilings": proof_ceilings,
        "live_test_candidate_ids": candidate_ids,
        "blocked_live_test_candidate_ids": blocked_candidate_ids,
        "test_now_count": by_readiness.get("test_now", 0),
        "blocked_candidate_count": len(blocked_candidate_ids),
    }


def public_action_summaries(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_fields = (
        "action_id",
        "label",
        "appliance_id",
        "target_label",
        "verb",
        "pre_action_phrase",
        "requires_confirmation",
        "confirmation_reason",
        "risk_level",
        "current_state",
        "expected_state",
        "control_type",
        "state_authority",
        "verification_mode",
        "state_tracking",
        "proof_ceiling",
        "live_test_candidate",
        "live_test_readiness",
        "live_test_blockers",
        "restore_action_id",
        "stop_action_id",
        "terminal_action",
        "safety_requirements",
        "available",
        "noop",
        "reason",
        "reason_text",
    )
    summaries: list[dict[str, Any]] = []
    for action in actions:
        summary = {field: action[field] for field in allowed_fields if field in action}
        recheck = action.get("recheck_visibility")
        if isinstance(recheck, dict):
            summary["recheck_visibility"] = {
                field: recheck[field]
                for field in (
                    "status",
                    "structured_signal",
                    "evidence_class",
                    "physical_state_source",
                    "unverified_state_label",
                    "review_note",
                    "proof_ceiling",
                    "live_test_candidate",
                    "live_test_readiness",
                    "live_test_blockers",
                    "restore_action_id",
                    "stop_action_id",
                    "terminal_action",
                    "safety_requirements",
                )
                if field in recheck
            }
        summaries.append(summary)
    return summaries


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
