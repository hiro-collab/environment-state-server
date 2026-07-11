from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

from environment_state_server.state import EnvironmentStateStore


ROOM_LIGHT_DOES_NOT_PROVE = [
    "physical_room_light_state",
    "home_assistant_light_state",
]

SHARED_VECTOR_ENV = "SWORD_T1_ROOM_LIGHT_SHARED_VECTOR_PATH"
SHARED_VECTOR_PATH = os.environ.get(SHARED_VECTOR_ENV)
SHARED_VECTOR_MAX_BYTES = 256 * 1024
SHARED_VECTOR_SHA256 = "e1f43a6dd5047ea48818079b0510b9f487c93b0f493c77bba57fc6abc89e41bd"
ROOM_LIGHT_FIXTURE_UNAVAILABLE = "room_light_fixture_unavailable"
ROOM_LIGHT_FIXTURE_INVALID = "room_light_fixture_invalid"
SHARED_VECTOR_CASE_IDS = [
    "canonical_camera_hub",
    "canonical_vision_snapshot_processor",
    "malformed_nested_sequence",
    "wrong_numeric_type",
    "nonfinite_numeric",
    "out_of_range_numeric",
    "wrong_case",
    "stale_freshness",
    "reversed_ordered_nonclaims",
    "non_room_light",
    "unknown_field_non_echo",
    "wrong_proof_ceiling",
    "responsiveness_same_identity_material_movement",
    "responsiveness_changed_identity_no_material_movement",
    "responsiveness_changed_identity_material_movement",
]
SHARED_VECTOR_EXPECTED = {
    "canonical_camera_hub": ("valid", "camera-environment-estimate-high-confidence", "pass", "material_camera_environment_estimate_change_with_new_observation"),
    "canonical_vision_snapshot_processor": ("valid", "camera-environment-estimate-high-confidence", "pass", "material_camera_environment_estimate_change_with_new_observation"),
    "malformed_nested_sequence": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate"),
    "wrong_numeric_type": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate"),
    "nonfinite_numeric": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate"),
    "out_of_range_numeric": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate"),
    "wrong_case": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate"),
    "stale_freshness": ("valid", "unavailable", "partial", "material_camera_environment_estimate_change_with_new_observation"),
    "reversed_ordered_nonclaims": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate"),
    "non_room_light": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate"),
    "unknown_field_non_echo": ("valid", "camera-environment-estimate-high-confidence", "pass", "material_camera_environment_estimate_change_with_new_observation"),
    "wrong_proof_ceiling": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate"),
    "responsiveness_same_identity_material_movement": ("valid", "camera-environment-estimate-high-confidence", "fail", "material_camera_environment_estimate_change_without_new_observation"),
    "responsiveness_changed_identity_no_material_movement": ("valid", "camera-environment-estimate-high-confidence", "fail", "new_observation_without_material_camera_environment_estimate_change"),
    "responsiveness_changed_identity_material_movement": ("valid", "camera-environment-estimate-high-confidence", "pass", "material_camera_environment_estimate_change_with_new_observation"),
}
SHARED_VECTOR_PAYLOAD_KEYS = {
    "type", "schema_version", "observation_bucket", "confidence",
    "daylight_ambiguity", "cue_likelihoods", "source", "source_class",
    "observed_at", "observation_id", "source_snapshot_id", "sequence",
    "model", "freshness", "proof_ceiling", "does_not_prove",
}


def _require_room_light_fixture(condition: bool) -> None:
    if not condition:
        raise AssertionError(ROOM_LIGHT_FIXTURE_INVALID) from None


def _assert_safe_shared_value(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_room_light_fixture(isinstance(key, str) and key.isprintable() and len(key) <= 80)
            _assert_safe_shared_value(nested)
    elif isinstance(value, list):
        _require_room_light_fixture(len(value) <= 20)
        for nested in value:
            _assert_safe_shared_value(nested)
    elif isinstance(value, str):
        _require_room_light_fixture(value.isprintable() and len(value) <= 200)
        _require_room_light_fixture("\\" not in value and "://" not in value)
    else:
        _require_room_light_fixture(value is None or isinstance(value, (bool, int, float)))
        if isinstance(value, float):
            _require_room_light_fixture(math.isfinite(value))


def _validate_shared_vector_fixture(data: object) -> dict[str, object]:
    try:
        _require_room_light_fixture(isinstance(data, dict))
        _require_room_light_fixture(set(data) == {"fixture_version", "fixture_kind", "unknown_field_sentinel", "cases"})
        _require_room_light_fixture(data["fixture_version"] == "room-light-shared-vectors.v1")
        _require_room_light_fixture(data["fixture_kind"] == "non_schema_test_vectors")
        _require_room_light_fixture(data["unknown_field_sentinel"] == "fixed-unknown-room-light-sentinel-7e57")
        cases = data["cases"]
        _require_room_light_fixture(isinstance(cases, list) and len(cases) == 15)
        _require_room_light_fixture([row.get("case_id") for row in cases if isinstance(row, dict)] == SHARED_VECTOR_CASE_IDS)
        for row in cases:
            _require_room_light_fixture(isinstance(row, dict))
            case_id = row["case_id"]
            expected_row_keys = {"case_id", "baseline", "followup", "expected"}
            if case_id == "nonfinite_numeric":
                expected_row_keys.add("synthetic_numeric_class")
                _require_room_light_fixture(row["synthetic_numeric_class"] == "followup_confidence_nan")
            _require_room_light_fixture(set(row) == expected_row_keys)
            expected = row["expected"]
            _require_room_light_fixture(isinstance(expected, dict))
            _require_room_light_fixture(set(expected) == {"validation_class", "claim_class", "responsiveness_class", "delta_class", "unknown_echo_class"})
            _require_room_light_fixture(tuple(expected[key] for key in ("validation_class", "claim_class", "responsiveness_class", "delta_class")) == SHARED_VECTOR_EXPECTED[case_id])
            _require_room_light_fixture(expected["unknown_echo_class"] == "not_echoed")
            for phase in ("baseline", "followup"):
                payload = row[phase]
                _require_room_light_fixture(isinstance(payload, dict))
                allowed = set(SHARED_VECTOR_PAYLOAD_KEYS)
                if phase == "followup" and case_id == "unknown_field_non_echo":
                    allowed.add("unknown_test_field")
                    _require_room_light_fixture(payload["unknown_test_field"] == data["unknown_field_sentinel"])
                _require_room_light_fixture(set(payload) == allowed)
                _require_room_light_fixture(set(payload["cue_likelihoods"]) == {"warm_light", "daylight", "darkness"})
                _require_room_light_fixture(set(payload["sequence"]) == {"first_frame_id", "last_frame_id", "frame_count", "temporal_window_ms"})
                _require_room_light_fixture(set(payload["model"]) == {"name", "kind"})
                _require_room_light_fixture(set(payload["freshness"]) == {"level"})
                _require_room_light_fixture(payload["source"] in {"camera_hub", "vision_snapshot_processor"})
                expected_nonclaims = list(ROOM_LIGHT_DOES_NOT_PROVE)
                if case_id == "reversed_ordered_nonclaims" and phase == "followup":
                    expected_nonclaims.reverse()
                _require_room_light_fixture(payload["does_not_prove"] == expected_nonclaims)
                _require_room_light_fixture(str(payload["observation_id"]).startswith("synthetic-"))
                _require_room_light_fixture("observation-" in str(payload["observation_id"]))
                _require_room_light_fixture(str(payload["source_snapshot_id"]).startswith("synthetic-"))
                _require_room_light_fixture("snapshot-" in str(payload["source_snapshot_id"]))
        _assert_safe_shared_value(data)
        return data
    except AssertionError:
        raise
    except (AttributeError, KeyError, OverflowError, RecursionError, TypeError, ValueError):
        raise AssertionError(ROOM_LIGHT_FIXTURE_INVALID) from None


def _load_shared_vector_fixture() -> dict[str, object]:
    if not SHARED_VECTOR_PATH:
        raise AssertionError(ROOM_LIGHT_FIXTURE_UNAVAILABLE) from None
    try:
        path = Path(SHARED_VECTOR_PATH).expanduser()
        _require_room_light_fixture(len(str(path)) <= 4096)
    except AssertionError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise AssertionError(ROOM_LIGHT_FIXTURE_INVALID) from None
    try:
        if not path.is_file():
            raise AssertionError(ROOM_LIGHT_FIXTURE_UNAVAILABLE) from None
        size = path.stat().st_size
        _require_room_light_fixture(0 < size <= SHARED_VECTOR_MAX_BYTES)
        raw = path.read_bytes()
    except AssertionError:
        raise
    except OSError:
        raise AssertionError(ROOM_LIGHT_FIXTURE_UNAVAILABLE) from None
    _require_room_light_fixture(len(raw) == size)
    _require_room_light_fixture(hashlib.sha256(raw).hexdigest() == SHARED_VECTOR_SHA256)
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise AssertionError(ROOM_LIGHT_FIXTURE_INVALID) from None
    return _validate_shared_vector_fixture(data)


def _shared_vector_envelope(vector: dict[str, object]) -> tuple[dict[str, object], str]:
    payload = deepcopy(vector)
    source = str(payload.pop("source"))
    payload.pop("freshness")
    payload.pop("source_snapshot_id")
    payload["source"] = "vision_snapshot_processor"
    observed = datetime.fromisoformat(str(payload["observed_at"]).replace("Z", "+00:00"))
    sequence = payload["sequence"]
    _require_room_light_fixture(isinstance(sequence, dict))
    header_seq = sequence.get("last_frame_id")
    if type(header_seq) is not int:
        header_seq = 0
    return {
        "schema_version": 1,
        "topic": "/vision/room_light/observation",
        "msg_type": "vision_snapshot_processor/RoomLightObservation",
        "header": {"seq": header_seq, "stamp": observed.timestamp(), "frame_id": "synthetic-camera"},
        "payload": payload,
    }, source


def _ingest_shared_vector(store: EnvironmentStateStore, vector: dict[str, object], *, nonfinite: bool = False) -> None:
    envelope, source = _shared_vector_envelope(vector)
    payload = envelope["payload"]
    _require_room_light_fixture(isinstance(payload, dict))
    if nonfinite:
        payload["confidence"] = float("nan")
    if source == "camera_hub":
        store.ingest_camera_hub_envelope(envelope)
    else:
        store.ingest_vision_envelope(envelope, source=source)


def _room_light_envelope(
    observed: datetime,
    *,
    seq: int = 42,
    frame_id: str = "camera",
    bucket: str = "balanced",
    confidence: float = 0.6,
    ambiguity: str = "medium",
    cues: dict[str, float] | None = None,
    observation_id: str = "obs-1",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "topic": "/vision/room_light/observation",
        "msg_type": "vision_snapshot_processor/RoomLightObservation",
        "header": {"seq": seq, "stamp": observed.timestamp(), "frame_id": frame_id},
        "payload": {
            "type": "room_light_observation",
            "schema_version": 1,
            "observation_bucket": bucket,
            "confidence": confidence,
            "daylight_ambiguity": ambiguity,
            "cue_likelihoods": cues
            or {"warm_light": 0.2, "daylight": 0.7, "darkness": 0.1},
            "source": "vision_snapshot_processor",
            "source_class": "camera_environment_estimate",
            "observed_at": observed.isoformat(),
            "observation_id": observation_id,
            "sequence": {
                "frame_count": 2,
                "first_frame_id": seq - 1,
                "last_frame_id": seq,
                "temporal_window_ms": 1000,
            },
            "model": {
                "name": "room-light-heuristic-snapshot-v3",
                "kind": "heuristic",
            },
            "proof_ceiling": "camera_environment_estimate_only",
            "does_not_prove": list(ROOM_LIGHT_DOES_NOT_PROVE),
        },
    }


class EnvironmentStateStoreTest(unittest.TestCase):
    def test_home_assistant_light_command_without_read_authority_stays_event_only(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_assistant_event(
            {
                "event": "execute_succeeded",
                "action_id": "light_off",
                "execution_id": "exec-1",
                "request_id": "ha-1",
                "executed": True,
                "status": "submitted",
                "timestamp": "2026-05-06T14:00:00+00:00",
                "expected_effect": {
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.demo_light",
                    "expected_state": "off",
                },
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        self.assertFalse(current["stale"])
        self.assertRegex(current["snapshot_id"], r"^env_20260506_140001_\d{6}$")
        self.assertEqual(current["sequence"], 1)
        self.assertEqual(current["age_ms"], 1000)
        self.assertEqual(current["freshness"]["level"], "fresh")
        self.assertEqual(current["freshness"]["age_ms"], 1000)
        self.assertEqual(current["freshness"]["ttl_ms"], 5000)
        self.assertNotIn("light", current["appliances"])
        self.assertEqual(current["last_home_assistant_events"][0]["status"], "succeeded")
        self.assertIn("capabilities", current)
        self.assertNotIn("actions", current["capabilities"])
        self.assertNotIn("actions", current)
        self.assertNotIn("action_readiness", current)

    def test_explicit_ha_state_light_event_updates_appliance_state(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_assistant_event(
            {
                "event": "execute_succeeded",
                "action_id": "light_off",
                "execution_id": "exec-1",
                "request_id": "ha-1",
                "executed": True,
                "status": "submitted",
                "timestamp": "2026-05-06T14:00:00+00:00",
                "expected_effect": {
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.demo_light",
                    "expected_state": "off",
                    "state_authority": "ha_entity",
                    "verification_mode": "ha_state",
                },
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        self.assertEqual(current["appliances"]["light"]["state"], "off")
        self.assertEqual(current["appliances"]["light"]["freshness"]["level"], "fresh")
        self.assertEqual(current["appliances"]["light"]["execution_id"], "exec-1")

    def test_action_without_expected_effect_does_not_create_appliance_state(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_assistant_event(
            {
                "event": "execute_succeeded",
                "action_id": "door_close",
                "execution_id": "exec-door",
                "executed": True,
                "timestamp": "2026-05-06T14:00:00+00:00",
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        self.assertNotIn("door", current["appliances"])

    def test_external_observation_light_event_does_not_create_appliance_state(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_assistant_event(
            {
                "event": "execute_succeeded",
                "action_id": "light_on",
                "execution_id": "exec-light-open-loop",
                "executed": True,
                "timestamp": "2026-05-06T14:00:00+00:00",
                "expected_effect": {
                    "expected_state": "on",
                    "control_type": "stateless_toggle",
                    "state_authority": "open_loop",
                    "verification_mode": "external_observation",
                    "physical_state_source": "not_supported",
                },
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        self.assertNotIn("light", current["appliances"])

    def test_command_ack_only_event_does_not_create_appliance_state(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_assistant_event(
            {
                "event": "execute_succeeded",
                "action_id": "fan_on",
                "execution_id": "exec-fan-command-only",
                "executed": True,
                "timestamp": "2026-05-06T14:00:00+00:00",
                "expected_effect": {
                    "expected_state": "on",
                    "control_type": "stateless_command",
                    "state_authority": "submitted_only",
                    "verification_mode": "command_ack_only",
                    "evidence_class": "command_ack_only",
                    "physical_state_source": "not_supported",
                },
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        self.assertNotIn("fan", current["appliances"])

    def test_ha_state_expected_effect_uses_expected_state_not_action_suffix(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_assistant_event(
            {
                "event": "execute_succeeded",
                "action_id": "vacuum_return",
                "execution_id": "exec-vacuum-return",
                "executed": True,
                "timestamp": "2026-05-06T14:00:00+00:00",
                "expected_effect": {
                    "domain": "vacuum",
                    "entity_id": "vacuum.demo",
                    "expected_state": "docked",
                    "state_authority": "ha_entity",
                    "verification_mode": "ha_state",
                },
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        self.assertEqual(current["appliances"]["vacuum"]["state"], "docked")

    def test_aircon_current_status_query_registers_readonly_checkstate(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_control_action_state(
            {
                "action_id": "aircon_hvac_off",
                "status": "matched",
                "control_type": "mode_command",
                "state_authority": "ha_entity",
                "verification_mode": "ha_state",
                "state_tracking": "tracked",
                "expected_state": "off",
                "expected_states": ["off"],
                "actual_state": "off",
                "observed_at": "2026-06-15T10:00:00+00:00",
            }
        )

        current = store.current(now=datetime(2026, 6, 15, 10, 0, 1, tzinfo=UTC))
        query = current["state_queries"]["aircon_current_status"]

        self.assertTrue(query["available"])
        self.assertFalse(query["stale"])
        self.assertEqual(query["state"], "off")
        self.assertEqual(query["current_status"], "off")
        self.assertFalse(query["state_is_last_known"])
        self.assertEqual(query["status_label"], "off")
        self.assertEqual(query["appliance_family"], "aircon")
        self.assertEqual(query["status_availability_class"], "available_fresh")
        self.assertEqual(query["status_match_class"], "off_matched")
        self.assertEqual(query["freshness_class"], "fresh")
        self.assertEqual(
            query["safe_wording_class"],
            "HA_visible_aircon_status_class_available",
        )
        self.assertEqual(query["source_class"], "Home_Control_HA_visible_tracked_climate")
        self.assertEqual(
            query["proof_ceiling"],
            "HA_visible_climate_state_only",
        )
        self.assertEqual(query["freshness"]["level"], "fresh")
        self.assertEqual(query["evidence"]["source"], "home_control_readonly")
        self.assertEqual(query["evidence"]["checkstate_status"], "matched")
        self.assertIn("physical_hvac_comfort", query["does_not_prove"])
        self.assertNotIn("entity_id", json.dumps(query, ensure_ascii=False))

        indicators = store.indicators_current(now=datetime(2026, 6, 15, 10, 0, 1, tzinfo=UTC))
        public_query = indicators["environment"]["state_queries"]["aircon_current_status"]
        self.assertEqual(public_query["state"], "off")
        self.assertNotIn("entity_id", json.dumps(public_query, ensure_ascii=False))

    def test_aircon_current_status_query_marks_readonly_checkstate_stale_as_last_known(self) -> None:
        store = EnvironmentStateStore(ttl_ms=1000)
        store.ingest_home_control_action_state(
            {
                "action_id": "aircon_cool",
                "status": "matched",
                "control_type": "mode_command",
                "state_authority": "ha_entity",
                "verification_mode": "ha_state",
                "state_tracking": "tracked",
                "expected_state": "cool",
                "expected_states": ["cool"],
                "actual_state": "cool",
                "observed_at": "2026-06-15T10:00:00+00:00",
            }
        )

        current = store.current(now=datetime(2026, 6, 15, 10, 0, 2, tzinfo=UTC))
        query = current["state_queries"]["aircon_current_status"]

        self.assertFalse(query["available"])
        self.assertTrue(query["stale"])
        self.assertEqual(query["state"], "cool")
        self.assertTrue(query["state_is_last_known"])
        self.assertEqual(query["status_availability_class"], "available_stale")
        self.assertEqual(query["status_match_class"], "cool_matched")
        self.assertEqual(query["freshness_class"], "stale")
        self.assertEqual(
            query["safe_wording_class"],
            "last_known_aircon_status_only_must_revalidate",
        )
        self.assertEqual(query["stale_reason"], "aircon_current_status_stale")
        self.assertEqual(query["freshness"]["level"], "stale")
        self.assertIn("最後に分かっている範囲", query["answer_hint"])

    def test_aircon_current_status_query_keeps_readable_mismatch_as_current_status(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_control_action_state(
            {
                "action_id": "aircon_cool",
                "status": "mismatch",
                "control_type": "mode_command",
                "state_authority": "ha_entity",
                "verification_mode": "ha_state",
                "state_tracking": "tracked",
                "expected_state": "cool",
                "expected_states": ["cool"],
                "actual_state": "off",
                "observed_at": "2026-06-15T10:00:00+00:00",
            }
        )

        current = store.current(now=datetime(2026, 6, 15, 10, 0, 1, tzinfo=UTC))
        query = current["state_queries"]["aircon_current_status"]

        self.assertTrue(query["available"])
        self.assertFalse(query["stale"])
        self.assertEqual(query["state"], "off")
        self.assertEqual(query["status_availability_class"], "available_fresh")
        self.assertEqual(query["status_match_class"], "mismatch")
        self.assertEqual(query["safe_wording_class"], "HA_visible_aircon_status_class_available")

    def test_aircon_current_status_query_rejects_untracked_status_source(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_control_action_state(
            {
                "action_id": "aircon_off",
                "status": "ack_only",
                "state_authority": "submitted_only",
                "verification_mode": "command_ack_only",
                "state_tracking": "ack_only",
                "actual_state": "off",
                "observed_at": "2026-06-15T10:00:00+00:00",
            }
        )

        current = store.current(now=datetime(2026, 6, 15, 10, 0, 1, tzinfo=UTC))
        query = current["state_queries"]["aircon_current_status"]

        self.assertNotIn("aircon", current["appliances"])
        self.assertFalse(query["available"])
        self.assertTrue(query["stale"])
        self.assertEqual(query["state"], "unknown")
        self.assertEqual(query["status_availability_class"], "unavailable")
        self.assertEqual(query["status_match_class"], "unknown")
        self.assertEqual(query["freshness_class"], "unavailable")
        self.assertEqual(query["stale_reason"], "aircon_current_status_missing")

    def test_ttl_marks_state_stale(self) -> None:
        store = EnvironmentStateStore(ttl_ms=1000)
        store.ingest_home_assistant_event(
            {
                "event": "execute_succeeded",
                "action_id": "fan_on",
                "executed": True,
                "timestamp": "2026-05-06T14:00:00+00:00",
                "expected_effect": {
                    "domain": "fan",
                    "entity_id": "fan.demo",
                    "expected_state": "on",
                    "state_authority": "ha_entity",
                    "verification_mode": "ha_state",
                },
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 2, tzinfo=UTC))

        self.assertTrue(current["stale"])
        self.assertEqual(current["freshness"]["level"], "stale")
        self.assertEqual(current["freshness"]["reason"], "age_exceeds_ttl")
        self.assertTrue(current["appliances"]["fan"]["stale"])
        self.assertEqual(current["appliances"]["fan"]["freshness"]["level"], "stale")

    def test_freshness_levels_match_snapshot_age_windows(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_home_assistant_event(
            {
                "event": "execute_succeeded",
                "action_id": "fan_on",
                "executed": True,
                "timestamp": "2026-05-06T14:00:00+00:00",
                "expected_effect": {
                    "domain": "fan",
                    "entity_id": "fan.demo",
                    "expected_state": "on",
                    "state_authority": "ha_entity",
                    "verification_mode": "ha_state",
                },
            }
        )

        recent = store.current(now=datetime(2026, 5, 6, 14, 0, 3, tzinfo=UTC))
        usable = store.current(now=datetime(2026, 5, 6, 14, 0, 4, 500000, tzinfo=UTC))
        stale = store.current(now=datetime(2026, 5, 6, 14, 0, 6, tzinfo=UTC))

        self.assertEqual(recent["freshness"]["level"], "recent")
        self.assertEqual(recent["appliances"]["fan"]["freshness"]["level"], "recent")
        self.assertEqual(usable["freshness"]["level"], "usable")
        self.assertEqual(usable["appliances"]["fan"]["freshness"]["level"], "usable")
        self.assertEqual(stale["freshness"]["level"], "stale")
        self.assertEqual(stale["appliances"]["fan"]["freshness"]["level"], "stale")

    def test_room_light_observation_updates_canonical_vision_state(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC)
        envelope = _room_light_envelope(
            observed,
            bucket="bright",
            confidence=0.82,
            cues={"warm_light": 0.7, "daylight": 0.6, "darkness": 0.05},
        )
        header = envelope["header"]
        payload = envelope["payload"]
        assert isinstance(header, dict)
        assert isinstance(payload, dict)
        assert isinstance(payload["cue_likelihoods"], dict)
        assert isinstance(payload["sequence"], dict)
        assert isinstance(payload["model"], dict)
        header["private_header"] = {"api_key": "header-secret"}
        payload["cue_likelihoods"]["private_cue"] = "cue-secret"
        payload["sequence"]["private_sequence"] = "sequence-secret"
        payload["model"]["private_model"] = "model-secret"
        payload["source_snapshot_id"] = "caller-private-snapshot"
        payload["producer"] = {"private_producer": "producer-secret"}
        store.ingest_camera_hub_envelope(envelope)

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))
        room_light = current["vision"]["room_light"]

        self.assertEqual(room_light["type"], "room_light_observation")
        self.assertEqual(room_light["observation_bucket"], "bright")
        self.assertEqual(room_light["confidence"], 0.82)
        self.assertEqual(room_light["daylight_ambiguity"], "medium")
        self.assertEqual(
            room_light["cue_likelihoods"],
            {"warm_light": 0.7, "daylight": 0.6, "darkness": 0.05},
        )
        self.assertEqual(room_light["source"], "camera_hub")
        self.assertEqual(room_light["source_class"], "camera_environment_estimate")
        self.assertEqual(room_light["observation_id"], "obs-1")
        self.assertEqual(room_light["source_snapshot_id"], "obs-1")
        self.assertEqual(
            room_light["sequence"],
            {
                "frame_count": 2,
                "first_frame_id": 41,
                "last_frame_id": 42,
                "temporal_window_ms": 1000,
            },
        )
        self.assertEqual(
            room_light["model"],
            {"name": "room-light-heuristic-snapshot-v3", "kind": "heuristic"},
        )
        self.assertEqual(room_light["freshness"]["level"], "fresh")
        self.assertEqual(room_light["freshness"]["age_ms"], 1000)
        self.assertEqual(room_light["freshness"]["ttl_ms"], 5000)
        self.assertEqual(room_light["proof_ceiling"], "camera_environment_estimate_only")
        self.assertEqual(room_light["does_not_prove"], ROOM_LIGHT_DOES_NOT_PROVE)
        self.assertEqual(
            room_light["provenance"]["topic"],
            "/vision/room_light/observation",
        )
        self.assertEqual(room_light["provenance"]["producer"], "vision_snapshot_processor")
        serialized = json.dumps(room_light)
        for private_value in (
            "header-secret",
            "cue-secret",
            "sequence-secret",
            "model-secret",
            "caller-private-snapshot",
            "producer-secret",
        ):
            self.assertNotIn(private_value, serialized)
        for removed in ("state", "electric_light", "lighting_type", "probabilities", "label"):
            self.assertNotIn(removed, room_light)
        self.assertNotIn("room_light", current["state_queries"])
        self.assertIn("aircon_current_status", current["state_queries"])

    @unittest.skipUnless(SHARED_VECTOR_PATH, f"set {SHARED_VECTOR_ENV} to run Parent shared-vector sensitivity")
    def test_shared_room_light_fixture_shape_sensitivity(self) -> None:
        fixture = _load_shared_vector_fixture()
        mutations = []

        reordered = deepcopy(fixture)
        reordered["cases"][0], reordered["cases"][1] = reordered["cases"][1], reordered["cases"][0]
        mutations.append(("reordered", reordered))
        removed = deepcopy(fixture)
        removed["cases"].pop()
        mutations.append(("removed", removed))
        extra = deepcopy(fixture)
        extra["unexpected"] = True
        mutations.append(("extra", extra))
        malformed = deepcopy(fixture)
        malformed["cases"][0]["followup"]["sequence"]["unexpected"] = 1
        mutations.append(("malformed_nested_field", malformed))

        for name, candidate in mutations:
            with self.subTest(name=name):
                with self.assertRaisesRegex(AssertionError, f"^{ROOM_LIGHT_FIXTURE_INVALID}$"):
                    _validate_shared_vector_fixture(candidate)

    def test_shared_room_light_loader_failures_are_fixed_and_non_echoing(self) -> None:
        global SHARED_VECTOR_PATH

        original_path = SHARED_VECTOR_PATH
        env_was_present = SHARED_VECTOR_ENV in os.environ
        original_env = os.environ.get(SHARED_VECTOR_ENV)
        injected_case = "injected-case-loader-secret"
        injected_value = "injected-value-loader-secret"
        injected_sentinel = "injected-sentinel-loader-secret"
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                missing = root / "missing-loader-secret.json"
                malformed = root / "malformed-loader-secret.json"
                malformed.write_text(
                    f'{{"case_id":"{injected_case}","value":"{injected_value}","sentinel":"{injected_sentinel}"',
                    encoding="utf-8",
                )
                oversized = root / "oversized-loader-secret.json"
                oversized.write_bytes(b"x" * (SHARED_VECTOR_MAX_BYTES + 1))
                unsafe = root / ("unsafe-loader-secret-" + "x" * 4096)
                cases = (
                    ("missing", str(missing), ROOM_LIGHT_FIXTURE_UNAVAILABLE),
                    ("malformed", str(malformed), ROOM_LIGHT_FIXTURE_INVALID),
                    ("size_violation", str(oversized), ROOM_LIGHT_FIXTURE_INVALID),
                    ("unsafe_path", str(unsafe), ROOM_LIGHT_FIXTURE_INVALID),
                )
                for name, configured, expected in cases:
                    with self.subTest(name=name):
                        os.environ[SHARED_VECTOR_ENV] = configured
                        SHARED_VECTOR_PATH = configured
                        with self.assertRaises(AssertionError) as raised:
                            _load_shared_vector_fixture()
                        message = str(raised.exception)
                        self.assertEqual(message, expected)
                        self.assertIsNone(raised.exception.__cause__)
                        for forbidden in (
                            configured,
                            Path(configured).name,
                            str(SHARED_VECTOR_MAX_BYTES + 1),
                            injected_case,
                            injected_value,
                            injected_sentinel,
                            "No such file",
                            "The system cannot find",
                            "[Errno",
                        ):
                            self.assertNotIn(forbidden, message)
        finally:
            SHARED_VECTOR_PATH = original_path
            if env_was_present:
                assert original_env is not None
                os.environ[SHARED_VECTOR_ENV] = original_env
            else:
                os.environ.pop(SHARED_VECTOR_ENV, None)

    @unittest.skipUnless(SHARED_VECTOR_PATH, f"set {SHARED_VECTOR_ENV} to run Parent shared-vector ESS state consumer")
    def test_shared_room_light_vectors_use_ess_state_contract(self) -> None:
        fixture = _load_shared_vector_fixture()
        cases = fixture["cases"]
        _require_room_light_fixture(isinstance(cases, list))
        sentinel = str(fixture["unknown_field_sentinel"])

        actual_counts = {"valid": 0, "invalid": 0, "available": 0, "unavailable": 0}
        for row in cases:
            _require_room_light_fixture(isinstance(row, dict))
            case_id = str(row["case_id"])
            baseline = row["baseline"]
            followup = row["followup"]
            expected = row["expected"]
            _require_room_light_fixture(isinstance(baseline, dict) and isinstance(followup, dict) and isinstance(expected, dict))
            with self.subTest(case_id=case_id):
                store = EnvironmentStateStore(ttl_ms=5000)
                _ingest_shared_vector(store, baseline)
                baseline_time = datetime.fromisoformat(str(baseline["observed_at"]).replace("Z", "+00:00"))
                baseline_state = store.current(now=baseline_time + timedelta(seconds=1))
                self.assertEqual(baseline_state["vision"]["room_light"]["observed_at"], baseline_time.isoformat())

                _ingest_shared_vector(store, followup, nonfinite=case_id == "nonfinite_numeric")
                followup_time = datetime.fromisoformat(str(followup["observed_at"]).replace("Z", "+00:00"))
                now = followup_time + timedelta(seconds=6 if case_id == "stale_freshness" else 1)
                current = store.current(now=now)
                room_light = current["vision"]["room_light"]
                accepted = room_light["observed_at"] == followup_time.isoformat()
                actual_validation = "valid" if accepted else "invalid"
                actual_claim = (
                    "camera-environment-estimate-high-confidence"
                    if accepted and not room_light["stale"] and room_light["confidence"] >= 0.9
                    else "unavailable"
                )
                actual_counts[actual_validation] += 1
                actual_counts["available" if actual_claim != "unavailable" else "unavailable"] += 1

                self.assertEqual(actual_validation, expected["validation_class"])
                self.assertEqual(actual_claim, expected["claim_class"])
                self.assertEqual(room_light["stale"], case_id == "stale_freshness")
                self.assertEqual(room_light["does_not_prove"], ROOM_LIGHT_DOES_NOT_PROVE)
                self.assertEqual(room_light["proof_ceiling"], "camera_environment_estimate_only")
                self.assertNotIn("room_light", current["state_queries"])
                serialized = json.dumps(current, allow_nan=False)
                self.assertNotIn(sentinel, serialized)
                self.assertNotIn(str(baseline["source_snapshot_id"]), serialized)
                self.assertNotIn(str(followup["source_snapshot_id"]), serialized)
                if accepted:
                    self.assertEqual(room_light["source"], followup["source"])
                    self.assertEqual(room_light["sequence"], followup["sequence"])

        self.assertEqual(actual_counts, {"valid": 7, "invalid": 8, "available": 6, "unavailable": 9})

    def test_invalid_room_light_observations_are_rejected_without_source_freshness(self) -> None:
        observed = datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC)
        valid_envelope = _room_light_envelope(observed)
        cases = [
            ("missing_envelope_schema", lambda envelope: envelope.pop("schema_version")),
            ("wrong_envelope_schema", lambda envelope: envelope.__setitem__("schema_version", 2)),
            ("boolean_envelope_schema", lambda envelope: envelope.__setitem__("schema_version", True)),
            ("wrong_topic", lambda envelope: envelope.__setitem__("topic", "/vision/room_light/wrong")),
            ("missing_msg_type", lambda envelope: envelope.pop("msg_type")),
            ("wrong_msg_type", lambda envelope: envelope.__setitem__("msg_type", "wrong")),
            ("missing_header", lambda envelope: envelope.pop("header")),
            ("header_not_object", lambda envelope: envelope.__setitem__("header", [])),
            ("seq_boolean", lambda envelope: envelope["header"].__setitem__("seq", True)),
            ("seq_negative", lambda envelope: envelope["header"].__setitem__("seq", -1)),
            ("seq_too_large", lambda envelope: envelope["header"].__setitem__("seq", 2**63)),
            ("stamp_not_numeric", lambda envelope: envelope["header"].__setitem__("stamp", "now")),
            ("stamp_nonfinite", lambda envelope: envelope["header"].__setitem__("stamp", float("inf"))),
            ("missing_frame_id", lambda envelope: envelope["header"].pop("frame_id")),
            ("blank_frame_id", lambda envelope: envelope["header"].__setitem__("frame_id", "   ")),
            ("control_frame_id", lambda envelope: envelope["header"].__setitem__("frame_id", "cam\n0")),
            ("payload_not_object", lambda envelope: envelope.__setitem__("payload", [])),
            ("missing_payload_type", lambda envelope: envelope["payload"].pop("type")),
            ("wrong_payload_type", lambda envelope: envelope["payload"].__setitem__("type", "wrong")),
            ("missing_payload_schema", lambda envelope: envelope["payload"].pop("schema_version")),
            ("wrong_payload_schema", lambda envelope: envelope["payload"].__setitem__("schema_version", 2)),
            ("boolean_payload_schema", lambda envelope: envelope["payload"].__setitem__("schema_version", True)),
            ("invalid_bucket", lambda envelope: envelope["payload"].__setitem__("observation_bucket", "unknown")),
            ("confidence_not_numeric", lambda envelope: envelope["payload"].__setitem__("confidence", "0.6")),
            ("confidence_boolean", lambda envelope: envelope["payload"].__setitem__("confidence", True)),
            ("confidence_nonfinite", lambda envelope: envelope["payload"].__setitem__("confidence", float("nan"))),
            ("confidence_out_of_range", lambda envelope: envelope["payload"].__setitem__("confidence", 1.01)),
            ("invalid_ambiguity", lambda envelope: envelope["payload"].__setitem__("daylight_ambiguity", "unknown")),
            ("missing_cues", lambda envelope: envelope["payload"].pop("cue_likelihoods")),
            ("cues_not_object", lambda envelope: envelope["payload"].__setitem__("cue_likelihoods", [])),
            ("missing_cue", lambda envelope: envelope["payload"]["cue_likelihoods"].pop("darkness")),
            ("cue_boolean", lambda envelope: envelope["payload"]["cue_likelihoods"].__setitem__("warm_light", False)),
            ("cue_nonfinite", lambda envelope: envelope["payload"]["cue_likelihoods"].__setitem__("daylight", float("inf"))),
            ("cue_out_of_range", lambda envelope: envelope["payload"]["cue_likelihoods"].__setitem__("darkness", -0.01)),
            ("missing_source", lambda envelope: envelope["payload"].pop("source")),
            ("wrong_source", lambda envelope: envelope["payload"].__setitem__("source", "camera_hub")),
            ("missing_source_class", lambda envelope: envelope["payload"].pop("source_class")),
            ("wrong_source_class", lambda envelope: envelope["payload"].__setitem__("source_class", "private")),
            ("missing_observed_at", lambda envelope: envelope["payload"].pop("observed_at")),
            ("invalid_observed_at", lambda envelope: envelope["payload"].__setitem__("observed_at", "not-a-time")),
            ("missing_observation_id", lambda envelope: envelope["payload"].pop("observation_id")),
            ("blank_observation_id", lambda envelope: envelope["payload"].__setitem__("observation_id", "   ")),
            ("control_observation_id", lambda envelope: envelope["payload"].__setitem__("observation_id", "obs\t1")),
            ("oversized_observation_id", lambda envelope: envelope["payload"].__setitem__("observation_id", "x" * 161)),
            ("missing_sequence", lambda envelope: envelope["payload"].pop("sequence")),
            ("sequence_not_object", lambda envelope: envelope["payload"].__setitem__("sequence", [])),
            ("frame_count_boolean", lambda envelope: envelope["payload"]["sequence"].__setitem__("frame_count", True)),
            ("frame_count_zero", lambda envelope: envelope["payload"]["sequence"].__setitem__("frame_count", 0)),
            ("first_frame_id_negative", lambda envelope: envelope["payload"]["sequence"].__setitem__("first_frame_id", -1)),
            ("last_frame_id_boolean", lambda envelope: envelope["payload"]["sequence"].__setitem__("last_frame_id", False)),
            ("window_boolean", lambda envelope: envelope["payload"]["sequence"].__setitem__("temporal_window_ms", True)),
            ("window_negative", lambda envelope: envelope["payload"]["sequence"].__setitem__("temporal_window_ms", -1)),
            ("first_after_last", lambda envelope: envelope["payload"]["sequence"].__setitem__("first_frame_id", 43)),
            ("impossible_frame_count", lambda envelope: envelope["payload"]["sequence"].__setitem__("frame_count", 3)),
            ("missing_model", lambda envelope: envelope["payload"].pop("model")),
            ("model_not_object", lambda envelope: envelope["payload"].__setitem__("model", [])),
            ("wrong_model_name", lambda envelope: envelope["payload"]["model"].__setitem__("name", "private-model")),
            ("wrong_model_kind", lambda envelope: envelope["payload"]["model"].__setitem__("kind", "ml")),
            ("missing_proof", lambda envelope: envelope["payload"].pop("proof_ceiling")),
            ("wrong_proof", lambda envelope: envelope["payload"].__setitem__("proof_ceiling", "physical_truth")),
            ("missing_nonclaims", lambda envelope: envelope["payload"].pop("does_not_prove")),
            ("nonclaims_not_list", lambda envelope: envelope["payload"].__setitem__("does_not_prove", "physical_room_light_state")),
            ("wrong_nonclaim", lambda envelope: envelope["payload"]["does_not_prove"].__setitem__(0, "private claim")),
            ("extra_nonclaim", lambda envelope: envelope["payload"]["does_not_prove"].append("private extra")),
        ]

        for name, mutate in cases:
            with self.subTest(name=name):
                envelope = deepcopy(valid_envelope)
                mutate(envelope)
                store = EnvironmentStateStore(ttl_ms=5000)
                store.ingest_camera_hub_envelope(envelope)

                current = store.current(now=observed + timedelta(seconds=1))
                self.assertNotIn("room_light", current["vision"])
                self.assertIsNone(current["observed_at"])
                self.assertIsNone(current["sources"]["camera_hub"]["updated_at"])

    def test_old_room_light_state_topic_is_not_consumed(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_camera_hub_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {"seq": 42, "stamp": datetime(2026, 5, 6, tzinfo=UTC).timestamp()},
                "payload": {"state": "on", "electric_light": {"state": "on"}},
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 0, 0, 1, tzinfo=UTC))
        self.assertNotIn("room_light", current["vision"])
        self.assertNotIn("room_light", current["state_queries"])

    def test_vision_source_provenance_stays_separate_from_camera_hub_source(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC)
        store.ingest_vision_envelope(
            _room_light_envelope(
                observed,
                frame_id="cam0",
                ambiguity="high",
                observation_id="obs-direct",
            ),
            source="vision_snapshot_processor",
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))
        sources = current["sources"]
        self.assertTrue(sources["camera_hub"]["stale"])
        self.assertFalse(sources["vision_snapshot_processor"]["stale"])
        self.assertEqual(
            current["vision"]["room_light"]["source"],
            "vision_snapshot_processor",
        )
        self.assertEqual(
            current["vision"]["room_light"]["source_snapshot_id"],
            "obs-direct",
        )

    def test_room_light_observation_reports_stale_freshness(self) -> None:
        store = EnvironmentStateStore(ttl_ms=1000)
        observed = datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC)
        store.ingest_vision_envelope(
            _room_light_envelope(
                observed,
                seq=1,
                frame_id="cam0",
                bucket="dim",
                confidence=0.71,
                ambiguity="low",
                cues={"warm_light": 0.4, "daylight": 0.1, "darkness": 0.6},
                observation_id="obs-stale",
            ),
            source="vision_snapshot_processor",
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 2, tzinfo=UTC))
        room_light = current["vision"]["room_light"]
        self.assertTrue(room_light["stale"])
        self.assertEqual(room_light["freshness"]["level"], "stale")
        self.assertEqual(room_light["freshness"]["reason"], "age_exceeds_ttl")
        self.assertEqual(room_light["freshness"]["ttl_ms"], 1000)
        self.assertNotIn("room_light", current["state_queries"])
    def test_camera_sword_sign_topic_preserves_primary_and_best_gesture(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_camera_hub_envelope(
            {
                "topic": "/vision/sword_sign/state",
                "header": {
                    "seq": 43,
                    "stamp": datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC).timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "type": "gesture_state",
                    "hand_detected": True,
                    "primary": "victory",
                    "gestures": {
                        "sword_sign": {"active": False, "confidence": 0.12, "label": 0},
                        "victory": {"active": True, "confidence": 0.96, "label": 1},
                        "none": {"active": False, "confidence": 0.01, "label": 2},
                    },
                    "stable": {
                        "gestures": {
                            "sword_sign": {
                                "active": False,
                                "confidence": 0.12,
                            }
                        }
                    },
                },
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        sword_sign = current["vision"]["sword_sign"]
        self.assertTrue(sword_sign["hand_detected"])
        self.assertEqual(sword_sign["primary_gesture"], "victory")
        self.assertEqual(sword_sign["best_gesture"]["name"], "victory")
        self.assertAlmostEqual(sword_sign["best_gesture"]["confidence"], 0.96)
        self.assertEqual(sword_sign["gestures"]["victory"]["label"], 1)
        self.assertFalse(sword_sign["active"])

    def test_camera_sword_sign_topic_omits_best_gesture_when_no_hand(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_camera_hub_envelope(
            {
                "topic": "/vision/sword_sign/state",
                "header": {
                    "seq": 44,
                    "stamp": datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC).timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "type": "gesture_state",
                    "hand_detected": False,
                    "primary": None,
                    "gestures": {
                        "sword_sign": {"active": False, "confidence": 0.0, "label": 0},
                        "victory": {"active": False, "confidence": 0.0, "label": 1},
                        "none": {"active": False, "confidence": 0.0, "label": 2},
                    },
                },
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        sword_sign = current["vision"]["sword_sign"]
        self.assertFalse(sword_sign["hand_detected"])
        self.assertIsNone(sword_sign["primary_gesture"])
        self.assertIsNone(sword_sign["best_gesture"])

    def test_empty_store_is_stale_but_available_as_payload(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)

        current = store.current(now=datetime.now(UTC) + timedelta(seconds=1))

        self.assertIsNone(current["observed_at"])
        self.assertTrue(current["stale"])
        self.assertEqual(current["appliances"], {})
        self.assertIsNone(current["age_ms"])

    def test_relations_can_be_updated_without_mixing_authorities(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)

        relations = store.update_relations(
            {
                "ha_request_id": "ha-1",
                "ha_execution_id": "exec-1",
                "snapshot_id": "env_1",
                "ignored": "nope",
            }
        )
        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        self.assertEqual(
            set(relations),
            {"ha_request_id", "ha_execution_id", "snapshot_id", "updated_at"},
        )
        self.assertEqual(relations["ha_request_id"], "ha-1")
        self.assertEqual(current["relations"]["ha_execution_id"], "exec-1")
        self.assertNotIn("ignored", current["relations"])

    def test_home_assistant_bridge_source_is_separate_from_event_feed(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_node_status(
            {
                "node_id": "home_assistant_bridge",
                "status": "ok",
                "phase": "ok",
                "observed_at": "2026-05-06T14:00:00+00:00",
                "ttl_ms": 5000,
                "detail": "ok",
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        self.assertTrue(current["sources"]["home_assistant"]["available"])
        self.assertFalse(current["sources"]["home_assistant"]["stale"])
        self.assertIn("home_assistant_events", current["sources"])


if __name__ == "__main__":
    unittest.main()
