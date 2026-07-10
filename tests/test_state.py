from __future__ import annotations

import json
import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta

from environment_state_server.state import EnvironmentStateStore


ROOM_LIGHT_DOES_NOT_PROVE = [
    "physical_room_light_state",
    "home_assistant_light_state",
]


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
