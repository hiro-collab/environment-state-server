from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta

from environment_state_server.state import EnvironmentStateStore


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

    def test_camera_room_light_topic_updates_vision_state(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        store.ingest_camera_hub_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {
                    "seq": 42,
                    "stamp": datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC).timestamp(),
                    "frame_id": "camera",
                },
                "payload": {
                    "label": "electric_on_daylit",
                    "confidence": 0.91,
                    "lighting_type": "mixed",
                    "electric_light": {"state": "on"},
                    "daylight": {"state": "present"},
                    "observation_id": "obs-1",
                    "sequence": {
                        "frame_count": 2,
                        "first_frame_id": 41,
                        "last_frame_id": 42,
                        "temporal_window_ms": 1000,
                    },
                    "evidence": {
                        "model": {
                            "name": "room-light-heuristic-v1",
                        },
                    },
                },
            }
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        room_light = current["vision"]["room_light"]
        self.assertEqual(room_light["state"], "on")
        self.assertEqual(room_light["lighting_type"], "mixed")
        self.assertEqual(room_light["daylight_state"], "present")
        self.assertEqual(room_light["observation_id"], "obs-1")
        self.assertEqual(room_light["sequence"]["frame_count"], 2)
        self.assertEqual(room_light["sequence"]["temporal_window_ms"], 1000)
        self.assertEqual(room_light["source"], "camera_hub")
        self.assertEqual(room_light["freshness"]["level"], "fresh")
        self.assertEqual(room_light["freshness"]["age_ms"], 1000)

        query = current["state_queries"]["room_light"]
        self.assertTrue(query["available"])
        self.assertFalse(query["stale"])
        self.assertEqual(query["state"], "on")
        self.assertEqual(query["confidence_label"], "high")
        self.assertIn("点いている", query["answer_hint"])
        self.assertEqual(query["authority"], "vision_snapshot_processor")
        self.assertEqual(query["projected_by"], "environment_state_server")
        self.assertEqual(query["observed_at"], "2026-05-06T14:00:00+00:00")
        self.assertEqual(query["updated_at"], "2026-05-06T14:00:00+00:00")
        self.assertEqual(query["source_snapshot_id"], "obs-1")
        self.assertEqual(query["freshness"]["level"], "fresh")
        self.assertEqual(query["freshness"]["age_ms"], 1000)
        self.assertEqual(query["stale_reason"], "")
        self.assertEqual(query["evidence"]["source"], "camera_hub")
        self.assertEqual(query["evidence"]["topic"], "/vision/room_light/state")

    def test_vision_snapshot_processor_source_does_not_refresh_camera_hub_source(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC)

        store.ingest_vision_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {
                    "seq": 42,
                    "stamp": observed.timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "label": "dark",
                    "confidence": 0.86,
                    "electric_light": {"state": "off"},
                    "daylight": {"state": "absent"},
                    "sequence": {"frame_count": 3},
                },
            },
            source="vision_snapshot_processor",
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))
        sources = current["sources"]

        self.assertTrue(sources["camera_hub"]["stale"])
        self.assertFalse(sources["camera_hub"]["available"])
        self.assertFalse(sources["vision_snapshot_processor"]["stale"])
        self.assertEqual(sources["camera_hub"]["freshness"]["level"], "stale")
        self.assertEqual(sources["vision_snapshot_processor"]["freshness"]["level"], "fresh")
        self.assertEqual(
            current["vision"]["room_light"]["source"],
            "vision_snapshot_processor",
        )

    def test_room_light_state_query_explains_daylight_unknown(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC)

        store.ingest_vision_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {
                    "seq": 183,
                    "stamp": observed.timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "state": "unknown",
                    "label": "daylight",
                    "confidence": 0.0,
                    "lighting_type": "daylight",
                    "probabilities": {
                        "electric_on": 0.58,
                        "daylight_present": 0.75,
                        "dark": 0.13,
                    },
                    "electric_light": {"state": "unknown", "probability": 0.58},
                    "daylight": {"state": "present", "probability": 0.75},
                    "sequence": {
                        "frame_count": 2,
                        "first_frame_id": 182,
                        "last_frame_id": 183,
                        "temporal_window_ms": 1000,
                    },
                },
            },
            source="vision_snapshot_processor",
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))
        query = current["state_queries"]["room_light"]

        self.assertTrue(query["available"])
        self.assertFalse(query["stale"])
        self.assertEqual(query["state"], "unknown")
        self.assertEqual(query["confidence_label"], "low")
        self.assertIn("日光", query["answer_hint"])
        self.assertEqual(query["authority"], "vision_snapshot_processor")
        self.assertEqual(query["projected_by"], "environment_state_server")
        self.assertEqual(query["observed_at"], "2026-05-06T14:00:00+00:00")
        self.assertEqual(query["updated_at"], "2026-05-06T14:00:00+00:00")
        self.assertEqual(query["source_snapshot_id"], "cam0:183")
        self.assertEqual(query["stale_reason"], "")
        self.assertEqual(query["evidence"]["source"], "vision_snapshot_processor")
        self.assertAlmostEqual(query["evidence"]["electric_on_probability"], 0.58)
        self.assertAlmostEqual(query["evidence"]["daylight_present_probability"], 0.75)
        self.assertAlmostEqual(query["evidence"]["dark_probability"], 0.13)
        self.assertEqual(query["evidence"]["lighting_type"], "daylight")
        self.assertEqual(query["evidence"]["daylight_state"], "present")

        indicators = store.indicators_current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))
        self.assertEqual(
            indicators["environment"]["state_queries"]["room_light"]["state"],
            "unknown",
        )

    def test_room_light_state_query_reports_missing_sensor_state(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))
        query = current["state_queries"]["room_light"]

        self.assertFalse(query["available"])
        self.assertTrue(query["stale"])
        self.assertEqual(query["state"], "unknown")
        self.assertEqual(query["confidence_label"], "none")
        self.assertEqual(query["authority"], "vision_snapshot_processor")
        self.assertEqual(query["projected_by"], "environment_state_server")
        self.assertIsNone(query["observed_at"])
        self.assertIsNone(query["updated_at"])
        self.assertEqual(query["source_snapshot_id"], "")
        self.assertEqual(query["freshness"]["level"], "stale")
        self.assertEqual(query["freshness"]["reason"], "room_light_missing")
        self.assertEqual(query["stale_reason"], "room_light_missing")
        self.assertEqual(query["evidence"]["reason"], "room_light_missing")

    def test_room_light_state_query_marks_stale_sensor_state_unavailable(self) -> None:
        store = EnvironmentStateStore(ttl_ms=1000)
        observed = datetime(2026, 5, 6, 14, 0, 0, tzinfo=UTC)

        store.ingest_vision_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {
                    "seq": 1,
                    "stamp": observed.timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "state": "on",
                    "confidence": 0.91,
                    "lighting_type": "electric",
                    "electric_light": {"state": "on", "probability": 0.91},
                    "daylight": {"state": "absent", "probability": 0.12},
                },
            },
            source="vision_snapshot_processor",
        )

        current = store.current(now=datetime(2026, 5, 6, 14, 0, 2, tzinfo=UTC))
        query = current["state_queries"]["room_light"]

        self.assertFalse(query["available"])
        self.assertTrue(query["stale"])
        self.assertEqual(query["state"], "on")
        self.assertEqual(query["confidence_label"], "none")
        self.assertEqual(query["stale_reason"], "room_light_stale")
        self.assertEqual(query["freshness"]["level"], "stale")
        self.assertEqual(query["freshness"]["reason"], "age_exceeds_ttl")
        self.assertIn("古く", query["answer_hint"])

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
                "dify_issue_id": "HCA-1",
                "ha_request_id": "ha-1",
                "ha_execution_id": "exec-1",
                "snapshot_id": "env_1",
                "ignored": "nope",
            }
        )
        current = store.current(now=datetime(2026, 5, 6, 14, 0, 1, tzinfo=UTC))

        self.assertEqual(relations["dify_issue_id"], "HCA-1")
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
