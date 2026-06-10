from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

from environment_state_server.feedback import StateQueryFeedbackStore
from environment_state_server.http_api import EnvironmentHttpServer
from environment_state_server.state import EnvironmentStateStore


class EnvironmentHttpServerTest(unittest.TestCase):
    def test_environment_current_requires_bearer_token(self) -> None:
        store = EnvironmentStateStore()
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/environment/current"
            try:
                urllib.request.urlopen(url, timeout=2)
            except urllib.error.HTTPError as exc:
                self.assertEqual(exc.code, 401)
            else:
                raise AssertionError("request without token unexpectedly succeeded")

            request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertEqual(body["schema_version"], 1)
            self.assertTrue(body["stale"])
            self.assertIn("snapshot_id", body)
            self.assertIn("sequence", body)
            self.assertIn("age_ms", body)
            self.assertIn("actions", body)
            self.assertTrue(body["capabilities"]["actions"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_environment_current_projects_room_light_learning(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC)
        store.ingest_vision_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {
                    "seq": 12,
                    "stamp": observed.timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "state": "unknown",
                    "confidence": 0.2,
                    "lighting_type": "daylight",
                    "daylight_state": "present",
                    "sequence": {"last_frame_id": 12},
                },
            },
            source="vision_snapshot_processor",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = StateQueryFeedbackStore(Path(temp_dir) / "state-query-feedback.jsonl")
            feedback_store.append(
                {
                    "target": "room_light",
                    "state_query_id": "room_light",
                    "idempotency_key": "state-query-feedback:test:env-1:on",
                    "snapshot_id": "env-1",
                    "predicted_state": "unknown",
                    "predicted_confidence_label": "low",
                    "user_label": "on",
                    "user_text": "ついてる",
                    "feedback_reason": "user_correction_after_state_query",
                    "source_context": "state_query",
                }
            )
            server = EnvironmentHttpServer(
                ("127.0.0.1", 0),
                store=store,
                api_token="secret",
                feedback_store=feedback_store,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/environment/current"
                request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
                with urllib.request.urlopen(request, timeout=2) as response:
                    body = json.loads(response.read().decode("utf-8"))

                learning = body["state_queries"]["room_light"]["learning"]
                self.assertEqual(learning["level"], "collecting")
                self.assertEqual(learning["accepted_count"], 1)
                self.assertEqual(learning["label_balance"]["on"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_environment_current_projects_room_light_calibration_from_home_assistant(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC)
        store.ingest_home_assistant_event(
            {
                "event": "execute_succeeded",
                "action_id": "light_on",
                "execution_id": "exec-light-on",
                "request_id": "ha-light-on",
                "executed": True,
                "status": "submitted",
                "timestamp": observed.isoformat(),
                "expected_effect": {
                    "domain": "switch",
                    "service": "turn_on",
                    "entity_id": "switch.demo_light",
                    "expected_state": "on",
                },
            }
        )
        store.ingest_vision_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {
                    "seq": 12,
                    "stamp": observed.timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "state": "unknown",
                    "confidence": 0.1,
                    "lighting_type": "daylight",
                    "daylight_state": "present",
                    "sequence": {"last_frame_id": 12},
                },
            },
            source="vision_snapshot_processor",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = StateQueryFeedbackStore(Path(temp_dir) / "state-query-feedback.jsonl")
            for index, label in enumerate(["on", "off"] * 6, start=1):
                feedback_store.append(
                    {
                        "target": "room_light",
                        "state_query_id": "room_light",
                        "idempotency_key": f"state-query-feedback:test:calibration:{index}:{label}",
                        "snapshot_id": f"env-calibration-{index}",
                        "predicted_state": "unknown",
                        "predicted_confidence_label": "low",
                        "user_label": label,
                        "user_text": "ついてる" if label == "on" else "消えてる",
                        "feedback_reason": "user_correction_after_light_action",
                        "source_context": "post_light_action",
                        "action_id": "light_on" if label == "on" else "light_off",
                        "expected_state": label,
                    }
                )
            server = EnvironmentHttpServer(
                ("127.0.0.1", 0),
                store=store,
                api_token="secret",
                feedback_store=feedback_store,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/environment/current"
                request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
                with urllib.request.urlopen(request, timeout=2) as response:
                    body = json.loads(response.read().decode("utf-8"))

                room_light = body["state_queries"]["room_light"]
                self.assertEqual(room_light["state"], "unknown")
                self.assertEqual(room_light["confidence_label"], "low")
                self.assertEqual(room_light["learning"]["level"], "reinforced")
                self.assertEqual(room_light["effective_state"], "on")
                self.assertEqual(room_light["effective_confidence_label"], "high")
                self.assertEqual(
                    room_light["effective_authority"],
                    "environment_state_server.calibration.home_assistant",
                )
                self.assertTrue(room_light["calibration"]["applied"])
                self.assertEqual(
                    room_light["calibration"]["reason"],
                    "fresh_home_assistant_light_state",
                )
                self.assertEqual(room_light["calibration"]["raw_state"], "unknown")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_indicators_current_is_public_and_sanitized(self) -> None:
        store = EnvironmentStateStore()
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/indicators/current"
            with urllib.request.urlopen(url, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertEqual(body["schema_version"], 1)
            self.assertIn("snapshot_id", body)
            self.assertIn("environment", body)
            self.assertIn("actions", body["environment"])
            self.assertNotIn("relations", body["environment"])
            self.assertNotIn("last_home_assistant_events", body["environment"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_indicators_current_allows_projection_visual_cors(self) -> None:
        store = EnvironmentStateStore()
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/indicators/current"
            request = urllib.request.Request(url, headers={"Origin": "http://127.0.0.1:3000"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertEqual(body["schema_version"], 1)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://127.0.0.1:3000")
            self.assertEqual(response.headers["Vary"], "Origin")

            options = urllib.request.Request(url, method="OPTIONS", headers={"Origin": "http://127.0.0.1:3000"})
            with urllib.request.urlopen(options, timeout=2) as response:
                self.assertEqual(response.status, 204)
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://127.0.0.1:3000")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_environment_current_waits_for_room_light_after_timestamp(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC)
        after_time = observed - timedelta(seconds=1)
        store.ingest_vision_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {
                    "seq": 12,
                    "stamp": observed.timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "state": "on",
                    "confidence": 0.92,
                    "lighting_type": "electric",
                    "electric_light": {"state": "on", "probability": 0.92},
                    "sequence": {"last_frame_id": 12},
                },
            },
            source="vision_snapshot_processor",
        )
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            after_text = after_time.isoformat()
            after = urllib.parse.quote(after_text, safe="")
            url = (
                f"http://127.0.0.1:{server.server_port}/environment/current"
                f"?wait_for=room_light&after={after}&timeout_ms=1500"
            )
            request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertTrue(body["wait_result"]["matched"])
            self.assertEqual(body["wait_result"]["target"], "room_light")
            self.assertEqual(body["wait_result"]["reason"], "matched")
            self.assertEqual(body["wait_result"]["timeout_ms"], 1500)
            self.assertEqual(body["wait_result"]["after"], after_text)
            self.assertEqual(body["wait_result"]["observed_at"], observed.isoformat())
            room_light = body["state_queries"]["room_light"]
            self.assertEqual(room_light["updated_at"], observed.isoformat())
            self.assertEqual(room_light["source_snapshot_id"], "cam0:12")
            self.assertEqual(room_light["stale_reason"], "")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_environment_current_wait_for_room_light_times_out_with_current_snapshot(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC)
        after_time = observed + timedelta(seconds=1)
        store.ingest_vision_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {
                    "seq": 12,
                    "stamp": observed.timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "state": "unknown",
                    "confidence": 0.0,
                    "electric_light": {"state": "unknown"},
                },
            },
            source="vision_snapshot_processor",
        )
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            after = urllib.parse.quote(after_time.isoformat(), safe="")
            url = (
                f"http://127.0.0.1:{server.server_port}/environment/current"
                f"?wait_for=room_light&after={after}&timeout_ms=20"
            )
            request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertFalse(body["wait_result"]["matched"])
            self.assertEqual(body["wait_result"]["target"], "room_light")
            self.assertEqual(body["wait_result"]["reason"], "timeout")
            self.assertEqual(body["wait_result"]["timeout_ms"], 20)
            self.assertEqual(body["wait_result"]["observed_at"], observed.isoformat())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_environment_current_wait_for_room_light_uses_observed_at_not_updated_at(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC) - timedelta(seconds=2)
        updated = datetime.now(UTC)
        after_time = observed + timedelta(seconds=1)
        store.ingest_vision_envelope(
            {
                "topic": "/vision/room_light/state",
                "header": {
                    "seq": 13,
                    "stamp": updated.timestamp(),
                    "frame_id": "cam0",
                },
                "payload": {
                    "state": "on",
                    "confidence": 0.9,
                    "observed_at": observed.isoformat(),
                    "electric_light": {"state": "on", "probability": 0.9},
                    "sequence": {"last_frame_id": 13},
                },
            },
            source="vision_snapshot_processor",
        )
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            after = urllib.parse.quote(after_time.isoformat(), safe="")
            url = (
                f"http://127.0.0.1:{server.server_port}/environment/current"
                f"?wait_for=room_light&after={after}&timeout_ms=20"
            )
            request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertFalse(body["wait_result"]["matched"])
            self.assertEqual(body["wait_result"]["reason"], "timeout")
            self.assertEqual(body["wait_result"]["observed_at"], observed.isoformat())
            room_light = body["state_queries"]["room_light"]
            self.assertGreater(room_light["updated_at"], after_time.isoformat())
            self.assertEqual(room_light["source_snapshot_id"], "cam0:13")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_environment_relations_can_be_updated_with_token(self) -> None:
        store = EnvironmentStateStore()
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/environment/relations"
            payload = json.dumps(
                {
                    "dify_issue_id": "HCA-42",
                    "ha_request_id": "ha-42",
                    "ha_execution_id": "exec-42",
                    "snapshot_id": "env_20260507_000000_000042",
                    "ignored": "not copied",
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": "Bearer secret",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertTrue(body["ok"])
            self.assertEqual(body["relations"]["dify_issue_id"], "HCA-42")
            self.assertEqual(body["relations"]["ha_execution_id"], "exec-42")
            self.assertNotIn("ignored", body["relations"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_state_query_feedback_can_be_posted_and_read_back(self) -> None:
        store = EnvironmentStateStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "state-query-feedback.jsonl"
            feedback_store = StateQueryFeedbackStore(feedback_path)
            server = EnvironmentHttpServer(
                ("127.0.0.1", 0),
                store=store,
                api_token="secret",
                feedback_store=feedback_store,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/feedback/state-query"
                snapshot_id = datetime.now(UTC).strftime("env_%Y%m%d_%H%M%S_000001")
                idempotency_key = f"state-query-feedback:conv-1:{snapshot_id}:on"
                payload = json.dumps(
                    {
                        "type": "state_query_feedback",
                        "target": "room_light",
                        "state_query_id": "room_light",
                        "idempotency_key": idempotency_key,
                        "snapshot_id": snapshot_id,
                        "current_snapshot_id": snapshot_id,
                        "predicted_state": "unknown",
                        "predicted_confidence_label": "low",
                        "user_label": "on",
                        "user_text": "ついてるよ\u0000\n",
                        "source": "dify",
                        "workflow_version": "home-control-assistant-test",
                        "feedback_reason": "user_correction_after_state_query",
                        "source_context": "state_query",
                        "pending": {
                            "authority": "vision_snapshot_processor",
                            "projected_by": "environment_state_server",
                            "answer_hint": "映像推定では断定できない。",
                            "evidence": {
                                "topic": "/vision/room_light/state",
                                "electric_on_probability": 0.58,
                            },
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                request = urllib.request.Request(
                    url,
                    data=payload,
                    method="POST",
                    headers={
                        "Authorization": "Bearer secret",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    body = json.loads(response.read().decode("utf-8"))

                self.assertEqual(response.status, 200)
                self.assertTrue(body["ok"])
                self.assertTrue(body["feedback_id"].startswith("sqf_"))
                self.assertTrue(body["received_snapshot_id"].startswith("env_"))
                self.assertFalse(body["duplicate"])
                self.assertEqual(body["status"], "accepted")
                self.assertEqual(body["warnings"], [])
                self.assertTrue(feedback_path.exists())

                record = json.loads(feedback_path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(record["schema_version"], 1)
                self.assertEqual(record["target"], "room_light")
                self.assertEqual(record["state_query_id"], "room_light")
                self.assertEqual(
                    record["idempotency_key"],
                    idempotency_key,
                )
                self.assertEqual(record["user_label"], "on")
                self.assertEqual(record["user_text"], "ついてるよ")
                self.assertEqual(record["authority"], "user_feedback")
                self.assertEqual(record["source"], "dify")
                self.assertEqual(record["workflow_version"], "home-control-assistant-test")
                self.assertEqual(record["feedback_reason"], "user_correction_after_state_query")
                self.assertEqual(record["source_context"], "state_query")
                self.assertEqual(record["pending"]["evidence"]["topic"], "/vision/room_light/state")

                duplicate_request = urllib.request.Request(
                    url,
                    data=payload,
                    method="POST",
                    headers={
                        "Authorization": "Bearer secret",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(duplicate_request, timeout=2) as response:
                    duplicate = json.loads(response.read().decode("utf-8"))

                self.assertEqual(response.status, 200)
                self.assertTrue(duplicate["ok"])
                self.assertTrue(duplicate["duplicate"])
                self.assertEqual(duplicate["feedback_id"], body["feedback_id"])
                self.assertEqual(len(feedback_path.read_text(encoding="utf-8").splitlines()), 1)

                recent_url = f"http://127.0.0.1:{server.server_port}/feedback/state-query/recent?target=room_light&limit=20"
                request = urllib.request.Request(
                    recent_url,
                    headers={"Authorization": "Bearer secret"},
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    recent = json.loads(response.read().decode("utf-8"))

                self.assertEqual(response.status, 200)
                self.assertTrue(recent["ok"])
                self.assertEqual(len(recent["items"]), 1)
                self.assertEqual(recent["items"][0]["feedback_id"], body["feedback_id"])

                summary_url = f"http://127.0.0.1:{server.server_port}/feedback/state-query/summary?target=room_light"
                request = urllib.request.Request(
                    summary_url,
                    headers={"Authorization": "Bearer secret"},
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    summary = json.loads(response.read().decode("utf-8"))

                self.assertEqual(response.status, 200)
                self.assertTrue(summary["ok"])
                self.assertEqual(summary["summary"]["total_count"], 1)
                self.assertEqual(summary["summary"]["label_counts"]["on"], 1)
                self.assertEqual(summary["summary"]["reason_counts"]["user_correction_after_state_query"], 1)
                self.assertEqual(summary["summary"]["source_context_counts"]["state_query"], 1)
                self.assertEqual(summary["summary"]["status_counts"]["accepted"], 1)
                self.assertEqual(summary["summary"]["status_counts"]["accepted_with_warning"], 0)
                self.assertEqual(summary["summary"]["status_counts"]["duplicate"], 1)
                self.assertEqual(summary["summary"]["status_counts"]["rejected"], 0)
                self.assertEqual(summary["summary"]["runtime_status_counts"]["duplicate"], 1)
                self.assertEqual(summary["summary"]["learning"]["level"], "collecting")
                self.assertEqual(summary["summary"]["learning"]["level_index"], 1)
                self.assertEqual(summary["summary"]["learning"]["accepted_count"], 1)
                self.assertEqual(summary["summary"]["learning"]["label_balance"]["on"], 1)
                self.assertIn(
                    "duplicate_feedback_seen",
                    [problem["code"] for problem in summary["summary"]["learning"]["problems"]],
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_state_query_feedback_summary_groups_post_light_action_context(self) -> None:
        store = EnvironmentStateStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "state-query-feedback.jsonl"
            server = EnvironmentHttpServer(
                ("127.0.0.1", 0),
                store=store,
                api_token="secret",
                feedback_store=StateQueryFeedbackStore(feedback_path),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/feedback/state-query"
                payload = json.dumps(
                    {
                        "target": "room_light",
                        "state_query_id": "room_light",
                        "idempotency_key": "state-query-feedback:conv-2:env_20260507_141500_000002:off",
                        "snapshot_id": "env_20260507_141500_000002",
                        "predicted_state": "unknown",
                        "predicted_confidence_label": "low",
                        "user_label": "off",
                        "user_text": "消えてるよ",
                        "source": "dify",
                        "workflow_version": "home-control-assistant-test",
                        "feedback_reason": "user_correction_after_light_action",
                        "source_context": "post_light_action",
                        "action_id": "light_off",
                        "issue_id": "HCA-42",
                        "expected_state": "off",
                        "pending": {
                            "authority": "vision_snapshot_processor",
                            "evidence": {},
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                request = urllib.request.Request(
                    url,
                    data=payload,
                    method="POST",
                    headers={
                        "Authorization": "Bearer secret",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    body = json.loads(response.read().decode("utf-8"))

                self.assertEqual(response.status, 200)
                self.assertTrue(body["ok"])

                record = json.loads(feedback_path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(record["feedback_reason"], "user_correction_after_light_action")
                self.assertEqual(record["source_context"], "post_light_action")
                self.assertEqual(record["action_id"], "light_off")
                self.assertEqual(record["issue_id"], "HCA-42")
                self.assertEqual(record["expected_state"], "off")

                summary_url = f"http://127.0.0.1:{server.server_port}/feedback/state-query/summary?target=room_light"
                request = urllib.request.Request(
                    summary_url,
                    headers={"Authorization": "Bearer secret"},
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    summary = json.loads(response.read().decode("utf-8"))

                self.assertEqual(response.status, 200)
                self.assertEqual(summary["summary"]["reason_counts"]["user_correction_after_light_action"], 1)
                self.assertEqual(summary["summary"]["source_context_counts"]["post_light_action"], 1)
                self.assertEqual(summary["summary"]["action_counts"]["light_off"], 1)
                self.assertEqual(summary["summary"]["expected_state_counts"]["off"], 1)
                self.assertEqual(summary["summary"]["learning"]["post_action_count"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_state_query_feedback_summary_reports_learning_level_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = StateQueryFeedbackStore(Path(temp_dir) / "state-query-feedback.jsonl")

            initial = feedback_store.summary(target="room_light")
            self.assertEqual(initial["learning"]["level"], "none")
            self.assertEqual(initial["learning"]["level_index"], 0)
            self.assertIn("no_feedback", [problem["code"] for problem in initial["learning"]["problems"]])
            self.assertFalse(initial["learning"]["ok"])

            for index, label in enumerate(["on", "off", "on", "off", "on", "off"], start=1):
                feedback_store.append(
                    {
                        "target": "room_light",
                        "state_query_id": "room_light",
                        "idempotency_key": f"state-query-feedback:test:{index}:{label}",
                        "snapshot_id": f"env_20260507_1415{index:02d}_000001",
                        "predicted_state": "unknown",
                        "predicted_confidence_label": "low",
                        "user_label": label,
                        "user_text": "ついてる" if label == "on" else "消えてる",
                        "feedback_reason": "user_correction_after_state_query",
                        "source_context": "state_query",
                    }
                )

            usable = feedback_store.summary(target="room_light")
            self.assertEqual(usable["learning"]["level"], "usable")
            self.assertEqual(usable["learning"]["level_index"], 3)
            self.assertEqual(usable["learning"]["accepted_count"], 6)
            self.assertEqual(usable["learning"]["label_balance"]["on"], 3)
            self.assertEqual(usable["learning"]["label_balance"]["off"], 3)
            self.assertIn(
                "no_post_action_feedback",
                [problem["code"] for problem in usable["learning"]["problems"]],
            )

            for index, label in enumerate(["on", "off", "on", "off", "on", "off"], start=7):
                feedback_store.append(
                    {
                        "target": "room_light",
                        "state_query_id": "room_light",
                        "idempotency_key": f"state-query-feedback:test:{index}:{label}",
                        "snapshot_id": f"env_20260507_1415{index:02d}_000001",
                        "predicted_state": "unknown",
                        "predicted_confidence_label": "low",
                        "user_label": label,
                        "user_text": "ついてる" if label == "on" else "消えてる",
                        "feedback_reason": "user_correction_after_light_action",
                        "source_context": "post_light_action",
                        "action_id": "light_on" if label == "on" else "light_off",
                        "expected_state": label,
                    }
                )

            reinforced = feedback_store.summary(target="room_light")
            self.assertEqual(reinforced["learning"]["level"], "reinforced")
            self.assertEqual(reinforced["learning"]["level_index"], 4)
            self.assertEqual(reinforced["learning"]["accepted_count"], 12)
            self.assertEqual(reinforced["learning"]["post_action_count"], 6)
            self.assertNotIn(
                "no_post_action_feedback",
                [problem["code"] for problem in reinforced["learning"]["problems"]],
            )
            self.assertTrue(reinforced["learning"]["ok"])

    def test_state_query_feedback_summary_reports_prediction_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store = StateQueryFeedbackStore(Path(temp_dir) / "state-query-feedback.jsonl")
            feedback_store.append(
                {
                    "target": "room_light",
                    "state_query_id": "room_light",
                    "idempotency_key": "state-query-feedback:test:quality:match",
                    "snapshot_id": "env_quality_match",
                    "predicted_state": "off",
                    "predicted_confidence_label": "high",
                    "user_label": "off",
                    "user_text": "消えてる",
                    "feedback_reason": "user_correction_after_state_query",
                    "source_context": "state_query",
                }
            )
            feedback_store.append(
                {
                    "target": "room_light",
                    "state_query_id": "room_light",
                    "idempotency_key": "state-query-feedback:test:quality:conflict",
                    "snapshot_id": "env_quality_conflict",
                    "predicted_state": "off",
                    "predicted_confidence_label": "high",
                    "user_label": "on",
                    "user_text": "ついてる",
                    "feedback_reason": "user_correction_after_state_query",
                    "source_context": "state_query",
                }
            )

            summary = feedback_store.summary(target="room_light")
            quality = summary["learning"]["prediction_quality"]

            self.assertEqual(quality["comparable_count"], 2)
            self.assertEqual(quality["match_count"], 1)
            self.assertEqual(quality["conflict_count"], 1)
            self.assertEqual(quality["high_confidence_conflict_count"], 1)
            self.assertIn(
                "high_confidence_prediction_conflicts",
                [problem["code"] for problem in summary["learning"]["problems"]],
            )

    def test_state_query_feedback_accepts_stale_pending_with_warning(self) -> None:
        store = EnvironmentStateStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "state-query-feedback.jsonl"
            server = EnvironmentHttpServer(
                ("127.0.0.1", 0),
                store=store,
                api_token="secret",
                feedback_store=StateQueryFeedbackStore(feedback_path),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                old_created_at = (datetime.now(UTC) - timedelta(seconds=300)).isoformat()
                url = f"http://127.0.0.1:{server.server_port}/feedback/state-query"
                payload = json.dumps(
                    {
                        "target": "room_light",
                        "state_query_id": "room_light",
                        "snapshot_id": "env_20260507_141500_000001",
                        "predicted_state": "unknown",
                        "predicted_confidence_label": "low",
                        "user_label": "off",
                        "user_text": "消えてるよ",
                        "workflow_version": "home-control-assistant-test",
                        "pending": {
                            "created_at": old_created_at,
                            "authority": "vision_snapshot_processor",
                            "evidence": {},
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                request = urllib.request.Request(
                    url,
                    data=payload,
                    method="POST",
                    headers={
                        "Authorization": "Bearer secret",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    body = json.loads(response.read().decode("utf-8"))

                self.assertEqual(response.status, 200)
                self.assertTrue(body["ok"])
                self.assertEqual(body["status"], "accepted_with_warning")
                self.assertEqual(body["warnings"], ["pending_stale"])

                record = json.loads(feedback_path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(record["status"], "accepted_with_warning")
                self.assertEqual(record["warnings"], ["pending_stale"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_state_query_feedback_rejects_invalid_label(self) -> None:
        store = EnvironmentStateStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            server = EnvironmentHttpServer(
                ("127.0.0.1", 0),
                store=store,
                api_token="secret",
                feedback_store=StateQueryFeedbackStore(Path(temp_dir) / "feedback.jsonl"),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/feedback/state-query"
                payload = json.dumps(
                    {
                        "target": "room_light",
                        "state_query_id": "room_light",
                        "user_label": "maybe",
                        "user_text": "たぶん",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                request = urllib.request.Request(
                    url,
                    data=payload,
                    method="POST",
                    headers={
                        "Authorization": "Bearer secret",
                        "Content-Type": "application/json",
                    },
                )

                try:
                    urllib.request.urlopen(request, timeout=2)
                except urllib.error.HTTPError as exc:
                    self.assertEqual(exc.code, 400)
                    body = json.loads(exc.read().decode("utf-8"))
                    self.assertEqual(body["error"], "unsupported_user_label")
                else:
                    raise AssertionError("invalid feedback label unexpectedly succeeded")

                summary_url = f"http://127.0.0.1:{server.server_port}/feedback/state-query/summary?target=room_light"
                request = urllib.request.Request(
                    summary_url,
                    headers={"Authorization": "Bearer secret"},
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    summary = json.loads(response.read().decode("utf-8"))

                self.assertEqual(response.status, 200)
                self.assertEqual(summary["summary"]["status_counts"]["accepted"], 0)
                self.assertEqual(summary["summary"]["status_counts"]["accepted_with_warning"], 0)
                self.assertEqual(summary["summary"]["status_counts"]["duplicate"], 0)
                self.assertEqual(summary["summary"]["status_counts"]["rejected"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_ready_requires_fresh_home_assistant_bridge_status(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/ready"
            try:
                urllib.request.urlopen(url, timeout=2)
            except urllib.error.HTTPError as exc:
                self.assertEqual(exc.code, 503)
                body = json.loads(exc.read().decode("utf-8"))
            else:
                raise AssertionError("ready unexpectedly succeeded without source status")

            self.assertFalse(body["ready"])
            self.assertIn("home_assistant_bridge_unavailable_or_stale", body["reasons"])
            self.assertIn("camera_hub_stale", body["reasons"])

            observed_at = datetime.now(UTC).isoformat()
            store.ingest_node_status(
                {
                    "node_id": "home_assistant_bridge",
                    "status": "ok",
                    "phase": "ok",
                    "observed_at": observed_at,
                    "ttl_ms": 5000,
                    "detail": "ok",
                }
            )
            store.ingest_camera_hub_envelope(
                {
                    "topic": "/camera/status",
                    "header": {
                        "seq": 1,
                        "stamp": datetime.fromisoformat(observed_at).timestamp(),
                        "frame_id": "camera",
                    },
                    "payload": {
                        "camera": {"opened": True},
                        "capture": {"read_fps": 30.0},
                        "fps": 30.0,
                    },
                }
            )

            with urllib.request.urlopen(url, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertTrue(body["ready"])
            self.assertEqual(body["reasons"], [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
