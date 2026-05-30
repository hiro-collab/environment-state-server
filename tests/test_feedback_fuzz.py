from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from environment_state_server.feedback import StateQueryFeedbackStore
from environment_state_server.http_api import EnvironmentHttpServer
from environment_state_server.state import EnvironmentStateStore


class EnvironmentFeedbackFuzzTest(unittest.TestCase):
    def _start_server(self, feedback_path: Path) -> tuple[EnvironmentHttpServer, threading.Thread]:
        server = EnvironmentHttpServer(
            ("127.0.0.1", 0),
            store=EnvironmentStateStore(),
            api_token="secret",
            feedback_store=StateQueryFeedbackStore(feedback_path),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _post_json(self, url: str, payload: object) -> tuple[int, dict[str, object]]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def _get_json(self, url: str) -> tuple[int, dict[str, object]]:
        request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_feedback_endpoint_rejects_mutation_corpus_without_appending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.jsonl"
            server, thread = self._start_server(feedback_path)
            try:
                url = f"http://127.0.0.1:{server.server_port}/feedback/state-query"
                invalid_payloads = [
                    {},
                    {"target": "room_light", "user_label": "maybe"},
                    {"target": "bad_target", "user_label": "on"},
                    {"target": "room_light", "state_query_id": "other", "user_label": "on"},
                    {"target": "room_light", "state_query_id": ["room_light"], "user_label": "on"},
                    {"target": ["room_light"], "user_label": "on"},
                    {"target": "room_light", "user_label": {"label": "on"}},
                ]

                for payload in invalid_payloads:
                    status, body = self._post_json(url, payload)
                    self.assertEqual(status, 400, payload)
                    self.assertFalse(body.get("ok"), payload)

                self.assertFalse(feedback_path.exists())

                summary_url = f"http://127.0.0.1:{server.server_port}/feedback/state-query/summary"
                status, summary = self._get_json(summary_url)
                self.assertEqual(status, 200)
                self.assertEqual(summary["summary"]["status_counts"]["accepted"], 0)
                self.assertEqual(summary["summary"]["status_counts"]["duplicate"], 0)
                self.assertEqual(summary["summary"]["status_counts"]["rejected"], len(invalid_payloads))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_feedback_idempotency_and_sanitization_survive_mutated_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.jsonl"
            server, thread = self._start_server(feedback_path)
            try:
                url = f"http://127.0.0.1:{server.server_port}/feedback/state-query"
                payload = {
                    "target": "room_light",
                    "state_query_id": "room_light",
                    "idempotency_key": "state-query-feedback:fuzz:\n\tduplicate",
                    "snapshot_id": "env_20260530_102117_000001",
                    "current_snapshot_id": "x" * 500,
                    "predicted_state": "on",
                    "predicted_confidence_label": "high",
                    "user_label": "off",
                    "user_text": ("line1\nline2\t" + "x" * 800),
                    "source": "fuzz",
                    "workflow_version": "communication-fuzz",
                    "feedback_reason": "user_correction_after_state_query",
                    "source_context": "state_query",
                    "pending": {
                        "nested": [{"value": index} for index in range(150)],
                        "control": "\x00\x01unsafe",
                    },
                }

                status, first = self._post_json(url, payload)
                self.assertEqual(status, 200)
                self.assertFalse(first["duplicate"])

                status, second = self._post_json(url, payload)
                self.assertEqual(status, 200)
                self.assertTrue(second["duplicate"])
                self.assertEqual(second["feedback_id"], first["feedback_id"])

                records = feedback_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(records), 1)
                record = json.loads(records[0])
                self.assertEqual(record["idempotency_key"], "state-query-feedback:fuzz: duplicate")
                self.assertLessEqual(len(record["current_snapshot_id"]), 160)
                self.assertLessEqual(len(record["user_text"]), 500)
                self.assertNotIn("\n", record["user_text"])
                self.assertNotIn("\t", record["user_text"])
                self.assertEqual(len(record["pending"]["nested"]), 100)
                self.assertEqual(record["pending"]["control"], "unsafe")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_feedback_recent_ignores_bad_lines_and_bounds_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.jsonl"
            store = StateQueryFeedbackStore(feedback_path)
            for index, label in enumerate(["on", "off", "on"], start=1):
                store.append(
                    {
                        "target": "room_light",
                        "state_query_id": "room_light",
                        "idempotency_key": f"state-query-feedback:fuzz:{index}",
                        "snapshot_id": f"env_20260530_10210{index}_000001",
                        "predicted_state": "unknown",
                        "predicted_confidence_label": "low",
                        "user_label": label,
                        "user_text": label,
                    }
                )

            with feedback_path.open("a", encoding="utf-8", newline="\n") as file:
                file.write("{not-json\n")
                file.write("[1, 2, 3]\n")

            self.assertEqual(len(store.recent(target="room_light", limit=-50)), 1)
            self.assertEqual(len(store.recent(target="room_light", limit=5000)), 3)
            self.assertEqual(store.summary(target="room_light")["total_count"], 3)

    def test_feedback_endpoint_rejects_non_object_and_oversized_bodies_without_appending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.jsonl"
            server, thread = self._start_server(feedback_path)
            try:
                url = f"http://127.0.0.1:{server.server_port}/feedback/state-query"
                for raw_body in (b"[]", b"{not-json", b"{\"target\":\"room_light\"}" + (b"x" * 70000)):
                    request = urllib.request.Request(
                        url,
                        data=raw_body,
                        method="POST",
                        headers={
                            "Authorization": "Bearer secret",
                            "Content-Type": "application/json",
                        },
                    )
                    with self.assertRaises(urllib.error.HTTPError) as raised:
                        urllib.request.urlopen(request, timeout=2)
                    self.assertEqual(raised.exception.code, 400)
                    body = json.loads(raised.exception.read().decode("utf-8"))
                    self.assertEqual(body["error"], "invalid_json")

                self.assertFalse(feedback_path.exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
