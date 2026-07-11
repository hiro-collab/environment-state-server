from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from environment_state_server.feedback import StateQueryFeedbackStore
from environment_state_server.http_api import EnvironmentHttpServer
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
    "canonical_camera_hub", "canonical_vision_snapshot_processor", "malformed_nested_sequence",
    "wrong_numeric_type", "nonfinite_numeric", "out_of_range_numeric", "wrong_case",
    "stale_freshness", "reversed_ordered_nonclaims", "non_room_light",
    "unknown_field_non_echo", "wrong_proof_ceiling",
    "responsiveness_same_identity_material_movement",
    "responsiveness_changed_identity_no_material_movement",
    "responsiveness_changed_identity_material_movement",
]
SHARED_VECTOR_PAYLOAD_KEYS = {
    "type", "schema_version", "observation_bucket", "confidence", "daylight_ambiguity",
    "cue_likelihoods", "source", "source_class", "observed_at", "observation_id",
    "source_snapshot_id", "sequence", "model", "freshness", "proof_ceiling",
    "does_not_prove",
}


def _http_expected_classes(case_id: str) -> tuple[str, str, str, str]:
    invalid = {
        "malformed_nested_sequence", "wrong_numeric_type", "nonfinite_numeric",
        "out_of_range_numeric", "wrong_case", "reversed_ordered_nonclaims",
        "non_room_light", "wrong_proof_ceiling",
    }
    if case_id in invalid:
        return "invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate"
    if case_id == "stale_freshness":
        return "valid", "unavailable", "partial", "material_camera_environment_estimate_change_with_new_observation"
    if case_id == "responsiveness_same_identity_material_movement":
        return "valid", "camera-environment-estimate-high-confidence", "fail", "material_camera_environment_estimate_change_without_new_observation"
    if case_id == "responsiveness_changed_identity_no_material_movement":
        return "valid", "camera-environment-estimate-high-confidence", "fail", "new_observation_without_material_camera_environment_estimate_change"
    return "valid", "camera-environment-estimate-high-confidence", "pass", "material_camera_environment_estimate_change_with_new_observation"


def _require_room_light_fixture(condition: bool) -> None:
    if not condition:
        raise AssertionError(ROOM_LIGHT_FIXTURE_INVALID) from None


def _assert_safe_http_shared_value(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_room_light_fixture(isinstance(key, str) and key.isprintable() and len(key) <= 80)
            _assert_safe_http_shared_value(nested)
    elif isinstance(value, list):
        _require_room_light_fixture(len(value) <= 20)
        for nested in value:
            _assert_safe_http_shared_value(nested)
    elif isinstance(value, str):
        _require_room_light_fixture(value.isprintable() and len(value) <= 200 and "\\" not in value and "://" not in value)
    else:
        _require_room_light_fixture(value is None or isinstance(value, (bool, int, float)))
        if isinstance(value, float):
            _require_room_light_fixture(math.isfinite(value))


def _validate_http_shared_fixture(data: object) -> dict[str, object]:
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
            row_keys = {"case_id", "baseline", "followup", "expected"}
            if case_id == "nonfinite_numeric":
                row_keys.add("synthetic_numeric_class")
                _require_room_light_fixture(row["synthetic_numeric_class"] == "followup_confidence_nan")
            _require_room_light_fixture(set(row) == row_keys)
            expected = row["expected"]
            _require_room_light_fixture(isinstance(expected, dict))
            _require_room_light_fixture(set(expected) == {"validation_class", "claim_class", "responsiveness_class", "delta_class", "unknown_echo_class"})
            _require_room_light_fixture(tuple(expected[key] for key in ("validation_class", "claim_class", "responsiveness_class", "delta_class")) == _http_expected_classes(case_id))
            _require_room_light_fixture(expected["unknown_echo_class"] == "not_echoed")
            for phase in ("baseline", "followup"):
                payload = row[phase]
                _require_room_light_fixture(isinstance(payload, dict))
                keys = set(SHARED_VECTOR_PAYLOAD_KEYS)
                if phase == "followup" and case_id == "unknown_field_non_echo":
                    keys.add("unknown_test_field")
                    _require_room_light_fixture(payload["unknown_test_field"] == data["unknown_field_sentinel"])
                _require_room_light_fixture(set(payload) == keys)
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
        _require_room_light_fixture(sum(row["expected"]["validation_class"] == "valid" for row in cases) == 7)
        _require_room_light_fixture(sum(row["expected"]["claim_class"] == "camera-environment-estimate-high-confidence" for row in cases) == 6)
        _assert_safe_http_shared_value(data)
        return data
    except AssertionError:
        raise
    except (AttributeError, KeyError, OverflowError, RecursionError, TypeError, ValueError):
        raise AssertionError(ROOM_LIGHT_FIXTURE_INVALID) from None


def _load_http_shared_fixture() -> dict[str, object]:
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
    return _validate_http_shared_fixture(data)


def _http_shared_envelope(vector: dict[str, object]) -> tuple[dict[str, object], str]:
    payload = deepcopy(vector)
    source = str(payload.pop("source"))
    payload.pop("freshness")
    payload.pop("source_snapshot_id")
    payload["source"] = "vision_snapshot_processor"
    observed = datetime.fromisoformat(str(payload["observed_at"]).replace("Z", "+00:00"))
    sequence = payload["sequence"]
    _require_room_light_fixture(isinstance(sequence, dict))
    seq = sequence.get("last_frame_id")
    return {
        "schema_version": 1,
        "topic": "/vision/room_light/observation",
        "msg_type": "vision_snapshot_processor/RoomLightObservation",
        "header": {"seq": seq if type(seq) is int else 0, "stamp": observed.timestamp(), "frame_id": "synthetic-camera"},
        "payload": payload,
    }, source


def _ingest_http_shared_vector(store: EnvironmentStateStore, vector: dict[str, object], *, nonfinite: bool = False) -> None:
    envelope, source = _http_shared_envelope(vector)
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
    received: datetime | None = None,
    seq: int = 12,
    bucket: str = "balanced",
    confidence: float = 0.6,
    ambiguity: str = "medium",
    cues: dict[str, float] | None = None,
    observation_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "topic": "/vision/room_light/observation",
        "msg_type": "vision_snapshot_processor/RoomLightObservation",
        "header": {
            "seq": seq,
            "stamp": (received or observed).timestamp(),
            "frame_id": "cam0",
        },
        "payload": {
            "type": "room_light_observation",
            "schema_version": 1,
            "observation_bucket": bucket,
            "confidence": confidence,
            "daylight_ambiguity": ambiguity,
            "cue_likelihoods": cues
            if cues is not None
            else {"warm_light": 0.2, "daylight": 0.7, "darkness": 0.1},
            "source": "vision_snapshot_processor",
            "source_class": "camera_environment_estimate",
            "observed_at": observed.isoformat(),
            "observation_id": observation_id or f"obs-{seq}",
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
            self.assertNotIn("actions", body)
            self.assertNotIn("action_readiness", body)
            self.assertNotIn("actions", body["capabilities"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_environment_current_keeps_feedback_separate_from_room_light_observation(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC)
        store.ingest_vision_envelope(
            _room_light_envelope(
                observed,
                seq=12,
                bucket="balanced",
                confidence=0.6,
                ambiguity="high",
                cues={"warm_light": 0.2, "daylight": 0.8, "darkness": 0.1},
            ),
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
                    "user_text": "on",
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

                room_light = body["vision"]["room_light"]
                self.assertEqual(room_light["type"], "room_light_observation")
                self.assertNotIn("learning", room_light)
                self.assertNotIn("room_light", body["state_queries"])
                self.assertEqual(
                    feedback_store.summary(target="room_light")["learning"]["accepted_count"],
                    1,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    @unittest.skipUnless(SHARED_VECTOR_PATH, f"set {SHARED_VECTOR_ENV} to run Parent shared-vector HTTP sensitivity")
    def test_shared_room_light_http_fixture_shape_sensitivity(self) -> None:
        fixture = _load_http_shared_fixture()
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
                    _validate_http_shared_fixture(candidate)

    def test_shared_room_light_http_loader_failures_are_fixed_and_non_echoing(self) -> None:
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
                            _load_http_shared_fixture()
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

    @unittest.skipUnless(SHARED_VECTOR_PATH, f"set {SHARED_VECTOR_ENV} to run Parent shared-vector HTTP consumer")
    def test_shared_room_light_vectors_use_http_safe_projection(self) -> None:
        fixture = _load_http_shared_fixture()
        cases = fixture["cases"]
        _require_room_light_fixture(isinstance(cases, list))
        sentinel = str(fixture["unknown_field_sentinel"])
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=EnvironmentStateStore(), api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        actual_counts = {"valid": 0, "invalid": 0, "available": 0, "unavailable": 0}
        try:
            for row in cases:
                _require_room_light_fixture(isinstance(row, dict))
                case_id = str(row["case_id"])
                baseline = row["baseline"]
                followup = row["followup"]
                expected = row["expected"]
                _require_room_light_fixture(isinstance(baseline, dict) and isinstance(followup, dict) and isinstance(expected, dict))
                with self.subTest(case_id=case_id):
                    store = EnvironmentStateStore(ttl_ms=5000)
                    server.store = store
                    _ingest_http_shared_vector(store, baseline)
                    _ingest_http_shared_vector(store, followup, nonfinite=case_id == "nonfinite_numeric")
                    followup_time = datetime.fromisoformat(str(followup["observed_at"]).replace("Z", "+00:00"))
                    now = followup_time + timedelta(seconds=6 if case_id == "stale_freshness" else 1)
                    url = f"http://127.0.0.1:{server.server_port}/environment/current"
                    request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
                    with patch("environment_state_server.state.utc_now", return_value=now):
                        with urllib.request.urlopen(request, timeout=2) as response:
                            body = json.loads(response.read().decode("utf-8"))

                    self.assertEqual(response.status, 200)
                    room_light = body["vision"]["room_light"]
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
                    self.assertNotIn("room_light", body["state_queries"])
                    self.assertNotIn("feedback", body)
                    serialized = json.dumps(body, allow_nan=False)
                    self.assertNotIn(sentinel, serialized)
                    self.assertNotIn(str(baseline["source_snapshot_id"]), serialized)
                    self.assertNotIn(str(followup["source_snapshot_id"]), serialized)
            self.assertEqual(actual_counts, {"valid": 7, "invalid": 8, "available": 6, "unavailable": 9})
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
            self.assertNotIn("actions", body["environment"])
            self.assertNotIn("action_readiness", body["environment"])
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

    def test_environment_current_waits_for_vision_room_light_after_timestamp(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC)
        after_time = observed - timedelta(seconds=1)
        store.ingest_vision_envelope(
            _room_light_envelope(
                observed,
                seq=12,
                bucket="bright",
                confidence=0.92,
                ambiguity="low",
                cues={"warm_light": 0.9, "daylight": 0.1, "darkness": 0.0},
            ),
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
                f"?wait_for=vision.room_light&after={after}&timeout_ms=1500"
            )
            request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertTrue(body["wait_result"]["matched"])
            self.assertEqual(body["wait_result"]["target"], "vision.room_light")
            self.assertEqual(body["wait_result"]["reason"], "matched")
            self.assertEqual(body["wait_result"]["timeout_ms"], 1500)
            self.assertEqual(body["wait_result"]["after"], after_text)
            self.assertEqual(body["wait_result"]["observed_at"], observed.isoformat())
            room_light = body["vision"]["room_light"]
            self.assertEqual(room_light["observed_at"], observed.isoformat())
            self.assertEqual(room_light["source_snapshot_id"], "obs-12")
            self.assertNotIn("room_light", body["state_queries"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
    def test_environment_current_wait_for_vision_room_light_times_out_with_current_snapshot(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC)
        after_time = observed + timedelta(seconds=1)
        store.ingest_vision_envelope(
            _room_light_envelope(
                observed,
                seq=12,
                bucket="balanced",
                confidence=0.4,
                ambiguity="high",
                cues={"warm_light": 0.2, "daylight": 0.7, "darkness": 0.1},
            ),
            source="vision_snapshot_processor",
        )
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            after = urllib.parse.quote(after_time.isoformat(), safe="")
            url = (
                f"http://127.0.0.1:{server.server_port}/environment/current"
                f"?wait_for=vision.room_light&after={after}&timeout_ms=20"
            )
            request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertFalse(body["wait_result"]["matched"])
            self.assertEqual(body["wait_result"]["target"], "vision.room_light")
            self.assertEqual(body["wait_result"]["reason"], "timeout")
            self.assertEqual(body["wait_result"]["timeout_ms"], 20)
            self.assertEqual(body["wait_result"]["observed_at"], observed.isoformat())
            self.assertEqual(body["vision"]["room_light"]["source_snapshot_id"], "obs-12")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
    def test_environment_current_wait_for_vision_room_light_uses_observed_at(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC) - timedelta(seconds=2)
        received = datetime.now(UTC)
        after_time = observed + timedelta(seconds=1)
        store.ingest_vision_envelope(
            _room_light_envelope(
                observed,
                received=received,
                seq=13,
                bucket="bright",
                confidence=0.9,
                ambiguity="low",
                cues={"warm_light": 0.9, "daylight": 0.1, "darkness": 0.0},
            ),
            source="vision_snapshot_processor",
        )
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            after = urllib.parse.quote(after_time.isoformat(), safe="")
            url = (
                f"http://127.0.0.1:{server.server_port}/environment/current"
                f"?wait_for=vision.room_light&after={after}&timeout_ms=20"
            )
            request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertFalse(body["wait_result"]["matched"])
            self.assertEqual(body["wait_result"]["reason"], "timeout")
            self.assertEqual(body["wait_result"]["observed_at"], observed.isoformat())
            room_light = body["vision"]["room_light"]
            self.assertEqual(room_light["observed_at"], observed.isoformat())
            self.assertEqual(room_light["source_snapshot_id"], "obs-13")
            self.assertIn("updated_at", room_light)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_environment_current_wait_rejects_invalid_room_light_observation(self) -> None:
        store = EnvironmentStateStore(ttl_ms=5000)
        observed = datetime.now(UTC)
        envelope = _room_light_envelope(
            observed,
            seq=14,
            bucket="bright",
            confidence=0.9,
            ambiguity="low",
            cues={"warm_light": 0.9, "daylight": 0.1, "darkness": 0.0},
        )
        payload = envelope["payload"]
        assert isinstance(payload, dict)
        payload.pop("proof_ceiling")
        store.ingest_vision_envelope(
            envelope,
            source="vision_snapshot_processor",
        )
        server = EnvironmentHttpServer(("127.0.0.1", 0), store=store, api_token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            after = urllib.parse.quote((observed - timedelta(seconds=1)).isoformat(), safe="")
            url = (
                f"http://127.0.0.1:{server.server_port}/environment/current"
                f"?wait_for=vision.room_light&after={after}&timeout_ms=20"
            )
            request = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertFalse(body["wait_result"]["matched"])
            self.assertEqual(body["wait_result"]["reason"], "timeout")
            self.assertEqual(body["wait_result"]["observed_at"], "")
            self.assertNotIn("room_light", body["vision"])
            self.assertNotIn("vision_snapshot_processor", body["sources"])
            self.assertIsNone(body["observed_at"])
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
            self.assertEqual(
                set(body["relations"]),
                {"ha_request_id", "ha_execution_id", "snapshot_id", "updated_at"},
            )
            self.assertEqual(body["relations"]["ha_request_id"], "ha-42")
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
