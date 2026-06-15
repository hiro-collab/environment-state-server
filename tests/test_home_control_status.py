from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from environment_state_server.home_control_status import (
    HomeControlActionStateProbe,
    probe_home_control_action_states,
)


class _FakeHomeControlHandler(BaseHTTPRequestHandler):
    responses: dict[str, dict[str, object]] = {}
    seen_paths: list[str] = []
    seen_auth_headers: list[str] = []

    def do_GET(self) -> None:  # noqa: N802
        self.__class__.seen_paths.append(self.path)
        self.__class__.seen_auth_headers.append(self.headers.get("Authorization", ""))
        action_id = self.path.split("/")[-2] if self.path.endswith("/state") else ""
        body = self.__class__.responses.get(
            action_id,
            {"ok": False, "action_id": action_id, "status": "unavailable"},
        )
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class HomeControlStatusTest(unittest.TestCase):
    def test_probe_prefers_matched_tracked_climate_state(self) -> None:
        _FakeHomeControlHandler.responses = {
            "aircon_cool": {
                "ok": False,
                "action_id": "aircon_cool",
                "status": "mismatch",
                "control_type": "mode_command",
                "state_authority": "ha_entity",
                "verification_mode": "ha_state",
                "state_tracking": "tracked",
                "expected_state": "cool",
                "expected_states": ["cool"],
                "actual_state": "off",
            },
            "aircon_hvac_off": {
                "ok": True,
                "action_id": "aircon_hvac_off",
                "status": "matched",
                "control_type": "mode_command",
                "state_authority": "ha_entity",
                "verification_mode": "ha_state",
                "state_tracking": "tracked",
                "expected_state": "off",
                "expected_states": ["off"],
                "actual_state": "off",
            },
        }
        _FakeHomeControlHandler.seen_paths = []
        _FakeHomeControlHandler.seen_auth_headers = []

        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHomeControlHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = probe_home_control_action_states(
                HomeControlActionStateProbe(
                    action_ids=("aircon_cool", "aircon_hvac_off"),
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    api_token="test-token",
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        assert result is not None
        self.assertEqual(result["action_id"], "aircon_hvac_off")
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["actual_state"], "off")
        self.assertIn("/actions/aircon_cool/state", _FakeHomeControlHandler.seen_paths)
        self.assertIn("/actions/aircon_hvac_off/state", _FakeHomeControlHandler.seen_paths)
        self.assertEqual(
            _FakeHomeControlHandler.seen_auth_headers,
            ["Bearer test-token", "Bearer test-token"],
        )


if __name__ == "__main__":
    unittest.main()
