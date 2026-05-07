from __future__ import annotations

import argparse
import functools
import os
import signal
import sys
import threading
from pathlib import Path

from .camera_hub import CameraHubSubscriber
from .feedback import StateQueryFeedbackStore
from .ha_events import HomeAssistantEventTailer
from .http_api import EnvironmentHttpServer
from .module_polling import ModuleProbe, ModuleStatusPoller
from .state import EnvironmentStateStore


def parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("--port must be between 1 and 65535")
    return port


def parse_positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local environment state server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=parse_port, default=8790)
    parser.add_argument("--ttl-ms", type=lambda value: parse_positive_int(value, name="--ttl-ms"), default=5000)
    parser.add_argument("--api-token-env", default="ENVIRONMENT_API_TOKEN")
    parser.add_argument("--fallback-api-token-env", default="HOME_CONTROL_API_TOKEN")
    parser.add_argument(
        "--ha-events-path",
        default=str(Path("..") / "home-assistant-server" / ".cache" / "home_control" / "events.jsonl"),
    )
    parser.add_argument(
        "--state-query-feedback-path",
        default=str(Path(".cache") / "environment_state_server" / "state_query_feedback.jsonl"),
        help="Append-only JSONL path for Dify state-query correction feedback.",
    )
    parser.add_argument("--camera-hub-url", default="ws://127.0.0.1:8765")
    parser.add_argument(
        "--vision-topic-url",
        action="append",
        default=[],
        help=(
            "Additional WebSocket topic source to ingest as vision snapshots, "
            "for example a vision snapshot processor."
        ),
    )
    parser.add_argument("--module-poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--home-assistant-health-url", default="http://127.0.0.1:8787/health")
    parser.add_argument("--aituber-url", default="http://127.0.0.1:3000")
    parser.add_argument("--dify-url", default="http://127.0.0.1:8080")
    parser.add_argument("--voicevox-health-url", default="http://127.0.0.1:50021/version")
    parser.add_argument("--disable-ha-events", action="store_true")
    parser.add_argument("--disable-camera-hub", action="store_true")
    parser.add_argument("--disable-module-polling", action="store_true")
    return parser


def resolve_api_token(primary_env: str, fallback_env: str | None) -> str:
    for name in (primary_env, fallback_env):
        if not name:
            continue
        token = os.environ.get(name, "").strip()
        if token:
            return token
    raise RuntimeError(
        f"API token is required. Set {primary_env}"
        + (f" or {fallback_env}" if fallback_env else "")
        + "."
    )


def run(args: argparse.Namespace) -> None:
    token = resolve_api_token(args.api_token_env, args.fallback_api_token_env)
    store = EnvironmentStateStore(ttl_ms=args.ttl_ms)
    feedback_store = StateQueryFeedbackStore(args.state_query_feedback_path)
    stop_event = threading.Event()
    workers: list[object] = []

    if not args.disable_ha_events:
        ha_tailer = HomeAssistantEventTailer(args.ha_events_path, store.ingest_home_assistant_event)
        ha_tailer.start()
        workers.append(ha_tailer)
        print(f"home assistant events: {Path(args.ha_events_path)}", flush=True)

    if not args.disable_camera_hub:
        camera_subscriber = CameraHubSubscriber(args.camera_hub_url, store.ingest_camera_hub_envelope)
        camera_subscriber.start()
        workers.append(camera_subscriber)
        print(f"camera hub: {args.camera_hub_url}", flush=True)

    for index, url in enumerate(args.vision_topic_url or [], start=1):
        source = "vision_snapshot_processor" if index == 1 else f"vision_topic_{index}"
        subscriber = CameraHubSubscriber(
            url,
            functools.partial(store.ingest_vision_envelope, source=source),
        )
        subscriber.start()
        workers.append(subscriber)
        print(f"vision topic {source}: {url}", flush=True)

    if not args.disable_module_polling:
        probes = _build_module_probes(args)
        if probes:
            module_poller = ModuleStatusPoller(
                probes,
                store.ingest_node_status,
                interval_seconds=args.module_poll_interval_seconds,
            )
            module_poller.start()
            workers.append(module_poller)
            print(f"module polling: {len(probes)} probes", flush=True)

    server = EnvironmentHttpServer(
        (args.host, args.port),
        store=store,
        api_token=token,
        feedback_store=feedback_store,
    )

    def stop(_signum: int, _frame: object) -> None:
        stop_event.set()
        server.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"environment state server listening on http://{args.host}:{args.port}", flush=True)
    print(f"state query feedback: {Path(args.state_query_feedback_path)}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        for worker in workers:
            stop_method = getattr(worker, "stop", None)
            if callable(stop_method):
                stop_method()


def _build_module_probes(args: argparse.Namespace) -> list[ModuleProbe]:
    probes: list[ModuleProbe] = []
    for node_id, url in (
        ("home_assistant_bridge", args.home_assistant_health_url),
        ("aituber_kit", args.aituber_url),
        ("dify", args.dify_url),
        ("voicevox", args.voicevox_health_url),
    ):
        if isinstance(url, str) and url.strip():
            probes.append(ModuleProbe(node_id=node_id, url=url.strip(), ttl_ms=args.ttl_ms))
    return probes


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
