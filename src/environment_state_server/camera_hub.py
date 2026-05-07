from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time
from typing import Callable
from urllib.parse import urlparse


class CameraHubSubscriber:
    def __init__(
        self,
        url: str,
        on_envelope: Callable[[dict], None],
        *,
        reconnect_seconds: float = 2.0,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.url = url
        self.on_envelope = on_envelope
        self.reconnect_seconds = float(reconnect_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-hub-subscriber", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._consume_once()
            except OSError:
                pass
            except ValueError:
                pass
            self._stop_event.wait(self.reconnect_seconds)

    def _consume_once(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError("camera hub url must use ws:// or wss://")
        if parsed.scheme == "wss":
            raise ValueError("wss camera hub urls are not supported by the lightweight client")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        with socket.create_connection((host, port), timeout=self.timeout_seconds) as sock:
            sock.settimeout(self.timeout_seconds)
            _websocket_handshake(sock, host=host, port=port, path=path)
            while not self._stop_event.is_set():
                opcode, payload = _read_frame(sock)
                if opcode == 0x1:
                    self._handle_text(payload.decode("utf-8", errors="replace"))
                elif opcode == 0x8:
                    return
                elif opcode == 0x9:
                    _write_frame(sock, 0xA, payload)

    def _handle_text(self, text: str) -> None:
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(message, dict):
            self.on_envelope(message)


def _websocket_handshake(sock: socket.socket, *, host: str, port: int, path: str) -> None:
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = "\r\n".join(
        [
            f"GET {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            "",
            "",
        ]
    )
    sock.sendall(request.encode("ascii"))
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise OSError("connection closed during websocket handshake")
        response += chunk
        if len(response) > 16384:
            raise OSError("websocket handshake response too large")
    status = response.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
    if " 101 " not in status:
        raise OSError(f"websocket handshake failed: {status}")


def _read_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise OSError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(sock: socket.socket) -> tuple[int, bytes]:
    first, second = _read_exact(sock, 2)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _read_exact(sock, 8))[0]
    mask = _read_exact(sock, 4) if masked else b""
    payload = _read_exact(sock, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return opcode, payload


def _write_frame(sock: socket.socket, opcode: int, payload: bytes) -> None:
    mask = os.urandom(4)
    header = bytearray([0x80 | (opcode & 0x0F)])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length <= 0xFFFF:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked)

