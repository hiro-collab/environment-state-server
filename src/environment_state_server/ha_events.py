from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable


class HomeAssistantEventTailer:
    def __init__(
        self,
        path: str | Path,
        on_event: Callable[[dict], None],
        *,
        poll_interval: float = 0.5,
        initial_lines: int = 100,
    ) -> None:
        self.path = Path(path)
        self.on_event = on_event
        self.poll_interval = float(poll_interval)
        self.initial_lines = int(initial_lines)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ha-event-tailer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        position = 0
        loaded_initial = False
        while not self._stop_event.is_set():
            if not self.path.exists():
                time.sleep(self.poll_interval)
                continue
            try:
                if not loaded_initial:
                    position = self._load_initial()
                    loaded_initial = True
                position = self._read_new_lines(position)
            except OSError:
                loaded_initial = False
                position = 0
            time.sleep(self.poll_interval)

    def _load_initial(self) -> int:
        with self.path.open("r", encoding="utf-8") as handle:
            lines = deque(handle.readlines(), maxlen=self.initial_lines)
            position = handle.tell()
        for line in lines:
            self._emit_line(line)
        return position

    def _read_new_lines(self, position: int) -> int:
        size = self.path.stat().st_size
        if size < position:
            position = 0
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(position)
            for line in handle:
                self._emit_line(line)
            return handle.tell()

    def _emit_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(event, dict):
            self.on_event(event)
