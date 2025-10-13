from __future__ import annotations

from typing import Optional

from .contracts import TranscriptionPort


class TranscriptionService:
    def __init__(self, adapter: TranscriptionPort) -> None:
        self._adapter = adapter

    def start(self) -> None:
        self._adapter.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._adapter.stop(timeout)

    def transcript(self) -> str:
        return self._adapter.get_text()
