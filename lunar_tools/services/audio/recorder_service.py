from __future__ import annotations

from typing import Optional

from .contracts import RecorderPort


class RecorderService:
    """
    Facade around a recorder adapter to provide a stable API.
    """

    def __init__(self, recorder: RecorderPort) -> None:
        self._recorder = recorder

    @property
    def output_filename(self) -> Optional[str]:
        return getattr(self._recorder, "output_filename", None)

    @property
    def is_recording(self) -> bool:
        return getattr(self._recorder, "is_recording", False)

    def start(self, output_filename: Optional[str] = None, max_time: Optional[float] = None) -> None:
        self._recorder.start_recording(output_filename, max_time)

    def stop(self) -> None:
        self._recorder.stop_recording()
