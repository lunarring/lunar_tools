from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from .contracts import SpeechToTextPort


class SpeechToTextService:
    """
    Service facade around a speech-to-text adapter.

    The adapter is expected to expose the recorder-style lifecycle used by the
    existing `Speech2Text` implementation while keeping the public surface
    small for callers.
    """

    def __init__(
        self,
        adapter: SpeechToTextPort,
        wait_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._adapter = adapter
        self._wait = wait_fn

    def start(self, output_filename: Optional[str] = None, max_time: Optional[float] = None) -> None:
        """Begin capturing audio via the underlying adapter."""
        self._adapter.start_recording(output_filename=output_filename, max_time=max_time)

    def stop(self, minimum_duration: float = 0.4) -> Optional[str]:
        """
        Stop capturing audio and return a translated transcript when one is available.

        Args:
            minimum_duration: Recordings shorter than this threshold are discarded.
        """
        return self._adapter.stop_recording(minimum_duration=minimum_duration)

    def translate(self, audio_filepath: str) -> str:
        """Translate an existing audio file using the adapter."""
        return self._adapter.translate(audio_filepath)

    def record_and_translate(
        self,
        *,
        output_filename: Optional[str] = None,
        max_time: Optional[float] = None,
        minimum_duration: float = 0.4,
    ) -> Optional[str]:
        """
        Convenience helper that captures audio and returns the transcript.

        When `max_time` is provided, the helper will block using the provided
        wait function to give the adapter time to record before stopping.
        """
        self.start(output_filename=output_filename, max_time=max_time)
        if max_time:
            self._wait(max_time)
        return self.stop(minimum_duration=minimum_duration)

    @contextmanager
    def recording(
        self,
        *,
        output_filename: Optional[str] = None,
        max_time: Optional[float] = None,
    ) -> Iterator[None]:
        """
        Context manager that starts recording on entry and stops on exit.
        """
        self.start(output_filename=output_filename, max_time=max_time)
        try:
            yield
        finally:
            self.stop()


__all__ = ["SpeechToTextService"]
