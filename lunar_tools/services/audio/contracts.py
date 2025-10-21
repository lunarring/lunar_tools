from __future__ import annotations

from typing import Protocol, Optional


class RecorderPort(Protocol):
    output_filename: Optional[str]
    is_recording: bool

    def start_recording(self, output_filename: Optional[str] = None, max_time: Optional[float] = None) -> None:
        ...

    def stop_recording(self) -> None:
        ...


class SpeechSynthesisPort(Protocol):
    def generate(self, text: str, output_filename: Optional[str] = None) -> str:
        ...


class SpeechToTextPort(Protocol):
    def start_recording(self, output_filename: Optional[str] = None, max_time: Optional[float] = None) -> None:
        ...

    def stop_recording(self, minimum_duration: float = 0.4) -> Optional[str]:
        ...

    def translate(self, audio_filepath: str) -> str:
        ...


class SoundPlaybackPort(Protocol):
    def play(self, file_path: str, **kwargs) -> None:
        ...

    def stop(self) -> None:
        ...


class TranscriptionPort(Protocol):
    def start(self) -> None:
        ...

    def stop(self, timeout: Optional[float] = None) -> None:
        ...

    def get_text(self) -> str:
        ...
