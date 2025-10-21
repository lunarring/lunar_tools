from __future__ import annotations

from typing import Optional, List, Dict, Any


class FakeRecorderPort:
    def __init__(self) -> None:
        self.output_filename: Optional[str] = None
        self.is_recording: bool = False
        self._recorded_chunks: List[bytes] = []

    def start_recording(self, output_filename: Optional[str] = None, max_time: Optional[float] = None) -> None:  # noqa: D401
        self.output_filename = output_filename or "fake-recording.mp3"
        self.is_recording = True
        self._recorded_chunks.clear()

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        self.is_recording = False
        self._recorded_chunks.append(b"audio-data")

    def chunks(self) -> List[bytes]:
        return list(self._recorded_chunks)


class FakeSpeechSynthesisPort:
    def __init__(self) -> None:
        self.generated: list[tuple[str, Optional[str]]] = []

    def generate(self, text: str, output_filename: Optional[str] = None) -> str:
        target = output_filename or "fake-tts.mp3"
        self.generated.append((text, target))
        return target


class FakeSpeechToTextPort:
    def __init__(self, transcript: str = "hello world") -> None:
        self._transcript = transcript
        self.started: bool = False
        self.stopped: bool = False
        self.output_filename: Optional[str] = None
        self.max_time: Optional[float] = None
        self.minimum_duration: Optional[float] = None
        self.translated_paths: list[str] = []

    def start_recording(self, output_filename: Optional[str] = None, max_time: Optional[float] = None) -> None:
        self.started = True
        self.output_filename = output_filename
        self.max_time = max_time

    def stop_recording(self, minimum_duration: float = 0.4) -> Optional[str]:
        self.stopped = True
        self.minimum_duration = minimum_duration
        return self._transcript

    def translate(self, audio_filepath: str) -> str:
        self.translated_paths.append(audio_filepath)
        return self._transcript


class FakeTranscriptionPort:
    def __init__(self, transcript: str = "") -> None:
        self._transcript = transcript
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: Optional[float] = None) -> None:
        self.stopped = True

    def get_text(self) -> str:
        return self._transcript


class FakeSoundPlaybackPort:
    def __init__(self) -> None:
        self.play_calls: list[tuple[str, Dict[str, Any]]] = []
        self.stopped = False

    def play(self, file_path: str, **kwargs) -> None:
        self.play_calls.append((file_path, dict(kwargs)))

    def stop(self) -> None:
        self.stopped = True
