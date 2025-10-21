from __future__ import annotations

import threading
from typing import Optional

from lunar_tools._optional import require_extra

try:  # pragma: no cover - optional dependency guard
    import simpleaudio
    from pydub import AudioSegment
except ImportError:  # pragma: no cover - import side effect
    require_extra("SoundPlayer", extras="audio")


class SoundPlayer:
    """
    Helper around simpleaudio for blocking / non-blocking playback with optional panning.
    """

    def __init__(self, blocking_playback: bool = False) -> None:
        self._play_thread: Optional[threading.Thread] = None
        self._playback_object = None
        self.blocking_playback = blocking_playback

    def _play_sound_threaded(self, sound: AudioSegment) -> None:
        self._playback_object = simpleaudio.play_buffer(
            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate,
        )
        self._playback_object.wait_done()

    def play_sound(self, file_path: str, pan_value: float = 0) -> None:
        self.stop_sound()

        sound = AudioSegment.from_file(file_path)
        if -1 < pan_value < 1 and pan_value != 0:
            sound = sound.pan(pan_value)

        self._play_thread = threading.Thread(target=self._play_sound_threaded, args=(sound,))
        self._play_thread.start()
        if self.blocking_playback:
            self._play_thread.join()

    def stop_sound(self) -> None:
        if self._play_thread and self._play_thread.is_alive():
            if self._playback_object:
                self._playback_object.stop()
            self._play_thread.join()
            self._play_thread = None

    # SoundPlaybackPort compatibility -------------------------------------------------
    def play(self, file_path: str, **kwargs) -> None:
        """
        Adapter-style entry point conforming to SoundPlaybackPort.
        """
        pan_value = float(kwargs.get("pan", 0))
        self.play_sound(file_path, pan_value=pan_value)

    def stop(self) -> None:
        self.stop_sound()
