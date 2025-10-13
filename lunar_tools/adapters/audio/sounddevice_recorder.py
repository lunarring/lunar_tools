from __future__ import annotations

import tempfile
import threading
import time
import wave
from typing import Optional

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

from lunar_tools.platform.logging import create_logger


class SoundDeviceRecorder:
    """
    Concrete recorder backed by the sounddevice library.
    """

    def __init__(
        self,
        channels: int = 1,
        rate: int = 44100,
        chunk: int = 1024,
        logger=None,
    ) -> None:
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.frames: list[np.ndarray] = []
        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None
        self.output_filename: Optional[str] = None
        self.logger = logger if logger else create_logger(__name__)
        self.thread: Optional[threading.Thread] = None

    def start_recording(self, output_filename: Optional[str] = None, max_time: Optional[float] = None) -> None:
        if self.is_recording:
            return

        self.is_recording = True
        if output_filename is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            self.output_filename = temp_file.name
            temp_file.close()
        else:
            output_filename = str(output_filename)
            if not output_filename.endswith(".mp3"):
                raise ValueError("Output filename must have a .mp3 extension")
            self.output_filename = output_filename

        self.thread = threading.Thread(target=self._record, args=(max_time,))
        self.thread.start()

    def _record(self, max_time: Optional[float]) -> None:
        self.stream = sd.InputStream(
            samplerate=self.rate,
            channels=self.channels,
            blocksize=self.chunk,
            dtype="float32",
        )
        self.logger.info("Recording...")
        self.frames = []
        start_time = time.time()
        with self.stream:
            while self.is_recording:
                if max_time and (time.time() - start_time) >= max_time:
                    break
                data, _overflowed = self.stream.read(self.chunk)
                self.frames.append(data.flatten())

        self.logger.info("Finished recording.")
        self._flush_to_file()

    def _flush_to_file(self) -> None:
        if not self.output_filename:
            return

        wav_filename = tempfile.mktemp(suffix=".wav")
        wf = wave.open(wav_filename, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)
        wf.setframerate(self.rate)
        self.frames = np.clip(self.frames, -1, +1)
        wf.writeframes(np.array(self.frames * 32767).astype(np.int16).tobytes())
        wf.close()

        wav_audio = AudioSegment.from_wav(wav_filename)
        wav_audio.export(self.output_filename, format="mp3")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return

        self.is_recording = False
        if self.thread:
            self.thread.join()
            self.thread = None
