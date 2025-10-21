from __future__ import annotations

import os
from typing import Optional

from lunar_tools.adapters.audio.sounddevice_recorder import SoundDeviceRecorder
from lunar_tools.platform.config import read_api_key
from lunar_tools.platform.logging import create_logger
from lunar_tools._optional import require_extra

try:  # pragma: no cover - optional audio stack dependency guard
    import numpy as np
    from openai import OpenAI
    from pydub import AudioSegment
except ImportError:  # pragma: no cover - import side effect
    require_extra("Speech2Text", extras="audio")


class Speech2Text:
    """
    Adapter providing speech-to-text capabilities via OpenAI (or optional Whisper offline model).
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        logger=None,
        audio_recorder: Optional[SoundDeviceRecorder] = None,
        offline_model_type: Optional[str] = None,
    ) -> None:
        self.transcript: Optional[str] = None

        if client is None:
            api_key = read_api_key("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client

        if offline_model_type is not None:
            try:
                import whisper
            except ImportError as exc:  # pragma: no cover - offline dependency
                require_extra("Speech2Text offline mode", extras="audio")

            self.whisper_model = whisper.load_model(offline_model_type)
            self.offline_mode = True
        else:
            self.offline_mode = False

        self.logger = logger if logger else create_logger(__name__)
        self.audio_recorder = audio_recorder or SoundDeviceRecorder(logger=logger)

    def start_recording(self, output_filename: Optional[str] = None, max_time: Optional[float] = None) -> None:
        if self.audio_recorder is None:
            raise ValueError("Audio recorder is not available")
        self.audio_recorder.start_recording(output_filename, max_time)

    def stop_recording(self, minimum_duration: float = 0.4) -> Optional[str]:
        if self.audio_recorder is None:
            raise ValueError("Audio recorder is not available")
        self.audio_recorder.stop_recording()

        audio_duration = AudioSegment.from_mp3(self.audio_recorder.output_filename).duration_seconds
        if audio_duration < minimum_duration:
            self.logger.warning(
                "Recording is too short, only %.2f seconds. Minimum required is %.2f seconds.",
                audio_duration,
                minimum_duration,
            )
            return None
        return self.translate(self.audio_recorder.output_filename)

    def translate(self, audio_filepath: str) -> str:
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

        if self.offline_mode:
            audio_segment = AudioSegment.from_file(audio_filepath)
            numpydata = np.array(audio_segment.get_array_of_samples()).astype(np.int16)
            numpydata = np.hstack(numpydata).astype(np.float32)
            numpydata = numpydata.astype(np.float32) / 32768.0
            options = dict(language="english", beam_size=5, best_of=5)
            translate_options = dict(task="translate", **options)
            result = self.whisper_model.transcribe(numpydata, **translate_options)
            return result["text"].strip()

        with open(audio_filepath, "rb") as audio_file:
            transcript = self.client.audio.translations.create(
                model="whisper-1",
                file=audio_file,
            )
            return transcript.text

    def handle_unmute_button(self, mic_button_state: bool) -> bool:
        if mic_button_state:
            if not self.audio_recorder.is_recording:
                self.start_recording()
        else:
            if self.audio_recorder.is_recording:
                try:
                    transcript = self.stop_recording()
                    if transcript:
                        self.transcript = transcript.strip().lower()
                        return True
                except Exception as exc:  # pragma: no cover - logging side effect
                    self.logger.error("Error stopping recording: %s", exc)
        return False
