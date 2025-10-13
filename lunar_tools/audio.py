"""High-level audio entry points for lunar_tools.

This module re-exports concrete adapters for backwards compatibility while
also providing service-layer factories that compose those adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lunar_tools._optional import optional_import_attr
from lunar_tools.platform.logging import create_logger
from lunar_tools.services.audio.recorder_service import RecorderService
from lunar_tools.services.audio.transcription_service import TranscriptionService
from lunar_tools.services.audio.tts_service import TextToSpeechService

Speech2Text = optional_import_attr(
    "lunar_tools.adapters.audio.openai_transcribe",
    "Speech2Text",
    feature="Speech2Text",
    extras="audio",
)
Text2SpeechOpenAI = optional_import_attr(
    "lunar_tools.adapters.audio.openai_tts",
    "Text2SpeechOpenAI",
    feature="Text2SpeechOpenAI",
    extras="audio",
)
Text2SpeechElevenlabs = optional_import_attr(
    "lunar_tools.adapters.audio.elevenlabs_tts",
    "Text2SpeechElevenlabs",
    feature="Text2SpeechElevenlabs",
    extras="audio",
)
SoundPlayer = optional_import_attr(
    "lunar_tools.adapters.audio.simpleaudio_player",
    "SoundPlayer",
    feature="SoundPlayer",
    extras="audio",
)
SoundDeviceRecorder = optional_import_attr(
    "lunar_tools.adapters.audio.sounddevice_recorder",
    "SoundDeviceRecorder",
    feature="SoundDeviceRecorder",
    extras="audio",
)
RealTimeTranscribe = optional_import_attr(
    "lunar_tools.adapters.audio.deepgram_transcribe",
    "RealTimeTranscribe",
    feature="RealTimeTranscribe",
    extras="audio",
)

# Backwards-compatible aliases
AudioRecorder = SoundDeviceRecorder


@dataclass
class AudioServices:
    """Bundle of audio-related services with their underlying adapters."""

    recorder_service: RecorderService
    speech_to_text: Speech2Text
    openai_tts: TextToSpeechService
    elevenlabs_tts: TextToSpeechService
    realtime_transcription: Optional[TranscriptionService]


def create_audio_services(
    *,
    include_elevenlabs: bool = True,
    include_realtime_transcription: bool = True,
) -> AudioServices:
    """Construct the default audio service stack.

    Args:
        include_elevenlabs: When False, the ElevenLabs adapter will be skipped.
        include_realtime_transcription: When False, Deepgram real-time
            transcription is not initialised.
    """

    recorder_adapter = SoundDeviceRecorder(logger=create_logger(__name__ + ".Recorder"))
    recorder_service = RecorderService(recorder_adapter)

    speech_to_text_adapter = Speech2Text(logger=create_logger(__name__ + ".Speech2Text"))

    openai_tts_adapter = Text2SpeechOpenAI(logger=create_logger(__name__ + ".TTS.OpenAI"))
    openai_tts_service = TextToSpeechService(openai_tts_adapter)

    if include_elevenlabs:
        elevenlabs_adapter = Text2SpeechElevenlabs(logger=create_logger(__name__ + ".TTS.ElevenLabs"))
        elevenlabs_service = TextToSpeechService(elevenlabs_adapter)
    else:
        elevenlabs_adapter = None
        elevenlabs_service = TextToSpeechService(Text2SpeechOpenAI(logger=create_logger(__name__ + ".TTS.OpenAI.Fallback")))

    realtime_service: Optional[TranscriptionService]
    if include_realtime_transcription:
        try:
            realtime_adapter = RealTimeTranscribe(logger=create_logger(__name__ + ".RealtimeTranscribe"))
            realtime_service = TranscriptionService(realtime_adapter)
        except ImportError:
            realtime_service = None
    else:
        realtime_service = None

    return AudioServices(
        recorder_service=recorder_service,
        speech_to_text=speech_to_text_adapter,
        openai_tts=openai_tts_service,
        elevenlabs_tts=elevenlabs_service,
        realtime_transcription=realtime_service,
    )


__all__ = [
    "AudioRecorder",
    "Speech2Text",
    "Text2SpeechOpenAI",
    "Text2SpeechElevenlabs",
    "SoundPlayer",
    "RealTimeTranscribe",
    "AudioServices",
    "create_audio_services",
]
