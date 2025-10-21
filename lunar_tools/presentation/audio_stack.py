from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from lunar_tools.audio import AudioServices, create_audio_services
from lunar_tools.platform.logging import create_logger
from lunar_tools.services.audio.tts_service import TextToSpeechService


@dataclass
class AudioStackConfig:
    """
    Configuration for bootstrapping the audio service stack.

    Attributes:
        include_elevenlabs: When True, construct the ElevenLabs adapter/service.
        include_realtime_transcription: When True, initialise the realtime transcription adapter.
        preferred_tts: Preferred synthesiser (`"openai"` or `"elevenlabs"` when available).
        enable_playback: Attach a playback adapter to TTS services when optional
            dependencies are installed.
        blocking_playback: Configure playback to block until audio completes when
            the adapter supports it.
        playback_kwargs: Default keyword arguments forwarded to playback adapters.
    """

    include_elevenlabs: bool = True
    include_realtime_transcription: bool = True
    preferred_tts: str = "openai"
    enable_playback: bool = False
    blocking_playback: bool = False
    playback_kwargs: dict[str, Any] = field(default_factory=dict)


def _attach_playback_adapter(
    services: AudioServices,
    *,
    enable_playback: bool,
    blocking_playback: bool,
    logger_name: str,
) -> None:
    if not enable_playback:
        return

    try:
        from lunar_tools.adapters.audio.simpleaudio_player import SoundPlayer
    except ImportError:
        create_logger(logger_name).warning(
            "Audio playback requested but optional dependencies are missing. "
            "Install lunar-tools[audio] to enable playback."
        )
        return

    playback_adapter = SoundPlayer(blocking_playback=blocking_playback)
    services.openai_tts.set_playback(playback_adapter)
    if services.elevenlabs_tts:
        services.elevenlabs_tts.set_playback(playback_adapter)


def bootstrap_audio_stack(config: Optional[AudioStackConfig] = None) -> tuple[AudioServices, TextToSpeechService]:
    """
    Build the default audio service bundle and select the preferred TTS service.

    Returns both the `AudioServices` container and the synthesiser picked via the
    configuration. Consumers can inject the audio services elsewhere while using
    the returned synthesiser as their primary voice.
    """
    config = config or AudioStackConfig()

    services = create_audio_services(
        include_elevenlabs=config.include_elevenlabs,
        include_realtime_transcription=config.include_realtime_transcription,
    )

    logger_name = __name__ + ".bootstrap"
    _attach_playback_adapter(
        services,
        enable_playback=config.enable_playback,
        blocking_playback=config.blocking_playback,
        logger_name=logger_name,
    )

    preferred = config.preferred_tts.lower()
    if preferred == "elevenlabs" and config.include_elevenlabs:
        synthesiser = services.elevenlabs_tts
    else:
        synthesiser = services.openai_tts

    return services, synthesiser


class AudioConversationController:
    """
    Small orchestration helper that wires together recorder, transcription, and TTS services.

    Intended as a presentation-layer utility to keep scripts clean while Phase B
    stabilises. It leans on the new service abstractions so tests can substitute
    fake adapters without touching device integrations.
    """

    def __init__(
        self,
        services: AudioServices,
        *,
        synthesiser: Optional[TextToSpeechService] = None,
        logger=None,
    ) -> None:
        self._services = services
        self._tts = synthesiser or services.openai_tts
        self._logger = logger if logger else create_logger(__name__ + ".controller")

    @property
    def services(self) -> AudioServices:
        return self._services

    def capture_transcript(
        self,
        *,
        output_filename: Optional[str] = None,
        max_time: Optional[float] = None,
        minimum_duration: float = 0.4,
    ) -> Optional[str]:
        """
        Capture microphone input and return the resulting transcript when available.
        """
        transcript = self._services.speech_to_text.record_and_translate(
            output_filename=output_filename,
            max_time=max_time,
            minimum_duration=minimum_duration,
        )
        if transcript:
            self._logger.info("Captured transcript: %s", transcript)
        return transcript

    def respond(
        self,
        text: str,
        *,
        output_filename: Optional[str] = None,
        play: bool = False,
        playback_kwargs: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generate speech for `text`, optionally playing it via the configured playback adapter.
        """
        playback_kwargs = playback_kwargs or {}
        result = self._tts.synthesize(
            text,
            output_filename=output_filename,
            play=play,
            playback_kwargs=playback_kwargs,
        )
        self._logger.info("Generated speech at %s", result)
        return result

    def speak(
        self,
        text: str,
        *,
        output_filename: Optional[str] = None,
        playback_kwargs: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Convenience wrapper that always plays back the generated audio.
        """
        return self.respond(
            text,
            output_filename=output_filename,
            play=True,
            playback_kwargs=playback_kwargs,
        )

    def stop_playback(self) -> None:
        """
        Stop any active playback on the underlying TTS service.
        """
        self._tts.stop()


__all__ = [
    "AudioConversationController",
    "AudioStackConfig",
    "bootstrap_audio_stack",
]
