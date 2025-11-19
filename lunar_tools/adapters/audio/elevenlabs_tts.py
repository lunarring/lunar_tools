from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from lunar_tools.platform.config import read_api_key
from lunar_tools.platform.logging import create_logger
from lunar_tools.services.audio.contracts import SoundPlaybackPort
from lunar_tools._optional import require_extra

try:  # pragma: no cover - optional ElevenLabs dependency guard
    from elevenlabs import Voice, VoiceSettings, save
    from elevenlabs.client import ElevenLabs
except ImportError:  # pragma: no cover
    require_extra("Text2SpeechElevenlabs", extras="audio")


class Text2SpeechElevenlabs:
    """
    Adapter for ElevenLabs text-to-speech.
    """

    def __init__(
        self,
        logger=None,
        sound_player: Optional[SoundPlaybackPort] = None,
        voice_id: Optional[str] = None,
        blocking_playback: bool = False,
    ) -> None:
        self.client = ElevenLabs(api_key=read_api_key("ELEVEN_API_KEY"))
        self.logger = logger if logger else create_logger(__name__)
        self.sound_player = sound_player
        self.output_filename: Optional[str] = None
        self.default_voice_id = "EXAVITQu4vr4xnSDxMaL"
        self.voice_id = voice_id or self.default_voice_id
        self.blocking_playback = blocking_playback

    def _resolve_output_path(self, output_filename: Optional[str]) -> str:
        if output_filename:
            return output_filename
        temp_file = tempfile.NamedTemporaryFile(prefix="lunar_eleven_tts_", suffix=".mp3", delete=False)
        temp_file.close()
        return temp_file.name

    def play(
        self,
        text: str,
        output_filename: Optional[str] = None,
        stability: float = 0.71,
        similarity_boost: float = 0.5,
        style: float = 0.0,
        use_speaker_boost: bool = True,
    ) -> None:
        output = self.generate(text, output_filename, self.voice_id, stability, similarity_boost, style, use_speaker_boost)
        if self.sound_player is None:
            try:
                from lunar_tools.adapters.audio.simpleaudio_player import SoundPlayer
            except ImportError:  # pragma: no cover
                require_extra("SoundPlayer", extras="audio")
            self.sound_player = SoundPlayer(blocking_playback=self.blocking_playback)
        self.sound_player.play(output)

    def change_voice(self, voice_id: str) -> None:
        self.voice_id = voice_id
        self.logger.info("Voice model changed to %s", voice_id)

    def stop(self) -> None:
        if self.sound_player:
            self.sound_player.stop()

    def generate(
        self,
        text: str,
        output_filename: Optional[str] = None,
        voice_id: Optional[str] = None,
        stability: float = 0.71,
        similarity_boost: float = 0.5,
        style: float = 0.0,
        use_speaker_boost: bool = True,
    ) -> str:
        if not text:
            raise ValueError("text is invalid!")

        voice_to_use = voice_id or self.default_voice_id
        audio = self.client.generate(
            text=text,
            voice=Voice(
                voice_id=voice_to_use,
                settings=VoiceSettings(
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=style,
                    use_speaker_boost=use_speaker_boost,
                ),
            ),
        )

        self.output_filename = self._resolve_output_path(output_filename)
        save(audio, self.output_filename)
        self.logger.info("Generated speech saved to %s", Path(self.output_filename).resolve())
        return self.output_filename
