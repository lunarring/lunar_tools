from __future__ import annotations

from typing import Optional

from lunar_tools.platform.config import read_api_key
from lunar_tools.platform.logging import create_logger

from lunar_tools._optional import require_extra
from lunar_tools.services.audio.contracts import SoundPlaybackPort

try:  # pragma: no cover - optional dependency guard for OpenAI SDK
    from openai import OpenAI
except ImportError:  # pragma: no cover - import side effect
    require_extra("Text2SpeechOpenAI", extras="audio")


class Text2SpeechOpenAI:
    """
    Adapter generating speech using OpenAI text-to-speech models.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        logger=None,
        voice_model: str = "nova",
        sound_player: Optional[SoundPlaybackPort] = None,
        blocking_playback: bool = False,
    ) -> None:
        if client is None:
            api_key = read_api_key("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client

        self.logger = logger if logger else create_logger(__name__)
        self.sound_player = sound_player
        self.output_filename: Optional[str] = None
        self.voice_model = voice_model
        self.blocking_playback = blocking_playback

    def play(self, text: Optional[str] = None) -> None:
        output = self.generate(text)
        if self.sound_player is None:
            try:
                from lunar_tools.adapters.audio.simpleaudio_player import SoundPlayer
            except ImportError:  # pragma: no cover - optional playback dependency
                require_extra("SoundPlayer", extras="audio")
            self.sound_player = SoundPlayer(blocking_playback=self.blocking_playback)
        self.sound_player.play(output)

    def stop(self) -> None:
        if self.sound_player:
            self.sound_player.stop()

    def generate(self, text: Optional[str], output_filename: Optional[str] = None) -> str:
        if not text:
            raise ValueError("text is invalid!")

        response = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice_model,
            input=text,
        )

        self.output_filename = output_filename if output_filename else "output_speech.mp3"
        response.stream_to_file(self.output_filename)
        self.logger.info("Generated speech saved to %s", self.output_filename)
        return self.output_filename

    def change_voice(self, new_voice: str) -> None:
        if new_voice in self.list_available_voices():
            self.voice_model = new_voice
            self.logger.info("Voice model changed to %s", new_voice)
        else:
            raise ValueError(f"Voice '{new_voice}' is not a valid voice model.")

    @staticmethod
    def list_available_voices() -> list[str]:
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
