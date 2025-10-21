from __future__ import annotations

from typing import Optional

from .contracts import SpeechSynthesisPort, SoundPlaybackPort


class TextToSpeechService:
    def __init__(
        self,
        synthesizer: SpeechSynthesisPort,
        playback: Optional[SoundPlaybackPort] = None,
    ) -> None:
        self._synthesizer = synthesizer
        self._playback = playback

    def synthesize(
        self,
        text: str,
        output_filename: Optional[str] = None,
        *,
        play: bool = False,
        playback_kwargs: Optional[dict] = None,
    ) -> str:
        final_path = self._synthesizer.generate(text, output_filename)
        if play and self._playback:
            self._playback.play(final_path, **(playback_kwargs or {}))
        return final_path

    def play(
        self,
        text: str,
        output_filename: Optional[str] = None,
        **playback_kwargs,
    ) -> str:
        final_path = self._synthesizer.generate(text, output_filename)
        if self._playback:
            self._playback.play(final_path, **playback_kwargs)
        return final_path

    def stop(self) -> None:
        if self._playback:
            self._playback.stop()

    def set_playback(self, playback: Optional[SoundPlaybackPort]) -> None:
        self._playback = playback
