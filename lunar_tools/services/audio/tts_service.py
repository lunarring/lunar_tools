from __future__ import annotations

from typing import Optional

from .contracts import SpeechSynthesisPort


class TextToSpeechService:
    def __init__(self, synthesizer: SpeechSynthesisPort) -> None:
        self._synthesizer = synthesizer

    def synthesize(self, text: str, output_filename: Optional[str] = None) -> str:
        return self._synthesizer.generate(text, output_filename)
