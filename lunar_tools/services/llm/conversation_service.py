from __future__ import annotations

from .contracts import LanguageModelPort


class ConversationService:
    def __init__(self, model: LanguageModelPort) -> None:
        self._model = model

    def complete(self, prompt: str) -> str:
        return self._model.generate(prompt)
