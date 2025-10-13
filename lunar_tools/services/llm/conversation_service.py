from __future__ import annotations

from typing import Protocol


class LanguageModelPort(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class ConversationService:
    def __init__(self, model: LanguageModelPort) -> None:
        self._model = model

    def complete(self, prompt: str) -> str:
        return self._model.generate(prompt)
