from __future__ import annotations

from typing import Protocol


class LanguageModelPort(Protocol):
    def generate(self, prompt: str) -> str:
        ...


__all__ = ["LanguageModelPort"]
