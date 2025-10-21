from __future__ import annotations

from typing import Protocol


class ImageGeneratorPort(Protocol):
    def generate(self, prompt: str, *args, **kwargs):
        ...


__all__ = ["ImageGeneratorPort"]
