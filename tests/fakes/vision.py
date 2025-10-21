from __future__ import annotations

from typing import Any, Dict, Tuple


class FakeImageGeneratorPort:
    def __init__(self, image: Any = None, metadata: Dict[str, Any] | None = None) -> None:
        self.image = image or "fake-image"
        self.metadata = metadata or {}
        self.prompts: list[Tuple[str, tuple, dict]] = []

    def generate(self, prompt: str, *args, **kwargs):
        self.prompts.append((prompt, args, kwargs))
        return self.image, self.metadata
