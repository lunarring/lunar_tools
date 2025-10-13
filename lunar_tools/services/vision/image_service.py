from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple


class ImageGeneratorPort(Protocol):
    def generate(self, prompt: str, *args, **kwargs):
        ...


@dataclass
class ImageGenerationResult:
    image: Any
    metadata: dict


class ImageService:
    def __init__(self, generator: ImageGeneratorPort) -> None:
        self._generator = generator

    def generate(self, prompt: str, *args, **kwargs) -> ImageGenerationResult:
        result = self._generator.generate(prompt, *args, **kwargs)
        if isinstance(result, tuple):
            image, metadata = result
        else:
            image, metadata = result, {}
        return ImageGenerationResult(image=image, metadata={"raw": metadata})
