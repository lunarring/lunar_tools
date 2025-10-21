from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .contracts import ImageGeneratorPort


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
