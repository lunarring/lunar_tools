from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

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


class VisionProviderRegistry:
    """
    Registry mapping provider names (and aliases) to image services.

    Consumers can query the registry without importing adapters directly,
    mirroring the approach used in the audio and communications stacks.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, ImageService] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        service: ImageService,
        *,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        canonical = name.lower()
        self._providers[canonical] = service
        for alias in aliases or ():
            self._aliases[alias.lower()] = canonical

    def get(self, name: str) -> ImageService:
        key = name.lower()
        canonical = self._aliases.get(key, key)
        try:
            return self._providers[canonical]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown image provider '{name}'. Available: {', '.join(self.available())}") from exc

    def available(self) -> list[str]:
        return sorted(self._providers.keys())


__all__ = ["ImageGenerationResult", "ImageService", "VisionProviderRegistry"]
