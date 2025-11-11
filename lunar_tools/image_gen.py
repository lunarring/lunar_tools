"""Vision generation entry points for lunar_tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lunar_tools._optional import optional_import_attr
from lunar_tools.services.vision.image_service import ImageService, VisionProviderRegistry

__all__ = [
    "FluxImageGenerator",
    "NanoBananaEditImageGenerator",
    "FluxKontextImageGenerator",
    "Dalle3ImageGenerator",
    "SDXL_LCM",
    "SDXL_TURBO",
    "GlifAPI",
    "ImageGenerators",
    "VisionProviderRegistry",
    "create_image_generators",
]

_OPTIONAL_EXPORTS = {
    "FluxImageGenerator": ("lunar_tools.adapters.vision.fal_flux", "FluxImageGenerator"),
    "NanoBananaEditImageGenerator": ("lunar_tools.adapters.vision.fal_flux", "NanoBananaEditImageGenerator"),
    "FluxKontextImageGenerator": ("lunar_tools.adapters.vision.fal_flux", "FluxKontextImageGenerator"),
    "Dalle3ImageGenerator": ("lunar_tools.adapters.vision.openai_dalle", "Dalle3ImageGenerator"),
    "SDXL_LCM": ("lunar_tools.adapters.vision.replicate_sdxl", "SDXL_LCM"),
    "SDXL_TURBO": ("lunar_tools.adapters.vision.replicate_sdxl", "SDXL_TURBO"),
    "GlifAPI": ("lunar_tools.adapters.vision.glif_api", "GlifAPI"),
}


def _load(name: str):
    module, attribute = _OPTIONAL_EXPORTS[name]
    return optional_import_attr(
        module,
        attribute,
        feature=name,
        extras="imaging",
    )


def __getattr__(name: str):
    if name in _OPTIONAL_EXPORTS:
        value = _load(name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class ImageGenerators:
    dalle3: ImageService
    sdxl_lcm: ImageService
    sdxl_turbo: ImageService
    flux: "FluxImageGenerator"
    nano_banana: "NanoBananaEditImageGenerator"
    flux_kontext: "FluxKontextImageGenerator"
    glif_api: Optional["GlifAPI"]
    registry: VisionProviderRegistry

    def get(self, name: str) -> ImageService:
        return self.registry.get(name)


def create_image_generators(*, include_glif: bool = True) -> ImageGenerators:
    dalle_adapter = _load("Dalle3ImageGenerator")()
    dalle_service = ImageService(dalle_adapter)

    sdxl_lcm_adapter = _load("SDXL_LCM")()
    sdxl_lcm_service = ImageService(sdxl_lcm_adapter)

    sdxl_turbo_adapter = _load("SDXL_TURBO")()
    sdxl_turbo_service = ImageService(sdxl_turbo_adapter)

    flux_adapter = _load("FluxImageGenerator")()
    nano_banana_adapter = _load("NanoBananaEditImageGenerator")()
    flux_kontext_adapter = _load("FluxKontextImageGenerator")()

    glif_api_adapter = _load("GlifAPI")() if include_glif else None
    flux_service = ImageService(flux_adapter)
    nano_banana_service = ImageService(nano_banana_adapter)
    flux_kontext_service = ImageService(flux_kontext_adapter)

    registry = VisionProviderRegistry()
    registry.register("dalle3", dalle_service, aliases=("openai", "dalle"))
    registry.register("sdxl_lcm", sdxl_lcm_service, aliases=("sdxl", "replicate_lcm"))
    registry.register("sdxl_turbo", sdxl_turbo_service, aliases=("sdxl_fast",))
    registry.register("flux", flux_service)
    registry.register("nano_banana", nano_banana_service, aliases=("nano",))
    registry.register("flux_kontext", flux_kontext_service, aliases=("kontext",))
    if glif_api_adapter:
        registry.register("glif", ImageService(glif_api_adapter))

    return ImageGenerators(
        dalle3=dalle_service,
        sdxl_lcm=sdxl_lcm_service,
        sdxl_turbo=sdxl_turbo_service,
        flux=flux_adapter,
        nano_banana=nano_banana_adapter,
        flux_kontext=flux_kontext_adapter,
        glif_api=glif_api_adapter,
        registry=registry,
    )
