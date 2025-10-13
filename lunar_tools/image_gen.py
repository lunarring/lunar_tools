"""Vision generation entry points for lunar_tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lunar_tools.adapters.vision.fal_flux import (
    FluxImageGenerator,
    FluxKontextImageGenerator,
    NanoBananaEditImageGenerator,
)
from lunar_tools.adapters.vision.glif_api import GlifAPI
from lunar_tools.adapters.vision.openai_dalle import Dalle3ImageGenerator
from lunar_tools.adapters.vision.replicate_sdxl import SDXL_LCM, SDXL_TURBO
from lunar_tools.services.vision.image_service import ImageService


@dataclass
class ImageGenerators:
    dalle3: ImageService
    sdxl_lcm: ImageService
    sdxl_turbo: ImageService
    flux: FluxImageGenerator
    nano_banana: NanoBananaEditImageGenerator
    flux_kontext: FluxKontextImageGenerator
    glif_api: Optional[GlifAPI]


def create_image_generators(*, include_glif: bool = True) -> ImageGenerators:
    dalle_adapter = Dalle3ImageGenerator()
    dalle_service = ImageService(dalle_adapter)

    sdxl_lcm_adapter = SDXL_LCM()
    sdxl_lcm_service = ImageService(sdxl_lcm_adapter)

    sdxl_turbo_adapter = SDXL_TURBO()
    sdxl_turbo_service = ImageService(sdxl_turbo_adapter)

    flux_adapter = FluxImageGenerator()
    nano_banana_adapter = NanoBananaEditImageGenerator()
    flux_kontext_adapter = FluxKontextImageGenerator()

    glif_api_adapter = GlifAPI() if include_glif else None

    return ImageGenerators(
        dalle3=dalle_service,
        sdxl_lcm=sdxl_lcm_service,
        sdxl_turbo=sdxl_turbo_service,
        flux=flux_adapter,
        nano_banana=nano_banana_adapter,
        flux_kontext=flux_kontext_adapter,
        glif_api=glif_api_adapter,
    )


__all__ = [
    "FluxImageGenerator",
    "NanoBananaEditImageGenerator",
    "FluxKontextImageGenerator",
    "Dalle3ImageGenerator",
    "SDXL_LCM",
    "SDXL_TURBO",
    "GlifAPI",
    "ImageGenerators",
    "create_image_generators",
]
