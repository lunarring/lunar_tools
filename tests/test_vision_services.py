from __future__ import annotations

import pytest

from lunar_tools.image_gen import create_image_generators
from lunar_tools.services.vision.image_service import ImageService, VisionProviderRegistry
from tests.fakes import FakeImageGeneratorPort


def test_vision_provider_registry_register_and_alias():
    registry = VisionProviderRegistry()
    service = ImageService(FakeImageGeneratorPort(image="alpha", metadata={"provider": "alpha"}))
    registry.register("alpha", service, aliases=("primary",))

    assert registry.get("alpha") is service
    assert registry.get("primary") is service
    assert registry.available() == ["alpha"]

    with pytest.raises(KeyError):
        registry.get("missing")


def test_create_image_generators_registers_services(monkeypatch):
    def fake_load(name: str):
        mapping = {
            "Dalle3ImageGenerator": "dalle",
            "SDXL_LCM": "sdxl_lcm",
            "SDXL_TURBO": "sdxl_turbo",
            "FluxImageGenerator": "flux",
            "NanoBananaEditImageGenerator": "nano_banana",
            "FluxKontextImageGenerator": "kontext",
            "GlifAPI": "glif",
        }

        class _FakeGenerator(FakeImageGeneratorPort):
            def __init__(self):
                super().__init__(image=mapping[name], metadata={"provider": mapping[name]})

        return _FakeGenerator

    monkeypatch.setattr("lunar_tools.image_gen._load", fake_load)

    generators = create_image_generators(include_glif=True)

    dalle_service = generators.get("openai")
    result = dalle_service.generate("prompt").metadata["raw"]
    assert result == {"provider": "dalle"}

    assert "flux" in generators.registry.available()
    flux_service = generators.get("flux")
    assert flux_service.generate("beam").metadata["raw"] == {"provider": "flux"}
