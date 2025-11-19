"""Utility script to verify optional extras load their modules.

Each extra maps to a set of modules that should import successfully once the
corresponding optional dependencies are installed. This script is intended to
run inside tox environments dedicated to extras validation (Phase E hardening).
"""

from __future__ import annotations

import argparse
import importlib
from typing import Dict, Iterable, List

EXTRA_MODULES: Dict[str, List[str]] = {
    "audio": [
        "lunar_tools.audio",
        "lunar_tools.adapters.audio.openai_tts",
        "lunar_tools.adapters.audio.elevenlabs_tts",
        "lunar_tools.adapters.audio.deepgram_transcribe",
    ],
    "llm": [
        "lunar_tools.llm",
        "lunar_tools.adapters.llm.openai_adapter",
        "lunar_tools.adapters.llm.deepseek_adapter",
        "lunar_tools.adapters.llm.gemini_adapter",
    ],
    "vision": [
        "lunar_tools.image_gen",
        "lunar_tools.adapters.vision.openai_dalle",
        "lunar_tools.adapters.vision.replicate_sdxl",
        "lunar_tools.adapters.vision.fal_flux",
    ],
    "presentation": [
        "lunar_tools.presentation.audio_stack",
        "lunar_tools.presentation.display_stack",
        "lunar_tools.presentation.movie_stack",
        "lunar_tools.presentation.realtime_voice",
    ],
}


def _smoke_imports(modules: Iterable[str]) -> None:
    failures = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - failure reporting
            failures.append((module, exc))
    if failures:
        details = "\n".join(f"- {name}: {err}" for name, err in failures)
        raise SystemExit(f"Extra smoke imports failed:\n{details}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "extra",
        choices=sorted(EXTRA_MODULES.keys()),
        help="Name of the optional extra to validate.",
    )
    args = parser.parse_args()
    _smoke_imports(EXTRA_MODULES[args.extra])


if __name__ == "__main__":
    main()
