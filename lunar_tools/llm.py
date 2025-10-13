"""High-level language model helpers for lunar_tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lunar_tools._optional import optional_import_attr
from lunar_tools.services.llm.conversation_service import ConversationService

__all__ = [
    "OpenAIWrapper",
    "Gemini",
    "Deepseek",
    "LanguageModels",
    "create_language_models",
]

_OPTIONAL_EXPORTS = {
    "OpenAIWrapper": ("lunar_tools.adapters.llm.openai_adapter", "OpenAIWrapper"),
    "Deepseek": ("lunar_tools.adapters.llm.deepseek_adapter", "Deepseek"),
    "Gemini": ("lunar_tools.adapters.llm.gemini_adapter", "Gemini"),
}


def _load(name: str):
    module, attribute = _OPTIONAL_EXPORTS[name]
    return optional_import_attr(
        module,
        attribute,
        feature=name,
        extras="llm",
    )


def __getattr__(name: str):
    if name in _OPTIONAL_EXPORTS:
        value = _load(name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class LanguageModels:
    openai: ConversationService
    deepseek: ConversationService
    gemini: Optional[ConversationService]


def create_language_models(include_gemini: bool = True) -> LanguageModels:
    openai_adapter = _load("OpenAIWrapper")()
    deepseek_adapter = _load("Deepseek")()

    openai_service = ConversationService(openai_adapter)
    deepseek_service = ConversationService(deepseek_adapter)

    gemini_service: Optional[ConversationService] = None
    if include_gemini:
        try:
            gemini_adapter = _load("Gemini")()
        except ImportError:
            gemini_service = None
        else:
            gemini_service = ConversationService(gemini_adapter)

    return LanguageModels(
        openai=openai_service,
        deepseek=deepseek_service,
        gemini=gemini_service,
    )
