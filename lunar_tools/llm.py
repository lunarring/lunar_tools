"""High-level language model helpers for lunar_tools."""

from __future__ import annotations

from dataclasses import dataclass

from lunar_tools.adapters.llm.deepseek_adapter import Deepseek
from lunar_tools.adapters.llm.gemini_adapter import GEMINI_AVAILABLE, Gemini
from lunar_tools.adapters.llm.openai_adapter import OpenAIWrapper
from lunar_tools.services.llm.conversation_service import ConversationService


@dataclass
class LanguageModels:
    openai: ConversationService
    deepseek: ConversationService
    gemini: ConversationService | None


def create_language_models(include_gemini: bool = True) -> LanguageModels:
    openai_adapter = OpenAIWrapper()
    deepseek_adapter = Deepseek()

    openai_service = ConversationService(openai_adapter)
    deepseek_service = ConversationService(deepseek_adapter)

    gemini_service = None
    if include_gemini and GEMINI_AVAILABLE:
        gemini_adapter = Gemini()
        gemini_service = ConversationService(gemini_adapter)

    return LanguageModels(
        openai=openai_service,
        deepseek=deepseek_service,
        gemini=gemini_service,
    )


__all__ = [
    "OpenAIWrapper",
    "Gemini",
    "Deepseek",
    "LanguageModels",
    "create_language_models",
]
