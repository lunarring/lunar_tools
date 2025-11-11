"""High-level language model helpers for lunar_tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from lunar_tools._optional import optional_import_attr
from lunar_tools.services.llm.conversation_service import ConversationService

__all__ = [
    "OpenAIWrapper",
    "Gemini",
    "Deepseek",
    "LanguageModels",
    "LanguageModelSelector",
    "LanguageStackConfig",
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
    openai: Optional[ConversationService]
    deepseek: Optional[ConversationService]
    gemini: Optional[ConversationService]
    selector: "LanguageModelSelector"
    preferred: ConversationService

    def get(self, name: str) -> ConversationService:
        return self.selector.get(name)


class LanguageModelSelector:
    def __init__(self) -> None:
        self._models: Dict[str, ConversationService] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        service: ConversationService,
        *,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        canonical = name.lower()
        self._models[canonical] = service
        for alias in aliases or ():
            self._aliases[alias.lower()] = canonical

    def get(self, name: str) -> ConversationService:
        key = name.lower()
        canonical = self._aliases.get(key, key)
        try:
            return self._models[canonical]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown language model '{name}'. Available: {', '.join(self.available())}") from exc

    def available(self) -> list[str]:
        return sorted(self._models.keys())


@dataclass
class LanguageStackConfig:
    include_gemini: bool = True
    preferred: str = "openai"


def create_language_models(
    *,
    include_gemini: Optional[bool] = None,
    preferred: Optional[str] = None,
    config: Optional[LanguageStackConfig] = None,
) -> LanguageModels:
    cfg = config or LanguageStackConfig()
    if include_gemini is not None:
        cfg.include_gemini = include_gemini
    if preferred is not None:
        cfg.preferred = preferred

    selector = LanguageModelSelector()

    openai_service: Optional[ConversationService] = None
    deepseek_service: Optional[ConversationService] = None
    gemini_service: Optional[ConversationService] = None

    try:
        openai_adapter = _load("OpenAIWrapper")()
    except ImportError:
        openai_service = None
    else:
        openai_service = ConversationService(openai_adapter)
        selector.register("openai", openai_service, aliases=("gpt", "openai_chat"))

    try:
        deepseek_adapter = _load("Deepseek")()
    except ImportError:
        deepseek_service = None
    else:
        deepseek_service = ConversationService(deepseek_adapter)
        selector.register("deepseek", deepseek_service)

    if cfg.include_gemini:
        try:
            gemini_adapter = _load("Gemini")()
        except ImportError:
            gemini_service = None
        else:
            gemini_service = ConversationService(gemini_adapter)
            selector.register("gemini", gemini_service, aliases=("google",))

    available_names = selector.available()
    if not available_names:
        raise ImportError("No language model adapters are available. Install the 'llm' extra.")

    try:
        preferred_service = selector.get(cfg.preferred)
    except KeyError:
        preferred_service = selector.get(available_names[0])

    return LanguageModels(
        openai=openai_service,
        deepseek=deepseek_service,
        gemini=gemini_service,
        selector=selector,
        preferred=preferred_service,
    )
