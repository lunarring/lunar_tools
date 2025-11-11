from __future__ import annotations

import pytest

from lunar_tools.llm import LanguageStackConfig, create_language_models


class DummyModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return f"{self.name}:{prompt}"


def test_language_models_selector(monkeypatch):
    providers = {
        "OpenAIWrapper": DummyModel("openai"),
        "Deepseek": DummyModel("deepseek"),
    }

    def fake_load(name: str):
        try:
            model = providers[name]
        except KeyError:
            raise ImportError(name)

        return lambda: model

    monkeypatch.setattr("lunar_tools.llm._load", fake_load)

    models = create_language_models(include_gemini=False, preferred="deepseek")

    deepseek = models.get("deepseek")
    assert deepseek.complete("hello") == "deepseek:hello"

    openai = models.get("gpt")
    assert openai.complete("hi") == "openai:hi"

    assert models.preferred is deepseek
    assert models.selector.available() == ["deepseek", "openai"]

    with pytest.raises(KeyError):
        models.get("gemini")


def test_language_models_fallback_preferred(monkeypatch):
    providers = {
        "OpenAIWrapper": DummyModel("openai"),
    }

    def fake_load(name: str):
        if name == "Deepseek":
            raise ImportError("deepseek missing")
        if name == "Gemini":
            raise ImportError("gemini missing")
        return lambda: providers[name]

    monkeypatch.setattr("lunar_tools.llm._load", fake_load)

    models = create_language_models(config=LanguageStackConfig(include_gemini=True, preferred="nonexistent"))

    assert models.preferred is models.get("openai")
