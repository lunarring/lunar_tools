from __future__ import annotations


class FakeLanguageModelPort:
    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response
