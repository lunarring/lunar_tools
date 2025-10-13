from __future__ import annotations

from typing import Optional

from openai import OpenAI

from lunar_tools.platform.config import read_api_key
from lunar_tools.platform.logging import create_logger


class Deepseek:
    def __init__(self, client: Optional[OpenAI] = None, logger=None, model: str = "deepseek-chat") -> None:
        if client is None:
            api_key = read_api_key("DEEPSEEK_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            self.client = client

        self.logger = logger if logger else create_logger(__name__)
        self.model = model
        self.available_models = ["deepseek-chat", "deepseek-reasoner"]

    def list_available_models(self):
        return self.available_models

    def set_model(self, model_name: str) -> None:
        if model_name in self.available_models:
            self.model = model_name
        else:
            raise ValueError(f"Model {model_name} is not available.")

    def generate(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return chat_completion.choices[0].message.content
