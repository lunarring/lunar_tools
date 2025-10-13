from __future__ import annotations

from typing import Optional

from openai import OpenAI

from lunar_tools.platform.config import read_api_key
from lunar_tools.platform.logging import create_logger


class OpenAIWrapper:
    def __init__(self, client: Optional[OpenAI] = None, logger=None, model: str = "gpt-4-0613") -> None:
        if client is None:
            api_key = read_api_key("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client
        self.logger = logger if logger else create_logger(__name__)
        self.model = model
        self.available_models = ["gpt-4-1106-preview", "gpt-4-0613", "gpt-3.5-turbo-1106"]

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
