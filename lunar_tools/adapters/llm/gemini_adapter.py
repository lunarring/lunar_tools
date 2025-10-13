from __future__ import annotations

from typing import Optional

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    genai = None
    GEMINI_AVAILABLE = False

from lunar_tools.platform.config import read_api_key
from lunar_tools.platform.logging import create_logger


class Gemini:
    def __init__(self, client: Optional["genai.Client"] = None, logger=None, model: str = "gemini-2.0-flash-exp") -> None:
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package is not installed. Install it with: pip install google-genai")

        if client is None:
            api_key = read_api_key("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = client

        self.logger = logger if logger else create_logger(__name__)
        self.model = model
        self.available_models = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
        ]

    def list_available_models(self):
        return self.available_models

    def set_model(self, model_name: str) -> None:
        if model_name in self.available_models:
            self.model = model_name
        else:
            raise ValueError(f"Model {model_name} is not available.")

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text
