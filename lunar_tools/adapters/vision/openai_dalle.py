from __future__ import annotations

import time
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import requests
from openai import OpenAI
from PIL import Image

from lunar_tools.platform.config import read_api_key
from lunar_tools.platform.logging import create_logger


class Dalle3ImageGenerator:
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        logger=None,
        model: str = "dall-e-3",
        size_output: Tuple[int, int] = (1792, 1024),
        quality: str = "standard",
    ) -> None:
        if client is None:
            api_key = read_api_key("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            if not isinstance(client, OpenAI):
                raise TypeError("Invalid client type. Expected a 'OpenAI' instance.")
            self.client = client

        self.logger = logger if logger else create_logger(__name__)
        self.model = model
        self.quality = quality
        self.set_dimensions(size_output)

    def set_dimensions(self, size_output: Tuple[int, int]) -> None:
        allowed_sizes = ["1024x1024", "1024x1792", "1792x1024"]
        if len(size_output) != 2:
            raise ValueError("size_output must have a length of 2.")
        size_str = f"{size_output[0]}x{size_output[1]}"
        if size_str not in allowed_sizes:
            raise ValueError("Invalid size. Allowed sizes are 1024x1024, 1024x1792, and 1792x1024.")
        self.size = size_str

    def generate(self, prompt: str, simulation: bool = False):
        if simulation:
            width, height = map(int, self.size.split("x"))
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, "RGB")
            self.logger.info("Dalle3ImageGenerator: Simulation mode - random image generated")
            revised_prompt = "Simulation mode - no revised prompt"
            return image, revised_prompt

        try:
            self.logger.info("Dalle3ImageGenerator: Starting image generation")
            start_time = time.time()

            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=self.size,
                quality=self.quality,
                n=1,
            )

            image_url = response.data[0].url
            response_http = requests.get(image_url)
            response_http.raise_for_status()

            image_data = BytesIO(response_http.content)
            image = Image.open(image_data)
            end_time = time.time()
            revised_prompt = response.data[0].revised_prompt
            self.logger.info(
                "Dalle3ImageGenerator: Generation complete. Time taken: %s seconds",
                int(end_time - start_time),
            )
            return image, revised_prompt

        except requests.exceptions.RequestException as exc:
            self.logger.error("Dalle3ImageGenerator: HTTP request failed: %s", exc)
        except Exception as exc:  # pragma: no cover - catch unexpected API failures
            self.logger.error("Dalle3ImageGenerator: An error occurred: %s", exc)
        return None, None
