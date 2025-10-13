from __future__ import annotations

import time
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import requests
import replicate
from PIL import Image

from lunar_tools.platform.config import read_api_key
from lunar_tools.platform.logging import create_logger


class SDXL_LCM:
    def __init__(
        self,
        client: Optional[replicate.Client] = None,
        logger=None,
        size_output: Tuple[int, int] = (1024, 1024),
        num_inference_steps: int = 4,
    ) -> None:
        if client is None:
            self.client = replicate.Client(api_token=read_api_key("REPLICATE_API_TOKEN"))
        else:
            if not isinstance(client, replicate.Client):
                raise TypeError("Invalid client type. Expected a 'replicate.Client' instance.")
            self.client = client

        self.logger = logger if logger else create_logger(__name__ + ".SDXL_LCM")
        self.size = size_output
        self.num_inference_steps = num_inference_steps

    def set_dimensions(self, size_output: Tuple[int, int]) -> None:
        self.size = size_output

    def set_num_inference_steps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps

    def generate(self, prompt: str, negative_prompt: str = "", simulation: bool = False):
        width, height = self.size
        num_inference_steps = self.num_inference_steps

        if simulation:
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, "RGB")
            self.logger.info("SDXL_LCM: Simulation mode - random image generated")
            img_url = "Simulation mode - no image URL"
            return image, img_url

        try:
            self.logger.info("SDXL_LCM: Starting image generation")
            start_time = time.time()

            output = self.client.run(
                "lucataco/sdxl-lcm:fbbd475b1084de80c47c35bfe4ae64b964294aa7e237e6537eed938cfd24903d",
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                },
            )

            img_url = output[0]
            response_http = requests.get(img_url)
            response_http.raise_for_status()

            image_data = BytesIO(response_http.content)
            image = Image.open(image_data)
            end_time = time.time()
            self.logger.info(
                "SDXL_LCM: Generation complete. Time taken: %s seconds",
                int(end_time - start_time),
            )
            return image, img_url

        except requests.exceptions.RequestException as exc:
            self.logger.error("SDXL_LCM: HTTP request failed: %s", exc)
        return None, None


class SDXL_TURBO:
    def __init__(
        self,
        client: Optional[replicate.Client] = None,
        logger=None,
        size_output: Tuple[int, int] = (512, 512),
        num_inference_steps: int = 1,
    ) -> None:
        if client is None:
            self.client = replicate.Client(api_token=read_api_key("REPLICATE_API_TOKEN"))
        else:
            if not isinstance(client, replicate.Client):
                raise TypeError("Invalid client type. Expected a 'replicate.Client' instance.")
            self.client = client

        self.logger = logger if logger else create_logger(__name__ + ".SDXL_TURBO")
        self.size = size_output
        self.num_inference_steps = num_inference_steps

    def set_dimensions(self, size_output: Tuple[int, int]) -> None:
        self.size = size_output

    def set_num_inference_steps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps

    def generate(self, prompt: str, negative_prompt: str = "", simulation: bool = False):
        width, height = self.size
        num_inference_steps = self.num_inference_steps

        if simulation:
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, "RGB")
            self.logger.info("SDXL_TURBO: Simulation mode - random image generated")
            img_url = "Simulation mode - no image URL"
            return image, img_url

        try:
            self.logger.info("SDXL_TURBO: Starting image generation")
            start_time = time.time()

            output = self.client.run(
                "fofr/sdxl-turbo:6244ebc4d96ffcc48fa1270d22a1f014addf79c41732fe205fb1ff638c409267",
                input={
                    "prompt": prompt,
                    "agree_to_research_only": True,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                },
            )

            img_url = output[0]
            response_http = requests.get(img_url)
            response_http.raise_for_status()

            image_data = BytesIO(response_http.content)
            image = Image.open(image_data)
            end_time = time.time()
            self.logger.info(
                "SDXL_TURBO: Generation complete. Time taken: %s seconds",
                int(end_time - start_time),
            )
            return image, img_url

        except requests.exceptions.RequestException as exc:
            self.logger.error("SDXL_TURBO: HTTP request failed: %s", exc)
        return None, None
