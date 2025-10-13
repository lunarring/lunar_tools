from __future__ import annotations

from io import BytesIO
from typing import Optional

import fal_client
import numpy as np
import requests
from PIL import Image

from lunar_tools.platform.logging import create_logger


class FluxImageGenerator:
    def __init__(self, model: str = "fal-ai/flux/schnell") -> None:
        self.client = fal_client
        self.model = model
        self.last_result = None

    def generate(self, prompt: str, image_size: str = "landscape_4_3", num_inference_steps: int = 4, seed: int = 420):
        if image_size not in [
            "square_hd",
            "square",
            "portrait_4_3",
            "portrait_16_9",
            "landscape_4_3",
            "landscape_16_9",
        ]:
            raise ValueError("Invalid image size.")

        handler = self.client.submit(
            self.model,
            arguments={
                "prompt": prompt,
                "image_size": image_size,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
            },
        )

        result = handler.get()
        self.last_result = result
        image_url = result["images"][0]["url"]

        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image


class NanoBananaEditImageGenerator:
    def __init__(self, model: str = "fal-ai/nano-banana/edit") -> None:
        self.client = fal_client
        self.model = model
        self.last_result = None

    def generate(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        image_urls: Optional[list[str]] = None,
        mask_url: Optional[str] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
        sync_mode: bool = False,
        image_size: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ):
        if not image_url and not image_urls:
            raise ValueError("Provide either image_url or image_urls")

        arguments = {
            "prompt": prompt,
            "sync_mode": sync_mode,
        }

        if image_urls is not None:
            if not isinstance(image_urls, (list, tuple)):
                raise TypeError("image_urls must be a list of URLs/data URIs")
            arguments["image_urls"] = list(image_urls)
        else:
            arguments["image_urls"] = [image_url]

        if mask_url is not None:
            arguments["mask_url"] = mask_url
        if seed is not None:
            arguments["seed"] = seed
        if num_images is not None:
            arguments["num_images"] = int(num_images)
        if image_size is not None and image_size != "custom":
            arguments["image_size"] = image_size
        if width is not None:
            arguments["width"] = int(width)
        if height is not None:
            arguments["height"] = int(height)

        arguments.update(kwargs)

        def _on_update(update):
            try:
                status = getattr(update, "status", None) or update.get("status")
                message = getattr(update, "message", None) or update.get("message")
                if status or message:
                    print(f"[NanoBanana][queue] status={status} msg={message}")
            except Exception:
                pass

        result = self.client.subscribe(
            self.model,
            arguments=arguments,
            with_logs=True,
            on_queue_update=_on_update,
        )
        self.last_result = result

        image_url_out = None
        if isinstance(result, dict):
            if "images" in result and result["images"]:
                image_url_out = result["images"][0].get("url")
            elif "data" in result and isinstance(result["data"], dict):
                images = result["data"].get("images")
                if images:
                    image_url_out = images[0].get("url")

        if not image_url_out:
            raise ValueError("NanoBanana: could not find images in result")

        response = requests.get(image_url_out)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img


class FluxKontextImageGenerator:
    def __init__(self, model: str = "fal-ai/flux-pro/kontext") -> None:
        self.client = fal_client
        self.model = model
        self.last_result = None
        self.logger = create_logger(__name__)

    def generate(
        self,
        prompt: str,
        image_url: str,
        seed: Optional[int] = None,
        guidance_scale: float = 3.5,
        num_images: int = 1,
        output_format: str = "jpeg",
        safety_tolerance: str = "2",
        aspect_ratio: Optional[str] = None,
        sync_mode: bool = False,
    ):
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance,
            "sync_mode": sync_mode,
        }

        if seed is not None:
            arguments["seed"] = seed
        if aspect_ratio:
            arguments["aspect_ratio"] = aspect_ratio

        result = self.client.subscribe(
            self.model,
            arguments=arguments,
            with_logs=True,
            on_queue_update=lambda update: None,
        )
        self.last_result = result
        return result
