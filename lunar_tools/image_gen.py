#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from io import BytesIO
import time
import os
from PIL import Image
import numpy as np
from openai import OpenAI
from lunar_tools.logprint import LogPrint


class Dalle3ImageGenerator:
    def __init__(self,
                 client=None,
                 logger=None,
                 model="dall-e-3",
                 size="1792x1024",
                 quality="standard"):
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client

        self.logger = logger if logger else LogPrint()
        self.model = model
        self.size = size
        self.quality = quality

    def generate(self, prompt, simulation=False):
        if simulation:
            # Simulation mode: Generate a random image
            width, height = map(int, self.size.split('x'))
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, 'RGB')
            self.logger.print("Dalle3ImageGenerator: Simulation mode - random image generated")
            revised_prompt = "Simulation mode - no revised prompt"
        else:
            # Normal mode: Call the API to generate an image
            try:
                self.logger.print("Dalle3ImageGenerator: Starting image generation")
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
                self.logger.print(f"Dalle3ImageGenerator: Generation complete. Time taken: {int(end_time - start_time)} seconds")

            except requests.exceptions.RequestException as e:
                self.logger.print(f"HTTP Request failed: {e}")
                return None, None
            except Exception as e:
                self.logger.print(f"An error occurred: {e}")
                return None, None

        return image, revised_prompt


if __name__ == "__main__":
    # Example usage
    dalle3 = Dalle3ImageGenerator()
    image, revised_prompt = dalle3.generate("realistic photo of a ")
    image = image.resize((1024, 576))
    image.save("/Users/jjj/glif/git_remote/generative-models/assets/fluid3.jpg")
    image.show()
