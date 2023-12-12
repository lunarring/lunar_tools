#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from io import BytesIO
import time
import os
from PIL import Image
import numpy as np
from openai import OpenAI
import replicate
from lunar_tools.logprint import LogPrint
from lunar_tools.utils import read_api_key

class Dalle3ImageGenerator:
    def __init__(self,
                 client=None,
                 logger=None,
                 model="dall-e-3",
                 size_output=(1792, 1024),
                 quality="standard"):
        if client is None:
            api_key = read_api_key("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OPENAI_API_KEY found in environment variables")
            self.client = OpenAI(api_key=api_key)
        else:
            if not isinstance(client, OpenAI):
                raise TypeError("Invalid client type. Expected a 'OpenAI' instance.")
            self.client = client

        self.logger = logger if logger else LogPrint()
        self.model = model
        self.quality = quality
        self.set_dimensions(size_output)

    def set_dimensions(self, size_output):
        allowed_sizes = ["1024x1024", "1024x1792", "1792x1024"]

        # Check if size_output has a length of 2
        if len(size_output) != 2:
            raise ValueError("size_output must have a length of 2.")

        # Convert the input tuple to a string format
        size_str = f"{size_output[0]}x{size_output[1]}"

        if size_str not in allowed_sizes:
            raise ValueError("Invalid size. Allowed sizes are 1024x1024, 1024x1792, and 1792x1024.")
        else:
            self.size = size_str

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


class LCM_SDXL:
    def __init__(self, client=None, logger=None, size_output=(1024, 1024), num_inference_steps=4):
        if client is None:
            self.client = replicate.Client(api_token=read_api_key("REPLICATE_API_KEY"))
        else:
            if not isinstance(client, replicate.Client):
                raise TypeError("Invalid client type. Expected a 'replicate.Client' instance.")
            self.client = client

        self.logger = logger if logger else LogPrint()
        self.size = size_output
        self.num_inference_steps = num_inference_steps

    def set_dimensions(self, size_output):
        self.size = size_output

    def set_num_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps

    def generate(self, prompt, negative_prompt="", simulation=False):
        width, height = self.size
        num_inference_steps = self.num_inference_steps

        if simulation:
            # Simulation mode: Generate a random image
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, 'RGB')
            self.logger.print("LCM_SDXL: Simulation mode - random image generated")
            img_url = "Simulation mode - no image URL"
            return image, img_url
        else:
            # Normal mode: Call the API to generate an image
            try:
                self.logger.print("LCM_SDXL: Starting image generation")
                start_time = time.time()

                output = self.client.run(
                    "lucataco/sdxl-lcm:fbbd475b1084de80c47c35bfe4ae64b964294aa7e237e6537eed938cfd24903d",
                    input={
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps
                    }
                )

                img_url = output[0]
                response_http = requests.get(img_url)
                response_http.raise_for_status()

                image_data = BytesIO(response_http.content)
                image = Image.open(image_data)
                end_time = time.time()
                return image, img_url
                self.logger.print(f"LCM_SDXL: Generation complete. Time taken: {int(end_time - start_time)} seconds")

            except requests.exceptions.RequestException as e:
                self.logger.print(f"HTTP Request failed: {e}")
                return None, None
           



if __name__ == "__main__":
    # Example usage Dalle3
    # dalle3 = Dalle3ImageGenerator()
    # image, revised_prompt = dalle3.generate("realistic photo of a ")
    # image = image.resize((1024, 576))
    # image.save("/Users/jjj/glif/git_remote/generative-models/assets/fluid3.jpg")
    # image.show()
    
    # output = replicate.run(
    #     "lucataco/sdxl-lcm:fbbd475b1084de80c47c35bfe4ae64b964294aa7e237e6537eed938cfd24903d",
    #     input={"prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic",
    #            "negative_prompt": "cartoon",
    #            "width": 1280,
    #            "height": 1024,
    #            "num_inference_steps": 6}
    # )
    # img_url = output[0]
    
    # Example usage
    # client = OpenAI()
    lcm_sdxl = LCM_SDXL()
    image, img_url = lcm_sdxl.generate("An astronaut riding a rainbow unicorn", "cartoon")


    
    
    """
    assert clients are correct
    """
    
    
