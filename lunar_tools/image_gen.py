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

import fal_client
from PIL import Image
import requests
from io import BytesIO


class FluxImageGenerator:
    ALLOWED_IMAGE_SIZES = [
        "square_hd", "square", "portrait_4_3", "portrait_16_9",
        "landscape_4_3", "landscape_16_9"
    ]

    def __init__(self, model="fal-ai/flux/schnell"):
        self.client = fal_client
        self.model = model
        self.last_result = None

    def generate_image(self, prompt, image_size, num_inference_steps=4, seed=420):
        if image_size not in self.ALLOWED_IMAGE_SIZES:
            raise ValueError(f"Invalid image size. Allowed sizes are: {', '.join(self.ALLOWED_IMAGE_SIZES)}")

        handler = self.client.submit(
            self.model,
            arguments={
                "prompt": prompt,
                "image_size": image_size,
                "num_inference_steps": num_inference_steps,
                "seed": seed
            },
        )

        result = handler.get()
        self.last_result = result
        image_url = result['images'][0]['url']

        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image


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


class SDXL_LCM:
    def __init__(self, client=None, logger=None, size_output=(1024, 1024), num_inference_steps=4):
        if client is None:
            self.client = replicate.Client(api_token=read_api_key("REPLICATE_API_TOKEN"))
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
            self.logger.print("SDXL_LCM: Simulation mode - random image generated")
            img_url = "Simulation mode - no image URL"
            return image, img_url
        else:
            # Normal mode: Call the API to generate an image
            try:
                self.logger.print("SDXL_LCM: Starting image generation")
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
                self.logger.print(f"SDXL_LCM: Generation complete. Time taken: {int(end_time - start_time)} seconds")

            except requests.exceptions.RequestException as e:
                self.logger.print(f"HTTP Request failed: {e}")
                return None, None
           
class SDXL_TURBO:
    def __init__(self, client=None, logger=None, size_output=(512, 512), num_inference_steps=1):
        if client is None:
            self.client = replicate.Client(api_token=read_api_key("REPLICATE_API_TOKEN"))
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
            self.logger.print("SDXL_TURBO: Simulation mode - random image generated")
            img_url = "Simulation mode - no image URL"
            return image, img_url
        else:
            # Normal mode: Call the API to generate an image
            try:
                self.logger.print("SDXL_TURBO: Starting image generation")
                start_time = time.time()

                output = self.client.run(
                    "fofr/sdxl-turbo:6244ebc4d96ffcc48fa1270d22a1f014addf79c41732fe205fb1ff638c409267",
                    input={
                        "prompt": prompt,
                        "agree_to_research_only": True,
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
                self.logger.print(f"SDXL_TURBO: Generation complete. Time taken: {int(end_time - start_time)} seconds")

            except requests.exceptions.RequestException as e:
                self.logger.print(f"HTTP Request failed: {e}")
                return None, None


class GlifAPI:
    def __init__(self, api_token=None):
        if api_token is None:
            api_token = read_api_key("GLIF_API_KEY")
        self.base_url = "https://simple-api.glif.app"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

    def run_glif(self, glif_id, inputs, timeout=60):
        """Run a glif with given inputs, with error handling and timeout."""
        url = f"{self.base_url}/{glif_id}"
        payload = {"inputs": inputs}
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=timeout)
            response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
            result = response.json()
            if 'output' in result and len(result['output']) > 0:
            
                image_url = result['output']
                if image_url.endswith(('.jpg', '.png')):
                    image_response = requests.get(image_url)
                    image_response.raise_for_status()
                    image = Image.open(BytesIO(image_response.content))
                    return {"image": image}
        except requests.exceptions.HTTPError as errh:
            return {"error": "Http Error", "message": str(errh)}
        except requests.exceptions.ConnectionError as errc:
            return {"error": "Connection Error", "message": str(errc)}
        except requests.exceptions.Timeout as errt:
            return {"error": "Timeout Error", "message": str(errt)}
        except requests.exceptions.RequestException as err:
            return {"error": "Something went wrong", "message": str(err)}


        return result

# Example usage
if __name__ == "__main__":

    generator = FluxImageGenerator()
    prompt_text = "photo of a person holding a sign with flüx written on it"
    image_size = "square_hd"
    num_inference_steps = 4
    seed = 420

    image = generator.generate_image(prompt_text, image_size, num_inference_steps, seed)

    
    # Example usage glifapi
    # glif = GlifAPI()
    # glif_id = "clgh1vxtu0011mo081dplq3xs"
    # inputs = {"node_6": "cute friendly oval shaped bot friend"}
    # result = glif.run_glif(glif_id, inputs)
    # print(result)


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
    # sdxl_turbo = SDXL_TURBO()
    # image, img_url = sdxl_turbo.generate("An astronaut riding a rainbow unicorn", "cartoon")

    # sdxl_lcm = SDXL_LCM()
    # image, img_url = sdxl_lcm.generate("An astronaut riding a rainbow unicorn", "cartoon")


    
