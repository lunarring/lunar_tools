#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from io import BytesIO
import time
import os
import tempfile
import base64
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

    def generate(self, prompt, image_size='landscape_4_3', num_inference_steps=4, seed=420):
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


class NanoBananaEditImageGenerator:
    """
    Wrapper for fal.ai model: "fal-ai/nano-banana/edit".

    Notes:
    - Accepts a single image (base64 data URI or URL) via `image_url`, or a list via `image_urls`.
    - If a mask is provided, pass it via `mask_url` (base64 data URI or URL). Areas to edit should be white.
    - Additional tuning args supported by fal can be forwarded via **kwargs.
    - Returns a PIL.Image for the first image in the response.
    """

    def __init__(self, model="fal-ai/nano-banana/edit"):
        self.client = fal_client
        self.model = model
        self.last_result = None

    def generate(self,
                 prompt: str,
                 image_url=None,
                 image_urls=None,
                 mask_url=None,
                 seed=None,
                 num_images: int = 1,
                 sync_mode: bool = False,
                 image_size: str = None,
                 width: int = None,
                 height: int = None,
                 **kwargs):
        """
        Generate edited image(s) with Nano Banana edit.

        Args:
            prompt: Edit prompt.
            image_url: Single input image (URL or base64 data URI).
            image_urls: Multiple input images (list of URLs or base64 data URIs).
            mask_url: Optional mask image (URL or base64 data URI). White = edit.
            seed: Optional random seed for reproducibility.
            num_images: Number of images to generate (for single input image).
            sync_mode: If True, wait for image before returning.
            **kwargs: Forwarded to fal model arguments for advanced controls.
        """

        if not image_url and not image_urls:
            raise ValueError("Provide either image_url or image_urls")

        arguments = {
            "prompt": prompt,
            "sync_mode": sync_mode,
        }

        # Always send list field expected by the API
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
        # If custom or explicit width/height provided, include them
        if width is not None:
            arguments["width"] = int(width)
        if height is not None:
            arguments["height"] = int(height)

        # Merge any additional kwargs (e.g., guidance_scale, steps, strength, negative_prompt, etc.)
        arguments.update(kwargs)

        # Verbose logging about the request
        try:
            num_inputs = len(arguments.get("image_urls", []))
        except Exception:
            num_inputs = 1 if arguments.get("image_url") else 0
        print(f"[NanoBanana] Submitting to {self.model} | images={num_inputs} | size={arguments.get('image_size') or 'custom'} | w={arguments.get('width')} | h={arguments.get('height')} | seed={arguments.get('seed')}")

        def _on_update(update):
            try:
                status = getattr(update, 'status', None) or update.get('status')
                message = getattr(update, 'message', None) or update.get('message')
                if status or message:
                    print(f"[NanoBanana][queue] status={status} msg={message}")
            except Exception:
                pass

        result = self.client.subscribe(
            self.model,
            arguments=arguments,
            with_logs=True,
            on_queue_update=_on_update
        )
        self.last_result = result

        # Extract first image URL from known result shapes
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

        print(f"[NanoBanana] Received response. Downloading image: {image_url_out}")
        response = requests.get(image_url_out)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        print(f"[NanoBanana] Image downloaded. size={img.size} mode={img.mode}")
        return img


class FluxKontextImageGenerator:
    ALLOWED_ASPECT_RATIOS = [
        "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"
    ]
    
    ALLOWED_OUTPUT_FORMATS = ["jpeg", "png"]
    
    ALLOWED_SAFETY_TOLERANCE = ["1", "2", "3", "4", "5", "6"]

    def __init__(self, model="fal-ai/flux-pro/kontext"):
        self.client = fal_client
        self.model = model
        self.last_result = None

    def generate(self, prompt, image_url, seed=None, guidance_scale=3.5, num_images=1, 
                 output_format="jpeg", safety_tolerance="2", aspect_ratio=None, sync_mode=False):
        """
        Generate an edited image using Flux Kontext.
        
        Args:
            prompt (str): Description of what to change in the image
            image_url (str): URL or path to the input image
            seed (int, optional): Random seed for reproducible results
            guidance_scale (float): CFG scale (default: 3.5)
            num_images (int): Number of images to generate (default: 1)
            output_format (str): Output format - "jpeg" or "png" (default: "jpeg")
            safety_tolerance (str): Safety level 1-6, 1 most strict (default: "2")
            aspect_ratio (str, optional): Aspect ratio like "16:9", "1:1", etc.
            sync_mode (bool): Wait for image before returning (default: False)
        """
        if output_format not in self.ALLOWED_OUTPUT_FORMATS:
            raise ValueError(f"Invalid output format. Allowed formats are: {', '.join(self.ALLOWED_OUTPUT_FORMATS)}")
            
        if safety_tolerance not in self.ALLOWED_SAFETY_TOLERANCE:
            raise ValueError(f"Invalid safety tolerance. Allowed values are: {', '.join(self.ALLOWED_SAFETY_TOLERANCE)}")
            
        if aspect_ratio and aspect_ratio not in self.ALLOWED_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect ratio. Allowed ratios are: {', '.join(self.ALLOWED_ASPECT_RATIOS)}")

        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance,
            "sync_mode": sync_mode
        }
        
        # Add optional parameters if provided
        if seed is not None:
            arguments["seed"] = seed
        if aspect_ratio:
            arguments["aspect_ratio"] = aspect_ratio

        result = self.client.subscribe(
            self.model,
            arguments=arguments,
            with_logs=True,
            on_queue_update=lambda update: None  # You can add logging here if needed
        )
        self.last_result = result
        
        # Try to access the image URL correctly based on the result structure
        if 'images' in result:
            image_url = result['images'][0]['url']
        elif 'data' in result and 'images' in result['data']:
            image_url = result['data']['images'][0]['url']
        else:
            raise ValueError("Could not find images in result structure")

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

    flux = FluxImageGenerator()
    prompt_text = "photo of a person holding a sign with flüx written on it"
    image_size = "square_hd"
    num_inference_steps = 4
    seed = 420

    image = flux.generate(prompt_text, image_size, num_inference_steps, seed)

    
    # Example usage Flux Kontext - image to image editing
    print("Running Flux Kontext example...")
    
    # Check if we already have an original image
    if os.path.exists("original.jpg"):
        print("Using existing original.jpg")
    else:
        # Download a random image and save it
        temp_url = "https://picsum.photos/600/400"
        print(f"Downloading image from: {temp_url}")
        
        response = requests.get(temp_url)
        response.raise_for_status()
        original_image = Image.open(BytesIO(response.content))
        original_image.save("original.jpg")
        print("Original image saved as: original.jpg")
    
    # Convert the local file to base64 data URI
    print("Converting original.jpg to base64...")
    with open("original.jpg", "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{image_base64}"
    print("Image converted to base64 data URI")
    
    # Use Flux Kontext to edit the image
    flux_kontext = FluxKontextImageGenerator()
    edit_prompt = "Change the colors to sunset colors but keep the original shapes"
    
    # Generate edited image using the base64 data URI
    edited_image = flux_kontext.generate(
        prompt=edit_prompt,
        image_url=data_uri,
        seed=420,
        guidance_scale=3.5
    )
    
    print("Flux Kontext editing complete!")
    
    # Save the edited result in current directory
    edited_image.save("edited.jpg")
    print("Edited image saved as: edited.jpg")
    
    # Show the images if running interactively
    # original_image.show()
    # edited_image.show()

    
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


    
