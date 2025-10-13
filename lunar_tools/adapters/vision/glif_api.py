from __future__ import annotations

from io import BytesIO
from typing import Optional

import requests
from PIL import Image

from lunar_tools.platform.config import read_api_key


class GlifAPI:
    def __init__(self, api_token: Optional[str] = None) -> None:
        if api_token is None:
            api_token = read_api_key("GLIF_API_KEY")
        self.base_url = "https://simple-api.glif.app"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def run_glif(self, glif_id: str, inputs: dict, timeout: int = 60):
        url = f"{self.base_url}/{glif_id}"
        payload = {"inputs": inputs}
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            if "output" in result and result["output"]:
                image_url = result["output"]
                if isinstance(image_url, str) and image_url.endswith((".jpg", ".png")):
                    image_response = requests.get(image_url)
                    image_response.raise_for_status()
                    image = Image.open(BytesIO(image_response.content))
                    return {"image": image}
            return result
        except requests.exceptions.HTTPError as errh:
            return {"error": "Http Error", "message": str(errh)}
        except requests.exceptions.ConnectionError as errc:
            return {"error": "Connection Error", "message": str(errc)}
        except requests.exceptions.Timeout as errt:
            return {"error": "Timeout Error", "message": str(errt)}
        except requests.exceptions.RequestException as err:
            return {"error": "Something went wrong", "message": str(err)}
