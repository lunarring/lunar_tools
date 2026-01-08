#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
from lunar_tools.image_gen import Dalle3ImageGenerator, SDXL_LCM
from lunar_tools.utils import read_api_key
from PIL import Image
from openai import OpenAI
import replicate

# class TestDalle3ImageGenerator(unittest.TestCase):
#     def test_openai_client_initialization(self):

#         # Set up a mock environment variable
#         with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
#             generator = Dalle3ImageGenerator()
#             self.assertIsNotNone(generator.client)

#     def test_image_generation(self):
#         # Mock the generate method
#         size_output = (1024, 1024)

#         # Test image generation with specified size
#         generator = Dalle3ImageGenerator(size_output=size_output)
#         image, _ = generator.generate("a beautiful blue house")
#         self.assertIsNotNone(image)
#         self.assertEqual(image.size, size_output)
        
#     def test_correct_client(self):
#         """Test Dalle3ImageGenerator with the correct client type."""
#         client = OpenAI(api_key=read_api_key("OPENAI_API_KEY"))
#         try:
#             generator = Dalle3ImageGenerator(client=client)
#         except Exception as e:
#             self.fail(f"Instantiation with correct client type raised an exception: {e}")

#     def test_incorrect_client(self):
#         """Test Dalle3ImageGenerator with an incorrect client type."""
#         client = WrongClient()
#         with self.assertRaises(TypeError):
#             generator = Dalle3ImageGenerator(client=client)


class TestLCMSDXL(unittest.TestCase):

    def setUp(self):
        self.sdxl_lcm = SDXL_LCM()
        
    def test_correct_client(self):
        """Test the SDXL_LCM class with the correct client type."""
        client = replicate.Client(api_token=read_api_key("REPLICATE_API_TOKEN"))
        try:
            sdxl_lcm = SDXL_LCM(client=client)
        except Exception as e:
            self.fail(f"Instantiation with correct client type raised an exception: {e}")

    def test_incorrect_client(self):
        """Test the SDXL_LCM class with an incorrect client type."""
        client = OpenAI(api_key=read_api_key("OPENAI_API_KEY"))
        with self.assertRaises(TypeError):
            sdxl_lcm = SDXL_LCM(client=client)


    def test_generate_simulation_mode(self):
        # Test the generate function in simulation mode
        image, img_url = self.sdxl_lcm.generate("An astronaut riding a rainbow unicorn", simulation=True)

        # Assertions to check if the output is as expected
        self.assertIsNotNone(image)
        self.assertEqual(img_url, "Simulation mode - no image URL")



class WrongClient:
    def __init__(self):
        pass




if __name__ == '__main__':
    unittest.main()
