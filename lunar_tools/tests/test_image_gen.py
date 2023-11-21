#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../lunar_tools/")
import unittest
from unittest.mock import patch, MagicMock
from image_gen import Dalle3ImageGenerator
import os
from PIL import Image

class TestDalle3ImageGenerator(unittest.TestCase):
    def test_openai_client_initialization(self):

        # Set up a mock environment variable
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            generator = Dalle3ImageGenerator()
            self.assertIsNotNone(generator.client)

    def test_image_generation(self):
        # Mock the generate method
        width, height = 1024, 1024
        size_str = f"{width}x{height}"

        # Test image generation with specified size
        generator = Dalle3ImageGenerator(size=size_str)
        image, _ = generator.generate("a beautiful blue house")
        self.assertIsNotNone(image)
        self.assertEqual(image.size, (width, height))

    # Additional tests for image size and other functionalities can be added here

if __name__ == '__main__':
    unittest.main()
