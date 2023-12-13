#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("../lunar_tools/")
import unittest
from utils import read_api_key_from_lunar_config, get_os_type, save_api_key_to_lunar_config
import random
import string

# Unit tests
class TestConfigUtils(unittest.TestCase):
    def test_get_os_type(self):
        os_type = get_os_type()
        self.assertIn(os_type, ["MacOS", "Linux", "Windows"])

    def test_save_and_read_api_key(self):
        random_key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        save_api_key_to_lunar_config("TEST", random_key)
        read_key = read_api_key_from_lunar_config("TEST")
        self.assertEqual(random_key, read_key)