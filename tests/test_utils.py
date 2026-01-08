#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import unittest
from lunar_tools.utils import read_api_key, get_os_type

class TestEnvVarAPIKeys(unittest.TestCase):
    def setUp(self):
        os.environ["TEST_API_KEY"] = "12345"
        
    def tearDown(self):
        if "TEST_API_KEY" in os.environ:
            del os.environ["TEST_API_KEY"]
            
    def test_get_os_type(self):
        os_type = get_os_type()
        self.assertIn(os_type, ["MacOS", "Linux", "Windows"])

    def test_read_existing_key(self):
        self.assertEqual(read_api_key("TEST_API_KEY"), "12345")
        
    def test_read_missing_key(self):
        self.assertIsNone(read_api_key("NON_EXISTENT_KEY"))
