#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../lunar_tools/")
import unittest
import os
from logprint import LogPrint  # Replace 'your_module_name' with the actual name of your module

class TestLogPrint(unittest.TestCase):

    def test_log_creation_with_specific_filename(self):
        """Test if a log file is created with a specific filename."""
        specific_filename = "tmp_testlog.txt"
        logger = LogPrint(filename=specific_filename)
        logger.print("Test log with specific filename.")
        self.assertTrue(os.path.exists(specific_filename))

    def test_log_creation_with_default_filename(self):
        """Test if a log file is created with the default filename."""
        logger = LogPrint()
        logger.print("Test log with default filename.")
        # Extract the filename from the logger
        filename = logger.logger.handlers[1].baseFilename
        self.assertTrue(os.path.exists(filename))

    def test_log_file_content(self):
        """Test if the log file contains the expected content."""
        test_filename = "specific_filename.txt"
        test_message = "This is a test log message."
        logger = LogPrint(filename=test_filename)
        logger.print(test_message)
        with open(test_filename, 'r') as file:
            content = file.read()
            self.assertIn(test_message, content)

if __name__ == '__main__':
    unittest.main()
