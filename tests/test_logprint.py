#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import unittest
from unittest import mock

from lunar_tools.logprint import LogPrint, dynamic_print

class TestLogPrint(unittest.TestCase):

    def test_log_creation_with_specific_filename(self):
        """Test if a log file is created with a specific filename."""
        specific_filename = "tmp_testlog.txt"
        logger = LogPrint(filename=specific_filename)
        logger.print("Test log with specific filename.")
        self.assertTrue(os.path.exists(specific_filename))

    def test_log_creation_with_default_filename(self):
        """When no filename is provided, LogPrint should avoid file logging."""
        logger = LogPrint()
        logger.print("Test log with default filename.")
        self.assertIsNone(logger.filename)

    def test_log_file_content(self):
        """Test if the log file contains the expected content."""
        test_filename = "specific_filename.txt"
        test_message = "This is a test log message."
        logger = LogPrint(filename=test_filename)
        logger.print(test_message)
        with open(test_filename, 'r') as file:
            content = file.read()
            self.assertIn(test_message, content)

    def test_dynamic_print_persist_writes_newline(self):
        """dynamic_print should append a newline when persist=True."""
        class FakeTTYStream(io.StringIO):
            def isatty(self):
                return True

        stream = FakeTTYStream()
        with mock.patch("lunar_tools.logprint._get_terminal_width", return_value=10):
            dynamic_print("Hello", stream=stream, persist=True)
        output = stream.getvalue()
        self.assertTrue(output.endswith("\n"))
        self.assertIn("Hello", output)

    def test_dynamic_print_non_tty_falls_back_to_plain_print(self):
        """dynamic_print should degrade gracefully when stream is not a TTY."""
        stream = io.StringIO()
        dynamic_print("Plain message", stream=stream)
        self.assertEqual(stream.getvalue(), "Plain message\n")

    def test_dynamic_print_truncates_long_messages(self):
        """Long messages should be truncated to fit the terminal width."""
        class FakeTTYStream(io.StringIO):
            def isatty(self):
                return True

        stream = FakeTTYStream()
        with mock.patch("lunar_tools.logprint._get_terminal_width", return_value=10):
            dynamic_print("This is a very long message", stream=stream)
        self.assertIn("This is...", stream.getvalue())

if __name__ == '__main__':
    unittest.main()
