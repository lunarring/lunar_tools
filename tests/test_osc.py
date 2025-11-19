import unittest
import os
import time
import sys
import string
import socket

import pytest

try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _probe_socket:
        _probe_socket.bind(("127.0.0.1", 0))
except OSError as exc:  # pragma: no cover - environment-specific
    pytest.skip(
        f"OSC tests require binding to 127.0.0.1:0 (failed with {exc}).",
        allow_module_level=True,
    )

import numpy as np

if getattr(np, "__lunar_stub__", False) or not hasattr(np, "ndarray"):
    pytest.skip("OSC tests require functional numpy.", allow_module_level=True)

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('lunar_tools'))
sys.path.append(os.path.join(os.getcwd(), 'lunar_tools'))
from comms import OSCSender, OSCReceiver

class TestOSCReceiverClient(unittest.TestCase):
    def setUp(self):
        self.sender = OSCSender('127.0.0.1')
        self.receiver = OSCReceiver('127.0.0.1')
        
    def tearDown(self):
        self.receiver.stop()
        
    def test_received_short(self):
        time.sleep(0.1)
        self.sender.send_message("/env1", 42)
        time.sleep(0.1)
        data = self.receiver.get_all_values("/env1")
        if data is not None:
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], 42)
        else:
            self.fail("No data received from OSC receiver.")
        
    def test_received_long(self):
        time.sleep(0.1)
        for i in range(10):
            self.sender.send_message("/env1", i)
            time.sleep(0.1)
        data = self.receiver.get_all_values("/env1")
        if data is not None:
            self.assertEqual(len(data), 10)
            for i in range(10):
                self.assertEqual(data[i], i)
        else:
            self.fail("No data received from OSC receiver.")



if __name__ == "__main__":
    unittest.main()
