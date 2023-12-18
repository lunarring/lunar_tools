import unittest
import os
import time
import sys
import string
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('lunar_tools'))
sys.path.append(os.path.join(os.getcwd(), 'lunar_tools'))
from osc import OSCSender, OSCReceiver
import numpy as np


class TestOSCReceiverClient(unittest.TestCase):
    def setUp(self):
        self.sender = OSCSender('127.0.0.1')
        self.receiver = OSCReceiver('127.0.0.1')
        
    def tearDown(self):
        self.receiver.stop()

        
    def test_received_short(self):
        self.sender.send_message("/env1", 42)
        time.sleep(0.1)
        data = self.receiver.get_all_values("/env1")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], 42)
        
        
    def test_received_long(self):
        for i in range(10):
            self.sender.send_message("/env1", i)
            time.sleep(0.1)
        data = self.receiver.get_all_values("/env1")
        self.assertEqual(len(data), 10)
        for i in range(10):
            self.assertEqual(data[i], i)

        



if __name__ == "__main__":
    unittest.main()

