import unittest
import os
import time
import sys
import string
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('lunar_tools'))
sys.path.append(os.path.join(os.getcwd(), 'lunar_tools'))
from zmq_comms import ZMQClient, ZMQServer
from unittest.mock import MagicMock, patch


class TestZMQServerClient(unittest.TestCase):
    def setUp(self):
        self.server = ZMQServer(ip_server='127.0.0.1', port=5556)
        self.client = ZMQClient(ip_server='127.0.0.1', port=5556)
        
    def tearDown(self):
        self.server.stop()

    def test_init(self):
        self.assertEqual(self.server.address, "tcp://127.0.0.1:5556")
        
    def test_send_short(self):
        reply = self.client.send_json({"message": "Hello, Server!"})
        self.assertEqual(reply, "Received")
        
    def test_send_long(self):
        reply = self.client.send_json({"message": "Hello, Server!", 'bobo': 'huhu'})
        self.assertEqual(reply, "Received")
        
        
    def test_received_short(self):
        reply = self.client.send_json({"field1": "payload1"})
        msgs = self.server.get_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0], {"field1": "payload1"})
        
    def test_received_long(self):
        reply = self.client.send_json({"field1": "payload1", "field2": "payload2"})
        msgs = self.server.get_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0], {"field1": "payload1", "field2": "payload2"})
        

    def test_received_double(self):
        reply = self.client.send_json({"field1": "val1"})
        reply = self.client.send_json({"field1": "val2"})
        msgs = self.server.get_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0], {"field1": "val1"})
        self.assertEqual(msgs[1], {"field1": "val2"})


if __name__ == "__main__":
    unittest.main()

