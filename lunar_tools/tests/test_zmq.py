import unittest
import os
import time
import sys
import string
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('lunar_tools'))
sys.path.append(os.path.join(os.getcwd(), 'lunar_tools'))
from zmq_comms import ZMQSender, ZMQReceiver


class TestZMQReceiverClient(unittest.TestCase):
    def setUp(self):
        self.receiver = ZMQReceiver(ip_receiver='127.0.0.1', port_receiver=5556)
        self.sender = ZMQSender(ip_receiver='127.0.0.1', port_receiver=5556)
        
    def tearDown(self):
        self.receiver.stop()

    def test_init(self):
        self.assertEqual(self.receiver.address, "tcp://127.0.0.1:5556")
        
    def test_send_short(self):
        reply = self.sender.send_json({"message": "Hello, receiver!"})
        self.assertEqual(reply, "Received")
        
    def test_send_long(self):
        reply = self.sender.send_json({"message": "Hello, receiver!", 'bobo': 'huhu'})
        self.assertEqual(reply, "Received")
        
        
    def test_received_short(self):
        reply = self.sender.send_json({"field1": "payload1"})
        msgs = self.receiver.get_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0], {"field1": "payload1"})
        
    def test_received_long(self):
        reply = self.sender.send_json({"field1": "payload1", "field2": "payload2"})
        msgs = self.receiver.get_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0], {"field1": "payload1", "field2": "payload2"})
        

    def test_received_double(self):
        reply = self.sender.send_json({"field1": "val1"})
        reply = self.sender.send_json({"field1": "val2"})
        msgs = self.receiver.get_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0], {"field1": "val1"})
        self.assertEqual(msgs[1], {"field1": "val2"})


if __name__ == "__main__":
    unittest.main()

