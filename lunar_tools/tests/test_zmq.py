import unittest
import os
import time
import sys
import string
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('lunar_tools'))
sys.path.append(os.path.join(os.getcwd(), 'lunar_tools'))
from zmq_comms import ZMQSender, ZMQReceiver
import numpy as np


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
        
    def test_send_image_size(self):
        # Send an image of specific size
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.sender.send_img(test_image)
        received_image = self.receiver.get_img()
        self.assertEqual(received_image.shape, (100, 100, 3))

    def test_send_image_different_size(self):
        # Send an image of a different size
        test_image = np.random.randint(0, 256, (200, 150, 3), dtype=np.uint8)
        self.sender.send_img(test_image)
        received_image = self.receiver.get_img()
        self.assertEqual(received_image.shape, (200, 150, 3))

    def test_send_image_update(self):
        # Send an image, then send another and check if it gets updated
        first_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.sender.send_img(first_image)
        first_received_image = self.receiver.get_img()

        second_image = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
        self.sender.send_img(second_image)
        second_received_image = self.receiver.get_img()

        self.assertNotEqual(first_received_image.tostring(), second_received_image.tostring())
        self.assertEqual(second_received_image.shape, (150, 150, 3))



if __name__ == "__main__":
    unittest.main()

