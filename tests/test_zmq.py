import unittest
import os
import time
import string
import numpy as np
from lunar_tools.comms import ZMQPairEndpoint
import time

class TestZMQPairEndpoint(unittest.TestCase):
    def setUp(self):
        self.server = ZMQPairEndpoint(is_server=True, ip='127.0.0.1', port='5556')
        self.client = ZMQPairEndpoint(is_server=False, ip='127.0.0.1', port='5556')

    def tearDown(self):
        self.server.stop()
        self.client.stop()

    def test_init_server(self):
        self.assertEqual(self.server.address, "tcp://127.0.0.1:5556")

    def test_init_client(self):
        self.assertEqual(self.client.address, "tcp://127.0.0.1:5556")

    def test_client_to_server_json(self):
        self.client.send_json({"message": "Hello, server!"})
        time.sleep(0.1)
        msgs = self.server.get_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0], {"message": "Hello, server!"})

    def test_server_to_client_json(self):
        self.server.send_json({"response": "Hello, client!"})
        time.sleep(0.1)
        msgs = self.client.get_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0], {"response": "Hello, client!"})

    def test_bidirectional_json(self):
        self.client.send_json({"message": "Client to Server"})
        time.sleep(0.1)
        self.server.send_json({"response": "Server to Client"})
        time.sleep(0.1)
        client_msgs = self.client.get_messages()
        server_msgs = self.server.get_messages()

        self.assertIn({"message": "Client to Server"}, server_msgs)
        self.assertIn({"response": "Server to Client"}, client_msgs)

    def test_client_to_server_image(self):
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.client.send_img(test_image)
        time.sleep(0.1)
        received_image = self.server.get_img()
        self.assertEqual(received_image.shape, (100, 100, 3))

    def test_server_to_client_image(self):
        test_image = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
        self.server.send_img(test_image)
        time.sleep(0.1)
        received_image = self.client.get_img()
        self.assertEqual(received_image.shape, (150, 150, 3))

    def test_bidirectional_image(self):
        client_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        server_image = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)

        self.client.send_img(client_image)
        self.server.send_img(server_image)

        time.sleep(0.1)
        server_received_image = self.server.get_img()
        client_received_image = self.client.get_img()

        self.assertEqual(server_received_image.shape, (100, 100, 3))
        self.assertEqual(client_received_image.shape, (150, 150, 3))

if __name__ == '__main__':
    unittest.main()
