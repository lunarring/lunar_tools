#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import json
import threading
import cv2
import numpy as np
from queue import Queue
from lunar_tools.logprint import LogPrint

class ZMQReceiver:
    def __init__(self, ip_receiver="*", port_receiver="5555", logger=None, start=True):
        self.address = f"tcp://{ip_receiver}:{port_receiver}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.address)
        self.messages = Queue()
        self.last_image = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.logger = logger if logger else LogPrint()
        self.running = False
        if start:
            self.start()

    def run(self):
        self.running = True
        while self.running:
            try:
                message = self.socket.recv(zmq.NOBLOCK)
                if message.startswith(b'img:'):
                    nparr = np.frombuffer(message[4:], np.uint8)
                    self.last_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    self.socket.send(b"Image Received")
                else:
                    json_data = json.loads(message.decode('utf-8'))
                    self.messages.put(json_data)
                    self.socket.send(b"Received")
            except zmq.Again:
                continue

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.socket.close()
        self.context.term()

    def get_messages(self, remove=True):
        messages = []
        while not self.messages.empty():
            messages.append(self.messages.get())
            if remove:
                self.messages.task_done()
        return messages

    def get_img(self):
        return self.last_image


class ZMQSender:
    def __init__(self, ip_receiver='localhost', port_receiver=5555, timeout=2, logger=None):
        self.address = f"tcp://{ip_receiver}:{port_receiver}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.address)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout * 1000)
        self.logger = logger if logger else LogPrint()

        # Default encoding properties
        self.format = '.jpg'  # Image format
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # Encoding parameters

    def configure_image_encoding(self, format='.jpg', **params):
        self.format = format
        self.encode_params = []
        for key, value in params.items():
            if key == 'jpeg_quality':
                self.encode_params.extend([cv2.IMWRITE_JPEG_QUALITY, value])
            elif key == 'png_compression':
                self.encode_params.extend([cv2.IMWRITE_PNG_COMPRESSION, value])
            elif key == 'webp_quality':
                self.encode_params.extend([cv2.IMWRITE_WEBP_QUALITY, value])

    def send_json(self, data):
        try:
            json_data = json.dumps(data).encode('utf-8')
            self.socket.send(json_data)
            message = self.socket.recv()
            return message.decode('utf-8')
        except zmq.Again:
            return "Timeout"

    def send_img(self, img):
        _, buffer = cv2.imencode(self.format, img, self.encode_params)
        try:
            self.socket.send(b'img:' + buffer.tobytes())
            message = self.socket.recv()
            return message.decode('utf-8')
        except zmq.Again:
            return "Timeout"



if __name__ == "__main__":
    # Example usage
    receiver = ZMQReceiver(ip_receiver='127.0.0.1', port_receiver=5556)
    sender = ZMQSender(ip_receiver='127.0.0.1', port_receiver=5556)

    # Send JSON
    reply = sender.send_json({"message": "Hello, Server!", 'bobo': 'huhu'})
    print(f"Received reply: {reply}")



    # Receive JSON
    msgs = receiver.get_messages()
    for msg in msgs:
        for field, payload in msg.items():
            print(f"Field: {field}, Payload: {payload}")

    # Receive Image
    import matplotlib.pyplot as plt
    for i in range(10):
        # Send Image
        test_image = np.random.randint(0, 256, (800, 800, 3), dtype=np.uint8)
        img_reply = sender.send_img(test_image)
        print(f"Received image reply: {img_reply}")
        received_image = receiver.get_img()
        plt.imshow(received_image)
        plt.show()
        
