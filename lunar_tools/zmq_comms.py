#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import json
import threading
from queue import Queue
from lunar_tools.logprint import LogPrint

class ZMQServer:
    def __init__(self, ip_server="*", port="5555", logger=None, start=True):
        self.address = f"tcp://{ip_server}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.address)
        self.messages = Queue()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.logger = logger if logger else LogPrint()
        self.running = False  # Added to track the running state
        if start:
            self.start()

    def run(self):
        self.running = True  # Mark the server as running
        while self.running:  # Check the running state
            try:
                message = self.socket.recv(zmq.NOBLOCK)  # Non-blocking receive
                json_data = json.loads(message.decode('utf-8'))
                self.messages.put(json_data)
                self.socket.send(b"Received")
            except zmq.Again:
                continue  # Ignore if no message is received

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False  # Set running to False to stop the loop
        self.thread.join()  # Wait for the thread to finish
        self.socket.close()  # Close the socket
        self.context.term()  # Terminate the ZMQ context

    def get_messages(self, remove=True):
        messages = []
        while not self.messages.empty():
            messages.append(self.messages.get())
            if remove:
                self.messages.task_done()
            
        return messages



class ZMQClient:
    def __init__(self, ip_server='localhost', port=5555, timeout=2, logger=None):
        self.address = f"tcp://{ip_server}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.address)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)  # milliseconds
        self.socket.setsockopt(zmq.SNDTIMEO, timeout * 1000)  # milliseconds
        self.logger = logger if logger else LogPrint()

    def send_json(self, data):
        try:
            json_data = json.dumps(data).encode('utf-8')
            self.socket.send(json_data)
            message = self.socket.recv()
            message = message.decode('utf-8')
            return message
        except zmq.Again:
            return "Timeout"



if __name__ == "__main__":
    # Example usage
    # First we launch a server
    # server = ZMQServer(ip_server='127.0.0.1', port=5556)

    # And we launch a client
    client = ZMQClient(ip_server='127.0.0.1', port=5556)
    reply = client.send_json({"message": "Hello, Server!", 'bobo': 'huhu'})
    print(f"Received reply: {reply}")
    
    # On the server, we can get the message
    msgs = server.get_messages()
    # for msg in msgs: # iterating over all messages that were received
    #     for field, payload in msg.items(): # iterating over all fields
    #         print(f"Field: {field}, Payload: {payload}")

    

