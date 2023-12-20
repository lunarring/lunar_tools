import zmq
import json
import threading
import cv2
import numpy as np
from queue import Queue
from lunar_tools.logprint import LogPrint

class ZMQPairEndpoint:
    def __init__(self, is_server, ip='localhost', port='5555', timeout=2, logger=None):
        self.address = f"tcp://{ip}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.messages = Queue()
        self.last_image = None
        self.logger = logger if logger else LogPrint()
        self.running = False

        if is_server:
            self.socket.bind(self.address)
        else:
            self.socket.connect(self.address)
            self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
            self.socket.setsockopt(zmq.SNDTIMEO, timeout * 1000)

        # Default encoding properties
        self.format = '.jpg'
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

        # Start listening thread
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.start()

    def start(self):
        self.thread.start()

    def listen(self):
        self.running = True
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            # Wait for a message for a short time
            socks = dict(poller.poll(1000))  # timeout in milliseconds
            if self.socket in socks:
                try:
                    message = self.socket.recv(zmq.NOBLOCK)
                    if message.startswith(b'img:'):
                        nparr = np.frombuffer(message[4:], np.uint8)
                        self.last_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        self.messages.put(('img', self.last_image))
                    else:
                        json_data = json.loads(message.decode('utf-8'))
                        self.messages.put(('json', json_data))
                    
                    
                except zmq.Again:
                    continue
                except Exception as e:
                    self.logger.error(f"Exception in listen thread: {e}")
                    break
    

    def stop(self):
        # Signal the thread to stop
        self.running = False

        # Wait for the listening thread to finish
        if self.thread.is_alive():
            self.thread.join()

        # Now safely close the socket
        self.socket.close()


    def get_messages(self):
        messages = []
        while not self.messages.empty():
            message_type, message_data = self.messages.get()
            if message_type == 'json':
                messages.append(message_data)
        return messages

    def get_img(self):
        while not self.messages.empty():
            message_type, message_data = self.messages.get()
            if message_type == 'img':
                return message_data
        return None

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
        json_data = json.dumps(data).encode('utf-8')
        self.socket.send(json_data)

    def send_img(self, img):
        _, buffer = cv2.imencode(self.format, img, self.encode_params)
        self.socket.send(b'img:' + buffer.tobytes())

# Example usage
if __name__ == "__main__":
    import numpy as np
    import time

    # Create server and client
    server = ZMQPairEndpoint(is_server=True, ip='127.0.0.1', port='5556')
    client = ZMQPairEndpoint(is_server=False, ip='127.0.0.1', port='5556')
    
    # Client: Send JSON to Server
    client.send_json({"message": "Hello from Client!"})
    time.sleep(0.01)
    # Server: Check for received messages
    server_msgs = server.get_messages()
    print("Messages received by server:", server_msgs)
    
    # Server: Send JSON to Client
    server.send_json({"response": "Hello from Server!"})
    time.sleep(0.01)

    # Client: Check for received messages
    client_msgs = client.get_messages()
    print("Messages received by client:", client_msgs)

    # Bidirectional Image Sending
    sz = (800, 800)
    client_image = np.random.randint(0, 256, (sz[0], sz[1], 3), dtype=np.uint8)
    server_image = np.random.randint(0, 256, (sz[0], sz[1], 3), dtype=np.uint8)

    # Client sends image to Server
    client.send_img(client_image)
    time.sleep(0.01)
    server_received_image = server.get_img()
    if server_received_image is not None:
        print("Server received image from Client")

    # Server sends image to Client
    server.send_img(server_image)
    time.sleep(0.01)
    client_received_image = client.get_img()
    if client_received_image is not None:
        print("Client received image from Server")
