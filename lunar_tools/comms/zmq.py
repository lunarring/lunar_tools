import json
import threading
from queue import Queue

import cv2
import numpy as np
import zmq

from lunar_tools.logprint import LogPrint


class ZMQPairEndpoint:
    def __init__(self, is_server, ip='localhost', port='5555', timeout=2, jpeg_quality=99, logger=None):
        self.address = f"tcp://{ip}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.messages = Queue()
        self._img_queue = Queue()
        self._audio_queue = Queue()
        self.last_image = None
        self.last_audio = None
        self.logger = logger if logger else LogPrint()
        self.running = False
        self.timeout_ms = int(timeout * 1000)  # Store timeout in milliseconds for polling

        # Set socket timeouts for both server and client
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

        if is_server:
            self.socket.bind(self.address)
        else:
            self.socket.connect(self.address)

        # Default encoding properties
        self.format = '.jpg'
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

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
            socks = dict(poller.poll(self.timeout_ms))  # timeout in milliseconds
            if self.socket in socks:
                try:
                    message = self.socket.recv(zmq.NOBLOCK)
                    if message.startswith(b'img:'):
                        nparr = np.frombuffer(message[4:], np.uint8)
                        self.last_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        self._img_queue.put(self.last_image)
                        continue
                    elif message.startswith(b'audio:'):
                        header_end = message.find(b':', 6)  # Find end of sample_rate
                        if header_end == -1:
                            continue
                        sample_rate = int(message[6:header_end])

                        channels_end = message.find(b':', header_end + 1)
                        if channels_end == -1:
                            continue
                        channels = int(message[header_end + 1:channels_end])

                        dtype_end = message.find(b':', channels_end + 1)
                        if dtype_end == -1:
                            continue
                        dtype_str = message[channels_end + 1:dtype_end].decode('utf-8')

                        audio_data = message[dtype_end + 1:]
                        audio_array = np.frombuffer(audio_data, dtype=np.dtype(dtype_str))

                        if channels == 2:
                            audio_array = audio_array.reshape(-1, 2)

                        self.last_audio = {
                            'data': audio_array,
                            'sample_rate': sample_rate,
                            'channels': channels,
                            'dtype': dtype_str
                        }
                        self._audio_queue.put(self.last_audio)
                        continue
                    try:
                        json_data = json.loads(message.decode('utf-8'))
                        self.messages.put(('json', json_data))
                    except json.JSONDecodeError:
                        if self.logger:
                            self.logger.error("Received malformed JSON payload; ignoring.")
                        continue

                except zmq.Again:
                    continue
                except Exception as e:
                    self.logger.error(f"Exception in listen thread: {e}")
                    break

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.socket.close()
        self.context.term()

    def get_messages(self):
        messages = []
        while not self.messages.empty():
            message_type, message_data = self.messages.get()
            if message_type == 'json':
                messages.append(message_data)
        return messages

    def get_img(self):
        while not self._img_queue.empty():
            return self._img_queue.get()
        return None

    def get_audio(self):
        """
        Retrieve the most recent audio data from the message queue.
        """
        while not self._audio_queue.empty():
            return self._audio_queue.get()
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

    def send_audio(self, audio_data, sample_rate, channels=None):
        """
        Send audio data over ZMQ without any encoding/compression.
        """
        audio_data = np.asarray(audio_data)

        if channels is None:
            if audio_data.ndim == 1:
                channels = 1
            elif audio_data.ndim == 2:
                channels = audio_data.shape[1]
            else:
                raise ValueError("Audio data must be 1D (mono) or 2D (stereo)")

        if channels not in [1, 2]:
            raise ValueError("Only mono (1) and stereo (2) audio are supported")

        if channels == 2 and audio_data.ndim == 2:
            audio_data = audio_data.flatten()
        elif channels == 1 and audio_data.ndim == 2:
            raise ValueError("Mono audio should be 1D array")

        dtype_str = str(audio_data.dtype)

        header = f"audio:{sample_rate}:{channels}:{dtype_str}:".encode('utf-8')
        message = header + audio_data.tobytes()

        self.socket.send(message)


__all__ = ["ZMQPairEndpoint"]
