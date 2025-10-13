from __future__ import annotations

import json
import threading
from queue import Empty, Full, Queue
from typing import Optional

import cv2
import numpy as np
import zmq

from lunar_tools.platform.logging import create_logger


class ZMQPairEndpoint:
    def __init__(
        self,
        is_server: bool,
        ip: str = "localhost",
        port: str = "5555",
        timeout: float = 2,
        jpeg_quality: int = 99,
        logger=None,
    ) -> None:
        self.address = f"tcp://{ip}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.messages: Queue = Queue()
        self.send_queue: Queue = Queue()
        self.last_image = None
        self.last_audio = None
        self.logger = logger if logger else create_logger(__name__)
        self.running = False
        timeout_seconds = float(timeout)
        self.timeout_ms = max(int(timeout_seconds * 1000), 1)
        self.poll_interval_ms = max(5, min(self.timeout_ms, 50))
        self._send_block_timeout = timeout_seconds if timeout_seconds > 0 else None

        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

        if is_server:
            self.socket.bind(self.address)
        else:
            self.socket.connect(self.address)

        self.format = ".jpg"
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.start()

    def start(self) -> None:
        self.thread.start()

    def listen(self) -> None:
        self.running = True
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            self._flush_outgoing()
            try:
                socks = dict(poller.poll(self.poll_interval_ms))
            except zmq.ZMQError as exc:
                if self.running:
                    self.logger.error("ZMQ poll error: %s", exc)
                break

            if not self.running:
                break

            if self.socket in socks and socks[self.socket] & zmq.POLLIN:
                try:
                    self._handle_incoming()
                except Exception as exc:  # pragma: no cover - logging side effect
                    self.logger.error("Exception in listen thread: %s", exc)
                    break

            self._flush_outgoing()

        while True:
            try:
                _, response_queue = self.send_queue.get_nowait()
            except Empty:
                break
            if response_queue is not None:
                response_queue.put_nowait(("error", zmq.Again("Endpoint stopped")))

    def _handle_incoming(self) -> None:
        try:
            message = self.socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return

        if message.startswith(b"img:"):
            nparr = np.frombuffer(message[4:], np.uint8)
            self.last_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.messages.put(("img", self.last_image))
        elif message.startswith(b"audio:"):
            header_end = message.find(b":", 6)
            if header_end == -1:
                return
            sample_rate = int(message[6:header_end])

            channels_end = message.find(b":", header_end + 1)
            if channels_end == -1:
                return
            channels = int(message[header_end + 1 : channels_end])

            dtype_end = message.find(b":", channels_end + 1)
            if dtype_end == -1:
                return
            dtype_str = message[channels_end + 1 : dtype_end].decode("utf-8")

            audio_data = message[dtype_end + 1 :]
            audio_array = np.frombuffer(audio_data, dtype=np.dtype(dtype_str))
            if channels == 2:
                audio_array = audio_array.reshape(-1, 2)

            self.last_audio = {
                "data": audio_array,
                "sample_rate": sample_rate,
                "channels": channels,
                "dtype": dtype_str,
            }
            self.messages.put(("audio", self.last_audio))
        else:
            json_data = json.loads(message.decode("utf-8"))
            self.messages.put(("json", json_data))

    def _flush_outgoing(self) -> None:
        while self.running:
            try:
                payload, response_queue = self.send_queue.get_nowait()
            except Empty:
                break

            if payload is None:
                if response_queue is not None:
                    response_queue.put_nowait(("ok", None))
                continue

            try:
                self.socket.send(payload)
            except zmq.Again as exc:
                if response_queue is not None:
                    response_queue.put_nowait(("error", exc))
                else:
                    self.logger.error("Send failed with EAGAIN and no response handle.")
            except zmq.ZMQError as exc:
                if response_queue is not None:
                    response_queue.put_nowait(("error", exc))
                else:
                    self.logger.error("Exception while sending message: %s", exc)
            else:
                if response_queue is not None:
                    response_queue.put_nowait(("ok", None))

    def _queue_send(self, payload) -> None:
        if payload is None:
            self.send_queue.put_nowait((None, None))
            return

        response_queue: Queue = Queue(maxsize=1)
        try:
            if self._send_block_timeout is None:
                self.send_queue.put((payload, response_queue))
            else:
                self.send_queue.put((payload, response_queue), timeout=self._send_block_timeout)
        except Full as exc:
            raise zmq.Again("Send queue full: peer not accepting data before timeout") from exc

        try:
            status, exc = response_queue.get(timeout=self._send_block_timeout or None)
        except Empty as exc:
            raise zmq.Again("Send acknowledgement timed out") from exc

        if status == "error":
            raise exc

    def stop(self) -> None:
        self.running = False
        self._queue_send(None)

        if hasattr(self, "thread") and self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join()

        if hasattr(self, "socket"):
            try:
                self.socket.close(linger=0)
            finally:
                self.socket = None

        if getattr(self, "context", None) is not None:
            try:
                self.context.term()
            finally:
                self.context = None

    def get_messages(self):
        messages = []
        while not self.messages.empty():
            message_type, message_data = self.messages.get()
            if message_type == "json":
                messages.append(message_data)
        return messages

    def get_img(self):
        while not self.messages.empty():
            message_type, message_data = self.messages.get()
            if message_type == "img":
                return message_data
        return None

    def get_audio(self):
        while not self.messages.empty():
            message_type, message_data = self.messages.get()
            if message_type == "audio":
                return message_data
        return None

    def configure_image_encoding(self, format=".jpg", **params) -> None:
        self.format = format
        self.encode_params = []
        for key, value in params.items():
            if key == "jpeg_quality":
                self.encode_params.extend([cv2.IMWRITE_JPEG_QUALITY, value])
            elif key == "png_compression":
                self.encode_params.extend([cv2.IMWRITE_PNG_COMPRESSION, value])
            elif key == "webp_quality":
                self.encode_params.extend([cv2.IMWRITE_WEBP_QUALITY, value])

    def send_json(self, data) -> None:
        json_data = json.dumps(data).encode("utf-8")
        self._queue_send(json_data)

    def send_img(self, img) -> None:
        success, buffer = cv2.imencode(self.format, img, self.encode_params)
        if not success:
            raise ValueError("Failed to encode image for transmission")
        self._queue_send(b"img:" + buffer.tobytes())

    def send_audio(self, audio_data, sample_rate, channels=None) -> None:
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
        header = f"audio:{sample_rate}:{channels}:{dtype_str}:".encode("utf-8")
        message = header + audio_data.tobytes()
        self._queue_send(message)
