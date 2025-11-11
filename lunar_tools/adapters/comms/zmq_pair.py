from __future__ import annotations

import json
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional

from lunar_tools._optional import require_extra
from lunar_tools.platform.logging import create_logger

try:  # pragma: no cover - optional dependency guard
    import cv2
    import numpy as np
    import zmq
except ImportError:  # pragma: no cover - optional dependency
    require_extra("ZeroMQ communication", extras="comms")


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


class ZMQMessageEndpoint:
    """
    Lightweight PAIR socket wrapper implementing the message bus contracts.
    """

    def __init__(
        self,
        *,
        bind: bool,
        host: str = "127.0.0.1",
        port: int = 5556,
        context: Optional["zmq.Context"] = None,
        poll_interval: float = 0.05,
        logger=None,
    ) -> None:
        self._host = host
        self._port = port
        self._bind = bind
        self._poll_interval_ms = max(int(poll_interval * 1000), 1)
        self._context = context or zmq.Context.instance()
        self._logger = logger if logger else create_logger(__name__ + ".endpoint")

        self._socket: Optional["zmq.Socket"] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._messages: List[Dict[str, Any]] = []
        self._condition = threading.Condition()

        self._ensure_socket()

    # Socket lifecycle -------------------------------------------------
    def _address(self) -> str:
        return f"tcp://{self._host}:{self._port}"

    def _ensure_socket(self) -> None:
        if self._socket is not None:
            return
        socket = self._context.socket(zmq.PAIR)
        socket.setsockopt(zmq.LINGER, 0)
        address = self._address()
        if self._bind:
            socket.bind(address)
        else:
            socket.connect(address)
        self._socket = socket

    def start(self) -> None:
        if self._running:
            return
        self._ensure_socket()
        if self._socket is None:
            raise RuntimeError("ZMQ socket is unavailable")
        self._running = True
        self._thread = threading.Thread(target=self._run, name="ZMQMessageEndpoint", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        socket = self._socket
        self._socket = None

        if socket is not None:
            try:
                socket.close(linger=0)
            except Exception:  # pragma: no cover - defensive shutdown
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        with self._condition:
            self._messages.clear()

    # Sending ----------------------------------------------------------
    def send(self, address: str, payload: Any) -> None:
        self._ensure_socket()
        if self._socket is None:
            raise RuntimeError("ZMQ socket is unavailable")
        envelope = {"address": address, "payload": payload}
        self._socket.send_pyobj(envelope)

    # Receiving --------------------------------------------------------
    def _run(self) -> None:
        socket = self._socket
        if socket is None:
            return
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        while self._running:
            try:
                events = dict(poller.poll(self._poll_interval_ms))
            except zmq.ZMQError as exc:  # pragma: no cover - poll interruption
                if self._running:
                    self._logger.error("ZMQ poll error: %s", exc)
                break

            if not self._running:
                break

            if socket in events and events[socket] & zmq.POLLIN:
                try:
                    message = socket.recv_pyobj(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except zmq.ZMQError as exc:  # pragma: no cover - recv error
                    self._logger.error("ZMQ recv error: %s", exc)
                    break
                with self._condition:
                    self._messages.append(message)
                    self._condition.notify_all()

        poller.unregister(socket)

    def receive(
        self,
        address: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            message = self._pop_message(address)
            while message is None:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                else:
                    remaining = None
                self._condition.wait(timeout=remaining)
                message = self._pop_message(address)
            return message

    def _pop_message(self, address: Optional[str]) -> Optional[Dict[str, Any]]:
        if not self._messages:
            return None
        if address is None:
            return self._messages.pop(0)
        for idx, message in enumerate(self._messages):
            if message.get("address") == address:
                return self._messages.pop(idx)
        return None
