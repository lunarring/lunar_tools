from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from lunar_tools._optional import require_extra
from lunar_tools.platform.logging import create_logger

try:  # pragma: no cover - optional dependency guard
    from pythonosc import osc_server, udp_client
    from pythonosc.dispatcher import Dispatcher
except ImportError:  # pragma: no cover - optional dependency
    require_extra("OSC communication", extras="comms")

osc_server.ThreadingOSCUDPServer.allow_reuse_address = True


class OSCSender:
    def __init__(
        self,
        ip_receiver: str,
        port_receiver: int = 8003,
        start_thread: bool = False,
        verbose_high: bool = False,
    ) -> None:
        self.ip_receiver = ip_receiver
        self.port_receiver = port_receiver
        self.client = udp_client.SimpleUDPClient(self.ip_receiver, self.port_receiver)
        self.DELIM = " "
        self.verbose_high = verbose_high

    def send_message(self, identifier: str, message) -> None:
        self.client.send_message(identifier, message)
        if self.verbose_high:
            print(f"OSCSender: {identifier} {message} to {self.ip_receiver}:{self.port_receiver}")


class OSCReceiver:
    def __init__(
        self,
        ip_receiver: str,
        start: bool = True,
        BUFFER_SIZE: int = 500,
        rescale_all_input: bool = False,
        dt_timeout: int = 3,
        port_receiver: int = 8003,
        verbose_high: bool = False,
    ) -> None:
        self.ip_receiver = ip_receiver
        self.port_receiver = port_receiver
        self.rescale_all_input = rescale_all_input
        self.thread_osc = Thread(target=self.runfunc_thread_osc)
        self.running = False

        self.dict_messages: Dict[str, List[float]] = {}
        self.dict_time: Dict[str, List[float]] = {}
        self.dt_timeout = dt_timeout
        self.BUFFER_SIZE = BUFFER_SIZE
        self.verbose_high = verbose_high
        self.filter_identifiers: List[str] = []

        if start:
            self.start()

    def runfunc_thread_osc(self) -> None:
        self.running = True
        dispatcher = Dispatcher()
        dispatcher.map("/*", self.process_incoming)
        self.server = osc_server.ThreadingOSCUDPServer((self.ip_receiver, self.port_receiver), dispatcher)
        print(f"Serving on {self.server.server_address}")
        self.server.serve_forever()

    def start(self) -> None:
        self.thread_osc.start()

    def stop(self) -> None:
        self.running = False
        try:
            if hasattr(self, "server") and self.server:
                self.server.shutdown()
                self.server.server_close()
        except Exception:
            pass
        if self.thread_osc.is_alive():
            self.thread_osc.join(timeout=1.0)

    def process_incoming(self, *args) -> None:
        identifier = args[0]
        message = args[1]

        if identifier not in self.dict_messages:
            self.dict_messages[identifier] = []

        if identifier not in self.dict_time:
            self.dict_time[identifier] = []

        if len(self.dict_messages[identifier]) >= self.BUFFER_SIZE:
            self.dict_messages[identifier].pop(0)
            self.dict_time[identifier].pop(0)

        try:
            message = float(message)
        except ValueError:
            print(f"Received non-numerical message: {message}")
            return
        self.dict_messages[identifier].append(message)
        self.dict_time[identifier].append(time.time())

        if self.verbose_high:
            print(f"OSCReceiver: {identifier} {message} from {self.ip_receiver}:{self.port_receiver}")

    def get_last_value(self, identifier: str, val_min: float = 0.0, val_max: float = 1.0, val_default: float = 0.0) -> float:
        if identifier in self.dict_messages:
            if len(self.dict_messages[identifier]) >= 1:
                value = self.dict_messages[identifier][-1]
                if self.rescale_all_input:
                    minval = min(self.dict_messages[identifier])
                    maxval = max(self.dict_messages[identifier])
                    if maxval - minval == 0:
                        fract = 1
                    else:
                        fract = (value - minval) / (maxval - minval)
                        fract = max(0, min(fract, 1))
                    value = val_min + (val_max - val_min) * fract
                return value
        if self.verbose_high:
            print(f"ERROR get_last_value: identifier {identifier} was never received!")
        return val_default

    def get_all_values(self, identifier: str):
        if identifier in self.dict_messages:
            if len(self.dict_messages[identifier]) >= 1:
                return self.dict_messages[identifier]
        return None

    def show_last_received(self) -> None:
        if not self.dict_time:
            print("Nothing was received since you started the receiver.")
            return
        current_time = time.time()
        for identifier, timestamp_all in self.dict_time.items():
            time_since_received = current_time - timestamp_all[-1]
            print(f"Signal '{identifier}' was last received {time_since_received:.2f} seconds ago.")


class OSCMessageSender:
    """
    Thin wrapper around python-osc used by the message bus service.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        logger=None,
    ) -> None:
        self._client = udp_client.SimpleUDPClient(host, port)
        self._logger = logger if logger else create_logger(__name__ + ".sender")

    def send(self, address: str, payload: Sequence[float] | bytes | str | Any) -> None:
        self._client.send_message(address, payload)
        if self._logger.isEnabledFor(10):  # DEBUG
            self._logger.debug("Sent OSC message %s -> %s", address, payload)


class OSCMessageReceiver:
    """
    Background OSC server that buffers incoming messages for the communications service.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_queue_size: int = 512,
        logger=None,
    ) -> None:
        self._host = host
        self._port = port
        self._max_queue_size = max_queue_size
        self._logger = logger if logger else create_logger(__name__ + ".receiver")

        self._dispatcher = Dispatcher()
        self._dispatcher.set_default_handler(self._handle_message)

        self._server: Optional[osc_server.ThreadingOSCUDPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._messages: List[dict[str, Any]] = []
        self._condition = threading.Condition()

    # Lifecycle --------------------------------------------------------
    def _ensure_server(self) -> None:
        if self._server is None:
            self._server = osc_server.ThreadingOSCUDPServer((self._host, self._port), self._dispatcher)

    def start(self) -> None:
        if self._running:
            return
        self._ensure_server()
        assert self._server is not None  # for type checkers
        self._running = True

        def _serve() -> None:
            try:
                self._logger.debug("Starting OSC server on %s:%s", self._host, self._port)
                self._server.serve_forever()
            except Exception as exc:  # pragma: no cover - logging side effect
                self._logger.error("OSC server thread exited with error: %s", exc)
            finally:
                self._logger.debug("OSC server thread finished")

        self._thread = threading.Thread(target=_serve, name="OSCMessageReceiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:  # pragma: no cover - defensive shutdown
                pass
            finally:
                self._server = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        with self._condition:
            self._messages.clear()

    # Message handling -------------------------------------------------
    def _handle_message(self, address: str, *args: Any) -> None:
        payload: Any
        if not args:
            payload = None
        elif len(args) == 1:
            payload = args[0]
        else:
            payload = list(args)

        message = {"address": address, "payload": payload}
        with self._condition:
            if len(self._messages) >= self._max_queue_size:
                self._messages.pop(0)
            self._messages.append(message)
            self._condition.notify_all()

    def receive(
        self,
        address: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
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

    def _pop_message(self, address: Optional[str]) -> Optional[dict[str, Any]]:
        if not self._messages:
            return None
        if address is None:
            return self._messages.pop(0)

        for idx, message in enumerate(self._messages):
            if message.get("address") == address:
                return self._messages.pop(idx)
        return None
