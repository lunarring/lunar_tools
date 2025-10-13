from __future__ import annotations

import time
from threading import Thread
from typing import Dict, List

from pythonosc import osc_server, udp_client
from pythonosc.dispatcher import Dispatcher

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
