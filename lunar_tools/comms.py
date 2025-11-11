"""Communication utilities for lunar_tools.

This module re-exports ZeroMQ and OSC adapters and retains helper utilities
such as ``get_local_ip`` for backwards compatibility.
"""

from __future__ import annotations

import re
import socket
import subprocess
from dataclasses import dataclass
from typing import Optional

from lunar_tools._optional import optional_import_attr
from lunar_tools.services.comms.message_bus import MessageBusService

_OPTIONAL_EXPORTS = {
    "OSCSender": ("lunar_tools.adapters.comms.osc_endpoints", "OSCSender"),
    "OSCReceiver": ("lunar_tools.adapters.comms.osc_endpoints", "OSCReceiver"),
    "ZMQPairEndpoint": ("lunar_tools.adapters.comms.zmq_pair", "ZMQPairEndpoint"),
}


def get_local_ip() -> str | None:
    """Attempt to determine the local IP address."""
    try:
        result = subprocess.run(["ifconfig"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        result = None

    if result and result.stdout:
        interface_blocks = re.split(r"\n(?=\w+:)", result.stdout)
        candidate_ips = []
        for block in interface_blocks:
            if not block.strip():
                continue
            if "UP" in block and "RUNNING" in block:
                inet_matches = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", block)
                for ip in inet_matches:
                    if ip.startswith("127."):
                        continue
                    if ip.startswith("10."):
                        candidate_ips.insert(0, ip)
                    elif ip.startswith("192.168."):
                        candidate_ips.append(ip)
                    elif ip.startswith("172.") and 16 <= int(ip.split(".")[1]) <= 31:
                        candidate_ips.append(ip)
                    else:
                        candidate_ips.append(ip)
        if candidate_ips:
            return candidate_ips[0]

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            if not local_ip.startswith("127."):
                return local_ip
    except (socket.error, OSError):
        pass

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith("127."):
            return local_ip
    except (socket.error, OSError):
        pass

    return None


@dataclass
class MessageBusConfig:
    """
    Configuration for bootstrapping the communications message bus.

    Attributes:
        osc_host: When set, initialise OSC sender/receiver on this host.
        osc_port: UDP port for OSC communication.
        osc_default_address: Default OSC address used when callers omit one.
        zmq_bind: Whether to bind (`True`) or connect (`False`) the ZeroMQ endpoint.
            Set to ``None`` to skip ZeroMQ entirely.
        zmq_host: Host binding or remote host for ZeroMQ.
        zmq_port: Port for the ZeroMQ PAIR socket.
        zmq_default_address: Default logical address for ZeroMQ messages.
    """

    osc_host: Optional[str] = None
    osc_port: int = 9000
    osc_default_address: Optional[str] = None

    zmq_bind: Optional[bool] = None
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 5556
    zmq_default_address: Optional[str] = None


@dataclass
class CommunicationServices:
    message_bus: MessageBusService
    osc_sender: Optional[object] = None
    osc_receiver: Optional[object] = None
    zmq_endpoint: Optional[object] = None


def create_message_bus(config: Optional[MessageBusConfig] = None) -> CommunicationServices:
    """
    Bootstrap the communications message bus with optional OSC and ZeroMQ endpoints.
    """
    config = config or MessageBusConfig()
    bus = MessageBusService()
    osc_sender = osc_receiver = zmq_endpoint = None

    if config.osc_host:
        OSCMessageSender = optional_import_attr(
            "lunar_tools.adapters.comms.osc_endpoints",
            "OSCMessageSender",
            feature="OSCMessageSender",
            extras="comms",
        )
        OSCMessageReceiver = optional_import_attr(
            "lunar_tools.adapters.comms.osc_endpoints",
            "OSCMessageReceiver",
            feature="OSCMessageReceiver",
            extras="comms",
        )

        osc_sender = OSCMessageSender(config.osc_host, config.osc_port)
        osc_receiver = OSCMessageReceiver(config.osc_host, config.osc_port)
        bus.register_sender("osc", osc_sender, default_address=config.osc_default_address)
        bus.register_receiver(
            "osc",
            osc_receiver,
            default_address=config.osc_default_address,
            auto_start=True,
        )

    if config.zmq_bind is not None:
        ZMQMessageEndpoint = optional_import_attr(
            "lunar_tools.adapters.comms.zmq_pair",
            "ZMQMessageEndpoint",
            feature="ZMQMessageEndpoint",
            extras="comms",
        )

        zmq_endpoint = ZMQMessageEndpoint(
            bind=config.zmq_bind,
            host=config.zmq_host,
            port=config.zmq_port,
        )
        bus.register_sender("zmq", zmq_endpoint, default_address=config.zmq_default_address)
        bus.register_receiver(
            "zmq",
            zmq_endpoint,
            default_address=config.zmq_default_address,
            auto_start=True,
        )

    return CommunicationServices(
        message_bus=bus,
        osc_sender=osc_sender,
        osc_receiver=osc_receiver,
        zmq_endpoint=zmq_endpoint,
    )


__all__ = [
    "get_local_ip",
    "OSCSender",
    "OSCReceiver",
    "ZMQPairEndpoint",
    "CommunicationServices",
    "MessageBusConfig",
    "MessageBusService",
    "create_message_bus",
]


def __getattr__(name: str):
    if name in _OPTIONAL_EXPORTS:
        module, attribute = _OPTIONAL_EXPORTS[name]
        value = optional_import_attr(
            module,
            attribute,
            feature=name,
            extras="comms",
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
