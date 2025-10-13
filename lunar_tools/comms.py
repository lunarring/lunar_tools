"""Communication utilities for lunar_tools.

This module re-exports ZeroMQ and OSC adapters and retains helper utilities
such as ``get_local_ip`` for backwards compatibility.
"""

from __future__ import annotations

import subprocess
import re
import socket

from lunar_tools.adapters.comms.osc_endpoints import OSCReceiver, OSCSender
from lunar_tools.adapters.comms.zmq_pair import ZMQPairEndpoint


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


__all__ = [
    "ZMQPairEndpoint",
    "OSCSender",
    "OSCReceiver",
    "get_local_ip",
]
