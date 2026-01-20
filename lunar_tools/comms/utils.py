import json
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


def get_local_ip():
    """
    Determine the local IP address on Ubuntu/Linux systems.

    Returns:
        str | None: The detected IP address or None if nothing could be found.
    """
    # Method 1: Use "ip route get" to infer the source address.
    try:
        result = subprocess.run(
            ["ip", "route", "get", "1.1.1.1"],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"\bsrc\s+(\d+\.\d+\.\d+\.\d+)", result.stdout)
        if match:
            local_ip = match.group(1)
            if not local_ip.startswith("127."):
                return local_ip
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 2: Parse ifconfig output (most accurate for Linux)
    try:
        result = subprocess.run(["ifconfig"], capture_output=True, text=True, check=True)
        output = result.stdout

        # Split by interface blocks (starts with interface name + colon)
        interface_blocks = re.split(r"\n(?=\w+:)", output)

        candidate_ips = []

        for block in interface_blocks:
            if not block.strip():
                continue

            # Check if interface is UP and RUNNING (active interface)
            if "UP" in block and "RUNNING" in block:
                # Find inet addresses in this interface block
                inet_pattern = r"inet (\d+\.\d+\.\d+\.\d+)"
                inet_matches = re.findall(inet_pattern, block)

                for ip in inet_matches:
                    # Skip localhost
                    if ip.startswith("127."):
                        continue

                    # Prioritize common private network ranges
                    if ip.startswith("10."):
                        candidate_ips.insert(0, ip)  # Highest priority
                    elif ip.startswith("192.168."):
                        candidate_ips.append(ip)  # Medium priority
                    elif ip.startswith("172.") and 16 <= int(ip.split(".")[1]) <= 31:
                        candidate_ips.append(ip)  # Medium priority
                    else:
                        candidate_ips.append(ip)  # Lowest priority

        if candidate_ips:
            return candidate_ips[0]

    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        pass

    # Method 3: Socket-based fallback (works on most systems)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]

            if not local_ip.startswith("127."):
                return local_ip

    except (socket.error, OSError):
        pass

    # Method 4: Last resort - get hostname IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith("127."):
            return local_ip
    except (socket.error, OSError):
        pass

    return None


WEBRTC_SESSION_CACHE_PATH = Path.home() / ".lunar_tools" / "webrtc_sessions.json"


def _load_session_cache() -> Dict[str, Dict[str, object]]:
    if not WEBRTC_SESSION_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(WEBRTC_SESSION_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data  # type: ignore[return-value]


def cache_webrtc_session_endpoint(session_id: str, host: str, port: int) -> Path:
    cache = _load_session_cache()
    cache[session_id] = {"host": host, "port": int(port), "updated": time.time()}
    WEBRTC_SESSION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEBRTC_SESSION_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return WEBRTC_SESSION_CACHE_PATH


def get_cached_webrtc_session_endpoint(session_id: str) -> Optional[Tuple[str, int]]:
    cache = _load_session_cache()
    entry = cache.get(session_id)
    if not isinstance(entry, dict):
        return None
    host = entry.get("host")
    port = entry.get("port")
    if not isinstance(host, str):
        return None
    if not isinstance(port, int):
        try:
            port = int(port)  # type: ignore[assignment]
        except (TypeError, ValueError):
            return None
    return host, int(port)


__all__ = [
    "get_local_ip",
    "WEBRTC_SESSION_CACHE_PATH",
    "cache_webrtc_session_endpoint",
    "get_cached_webrtc_session_endpoint",
]
