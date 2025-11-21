import re
import socket
import subprocess


def get_local_ip():
    """
    Determine the local IP address on Ubuntu/Linux systems.

    Returns:
        str | None: The detected IP address or None if nothing could be found.
    """
    # Method 1: Parse ifconfig output (most accurate for Linux)
    try:
        result = subprocess.run(['ifconfig'], capture_output=True, text=True, check=True)
        output = result.stdout

        # Split by interface blocks (starts with interface name + colon)
        interface_blocks = re.split(r'\n(?=\w+:)', output)

        candidate_ips = []

        for block in interface_blocks:
            if not block.strip():
                continue

            # Check if interface is UP and RUNNING (active interface)
            if 'UP' in block and 'RUNNING' in block:
                # Find inet addresses in this interface block
                inet_pattern = r'inet (\d+\.\d+\.\d+\.\d+)'
                inet_matches = re.findall(inet_pattern, block)

                for ip in inet_matches:
                    # Skip localhost
                    if ip.startswith('127.'):
                        continue

                    # Prioritize common private network ranges
                    if ip.startswith('10.'):
                        candidate_ips.insert(0, ip)  # Highest priority
                    elif ip.startswith('192.168.'):
                        candidate_ips.append(ip)  # Medium priority
                    elif ip.startswith('172.') and 16 <= int(ip.split('.')[1]) <= 31:
                        candidate_ips.append(ip)  # Medium priority
                    else:
                        candidate_ips.append(ip)  # Lowest priority

        if candidate_ips:
            return candidate_ips[0]

    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        pass

    # Method 2: Socket-based fallback (works on most systems)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]

            if not local_ip.startswith('127.'):
                return local_ip

    except (socket.error, OSError):
        pass

    # Method 3: Last resort - get hostname IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith('127.'):
            return local_ip
    except (socket.error, OSError):
        pass

    return None


__all__ = ["get_local_ip"]
