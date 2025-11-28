#!/usr/bin/env python3
"""Minimal example showing how to detect the local IP address."""

from lunar_tools.comms import get_local_ip


def main() -> None:
    local_ip = get_local_ip()
    if local_ip:
        print(f"Detected local IP: {local_ip}")
    else:
        print("Unable to determine the local IP address.")


if __name__ == "__main__":
    main()
