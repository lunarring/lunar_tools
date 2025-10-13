from __future__ import annotations

import os
import platform
from typing import Dict


def get_config_path() -> str:
    """
    Determine the path to the lunar_tools configuration file.

    Returns:
        Absolute path to the config file under the user's home directory.
    """
    system = platform.system()
    if system in {"Darwin", "Linux"}:
        return os.path.expanduser("~/.lunar_tools_env_vars")
    if system == "Windows":
        home = os.environ.get("USERPROFILE")
        if not home:
            raise RuntimeError("Unable to determine USERPROFILE on Windows")
        return os.path.join(home, ".lunar_tools_env_vars")
    raise ValueError(f"Unsupported operating system: {system}")


def read_all_api_keys_from_file() -> Dict[str, str]:
    """
    Load all key/value pairs from the config file.

    Returns:
        Dictionary of key/value pairs. Returns empty dict if file does not exist.
    """
    config_path = get_config_path()
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        result: Dict[str, str] = {}
        for line in file:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key] = value
        return result


def read_api_key_from_file(key_name: str) -> str | None:
    """
    Retrieve a specific key from the config file.

    Args:
        key_name: Key to lookup.

    Returns:
        Value if present, else None.
    """
    keys = read_all_api_keys_from_file()
    return keys.get(key_name)


def read_api_key(key_name: str) -> str | None:
    """
    Retrieve API key from environment variables (preferred lookup).
    """
    return os.getenv(key_name)
