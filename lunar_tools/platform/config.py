from __future__ import annotations

import os


def read_api_key(key_name: str) -> str | None:
    """
    Retrieve API keys directly from environment variables.
    """
    return os.getenv(key_name)


__all__ = ["read_api_key"]
