"""Utility helpers to cache signaling endpoints between runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

_SESSION_FILE = Path.home() / ".lunar_tools" / "webrtc_sessions.json"
SESSION_CACHE_PATH = _SESSION_FILE


def _load_store() -> Dict[str, Dict[str, object]]:
    if not _SESSION_FILE.exists():
        return {}
    try:
        data = json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data  # type: ignore[return-value]


def remember_session_endpoint(session_id: str, host: str, port: int) -> None:
    store = _load_store()
    store[session_id] = {
        "host": host,
        "port": int(port),
        "updated": time.time(),
    }
    _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SESSION_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def lookup_session_endpoint(session_id: str) -> Optional[Tuple[str, int]]:
    store = _load_store()
    entry = store.get(session_id)
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


__all__ = ["remember_session_endpoint", "lookup_session_endpoint", "SESSION_CACHE_PATH"]
