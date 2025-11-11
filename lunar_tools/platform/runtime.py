"""Runtime utilities for platform-layer consumers."""

from __future__ import annotations

from collections.abc import Callable
from threading import Thread
from typing import Any

import platform
import time


def get_os_type() -> str:
    """Return a simplified operating-system identifier."""
    os_name = platform.system()
    if os_name == "Darwin":
        return "MacOS"
    if os_name == "Linux":
        return "Linux"
    if os_name == "Windows":
        return "Windows"
    raise ValueError("unsupported OS")


def get_time_ms() -> int:
    """Return the current wall-clock time in milliseconds."""
    return int(round(time.time() * 1000))


class MultiThreader:
    """Lightweight helper to run a function periodically on a background thread."""

    def __init__(self, runfunc: Callable[[], Any], sleeptime: float = 0.5) -> None:
        self._runfunc = runfunc
        self._sleeptime = sleeptime
        self._running = True
        self._last_tick_ms = 0
        self._thread = Thread(target=self._thread_loop, daemon=True)
        self._thread.start()

    def _thread_loop(self) -> None:
        refresh_interval_ms = 10
        while self._running:
            now_ms = get_time_ms()
            if now_ms - self._last_tick_ms > refresh_interval_ms:
                self._last_tick_ms = now_ms
                self._runfunc()
            time.sleep(self._sleeptime)

    def stop(self) -> None:
        """Stop the background thread and wait for it to exit."""
        self._running = False
        self._thread.join()


def exception_handler(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that swallows exceptions and logs a best-effort error message."""

    def wrapper(*args: Any, **kwargs: Any) -> Any | None:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"Exception when running {func.__name__}: {exc}")
            return None

    return wrapper


__all__ = ["get_os_type", "get_time_ms", "MultiThreader", "exception_handler"]
