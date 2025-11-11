from __future__ import annotations

from typing import Any, Dict, List, Optional


class FakeMessageSender:
    def __init__(self) -> None:
        self.sent: List[tuple[str, Any]] = []

    def send(self, address: str, payload: Any) -> None:
        self.sent.append((address, payload))


class FakeMessageReceiver:
    def __init__(self) -> None:
        self._messages: List[Dict[str, Any]] = []
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def receive(self, address: Optional[str] = None, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if not self._messages:
            return None
        if address is None:
            return self._messages.pop(0)
        for idx, message in enumerate(self._messages):
            if message.get("address") == address:
                return self._messages.pop(idx)
        return None

    def inject(self, address: str, payload: Any) -> None:
        self._messages.append({"address": address, "payload": payload})
