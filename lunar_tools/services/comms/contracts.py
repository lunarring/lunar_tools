from __future__ import annotations

from typing import Optional, Protocol, Sequence


class MessageSenderPort(Protocol):
    """Port responsible for pushing messages to a transport."""

    def send(self, address: str, payload: Sequence[float] | bytes | str) -> None:
        ...


class MessageReceiverPort(Protocol):
    """Port that can receive or poll messages from a transport."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def receive(
        self,
        address: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Sequence[float] | bytes | str]:
        ...


__all__ = ["MessageSenderPort", "MessageReceiverPort"]
