from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

MessagePayload = Sequence[float] | bytes | str | dict[str, Any] | Any


class MessageSenderPort(Protocol):
    """Port responsible for pushing messages to a transport."""

    def send(self, address: str, payload: MessagePayload) -> None:
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
    ) -> Optional[MessagePayload]:
        ...


__all__ = ["MessagePayload", "MessageSenderPort", "MessageReceiverPort"]
