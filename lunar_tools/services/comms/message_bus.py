from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from .contracts import MessagePayload, MessageReceiverPort, MessageSenderPort


@dataclass
class _SenderRegistration:
    port: MessageSenderPort
    default_address: Optional[str] = None


@dataclass
class _ReceiverRegistration:
    port: MessageReceiverPort
    default_address: Optional[str] = None
    auto_started: bool = False


class MessageBusService:
    """
    Coordinate message senders and receivers behind the communications contracts.

    The service handles lifecycle management (start/stop) for receivers and
    provides helpers to send or broadcast messages without leaking adapter
    details to calling code.
    """

    def __init__(self) -> None:
        self._senders: Dict[str, _SenderRegistration] = {}
        self._receivers: Dict[str, _ReceiverRegistration] = {}

    # Sender management -------------------------------------------------
    def register_sender(
        self,
        name: str,
        sender: MessageSenderPort,
        *,
        default_address: Optional[str] = None,
    ) -> None:
        """Register a sender by name."""
        self._senders[name] = _SenderRegistration(
            port=sender,
            default_address=default_address,
        )

    def send(
        self,
        name: str,
        payload: MessagePayload,
        *,
        address: Optional[str] = None,
    ) -> None:
        """Send a payload via the named sender."""
        try:
            registration = self._senders[name]
        except KeyError as exc:
            raise KeyError(f"Sender {name!r} is not registered") from exc

        target_address = address or registration.default_address
        if target_address is None:
            raise ValueError(f"No address provided for sender {name!r}")
        registration.port.send(target_address, payload)

    def broadcast(
        self,
        payload: MessagePayload,
        *,
        address: Optional[str] = None,
    ) -> None:
        """Send the payload across all registered senders."""
        errors: Dict[str, Exception] = {}
        for name, registration in self._senders.items():
            target_address = address or registration.default_address
            if target_address is None:
                errors[name] = ValueError(f"No address available for sender {name!r}")
                continue
            try:
                registration.port.send(target_address, payload)
            except Exception as exc:  # pragma: no cover - error aggregation
                errors[name] = exc
        if errors:
            messages = ", ".join(f"{name}: {exc}" for name, exc in errors.items())
            raise RuntimeError(f"One or more senders failed: {messages}")

    # Receiver management ------------------------------------------------
    def register_receiver(
        self,
        name: str,
        receiver: MessageReceiverPort,
        *,
        default_address: Optional[str] = None,
        auto_start: bool = True,
    ) -> None:
        """Register a receiver and optionally start it immediately."""
        registration = _ReceiverRegistration(
            port=receiver,
            default_address=default_address,
            auto_started=False,
        )
        self._receivers[name] = registration
        if auto_start:
            receiver.start()
            registration.auto_started = True

    def start_receiver(self, name: str) -> None:
        """Start a receiver that was registered without auto_start."""
        try:
            registration = self._receivers[name]
        except KeyError as exc:
            raise KeyError(f"Receiver {name!r} is not registered") from exc
        if not registration.auto_started:
            registration.port.start()
            registration.auto_started = True

    def start_all(self) -> None:
        """Start every registered receiver."""
        for registration in self._receivers.values():
            if not registration.auto_started:
                registration.port.start()
                registration.auto_started = True

    def stop_receiver(self, name: str) -> None:
        """Stop a receiver and mark it as no longer running."""
        try:
            registration = self._receivers[name]
        except KeyError as exc:
            raise KeyError(f"Receiver {name!r} is not registered") from exc
        registration.port.stop()
        registration.auto_started = False

    def stop_all(self) -> None:
        """Stop every registered receiver."""
        for registration in self._receivers.values():
            registration.port.stop()
            registration.auto_started = False

    def poll(
        self,
        name: str,
        *,
        address: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Optional[MessagePayload]:
        """Poll the named receiver for a payload."""
        try:
            registration = self._receivers[name]
        except KeyError as exc:
            raise KeyError(f"Receiver {name!r} is not registered") from exc

        target_address = address or registration.default_address
        return registration.port.receive(
            address=target_address,
            timeout=timeout,
        )

    def wait_for(
        self,
        name: str,
        *,
        address: Optional[str] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 0.05,
    ) -> Optional[MessagePayload]:
        """
        Poll the receiver until a payload is returned or the timeout elapses.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            payload = self.poll(
                name,
                address=address,
                timeout=poll_interval,
            )
            if payload is not None:
                return payload

            if deadline is not None and time.monotonic() >= deadline:
                return None


__all__ = ["MessageBusService"]
