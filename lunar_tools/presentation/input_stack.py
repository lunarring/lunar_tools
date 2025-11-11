from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from lunar_tools.comms import CommunicationServices, MessageBusConfig, create_message_bus
from lunar_tools.platform.logging import create_logger
from lunar_tools.presentation.control_input import KeyboardInput, MetaInput


@dataclass
class ControlInputStackConfig:
    """
    Configuration for bootstrapping control input helpers.

    Attributes:
        use_meta: When True, create ``MetaInput`` (auto-detects MIDI and falls back
            to keyboard). When False, instantiate ``KeyboardInput`` directly.
        force_device: Override for the device used by ``MetaInput`` (mirrors the
            legacy ``force_device`` argument).
        attach_message_bus: When True, initialise the communications message bus so
            control values can be broadcast.
        message_bus_config: Optional settings passed to :func:`create_message_bus`.
        broadcast_address: Default address used when broadcasting control payloads.
    """

    use_meta: bool = True
    force_device: Optional[str] = None
    attach_message_bus: bool = False
    message_bus_config: Optional[MessageBusConfig] = None
    broadcast_address: str = "/lunar/controls"


@dataclass
class ControlInputStack:
    """
    Bundle of control input device and optional communications services.
    """

    controller: MetaInput | KeyboardInput
    communication: Optional[CommunicationServices] = None
    broadcast_address: Optional[str] = None

    @property
    def device_name(self) -> str:
        return getattr(self.controller, "device_name", "keyboard")

    def get(self, **kwargs: Any) -> Any:
        """Proxy to the underlying controller's ``get`` method."""
        return self.controller.get(**kwargs)

    def poll_and_broadcast(
        self,
        controls: Dict[str, Dict[str, Any]],
        *,
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch a dictionary of control values and broadcast them via the message bus.

        Args:
            controls: Mapping of logical control names to keyword arguments passed
                to :meth:`controller.get`.
            address: Optional override for the broadcast address. Defaults to
                ``broadcast_address`` from the configuration.
        """
        values: Dict[str, Any] = {}
        for name, kwargs in controls.items():
            values[name] = self.controller.get(**kwargs)

        if self.communication:
            payload = {
                "device": self.device_name,
                "values": values,
            }
            target_address = address or self.broadcast_address
            try:
                self.communication.message_bus.broadcast(payload, address=target_address)
            except Exception as exc:  # pragma: no cover - defensive emission
                logger = create_logger(__name__ + ".ControlInputStack")
                logger.warning("Failed to broadcast control payload: %s", exc)

        return values

    def close(self) -> None:
        """Stop communication services initialised for the control stack."""
        if not self.communication:
            return
        logger = create_logger(__name__ + ".ControlInputStack")
        try:
            self.communication.message_bus.stop_all()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to stop message bus cleanly: %s", exc)
        for endpoint_name in ("osc_receiver", "zmq_endpoint"):
            endpoint = getattr(self.communication, endpoint_name, None)
            if endpoint:
                try:
                    endpoint.stop()
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to stop %s: %s", endpoint_name, exc)


def bootstrap_control_inputs(
    config: Optional[ControlInputStackConfig] = None,
) -> ControlInputStack:
    """
    Instantiate control input helpers aligned with the modernised service layout.
    """
    cfg = config or ControlInputStackConfig()

    if cfg.use_meta:
        controller: MetaInput | KeyboardInput = MetaInput(force_device=cfg.force_device)
    else:
        controller = KeyboardInput()

    communication: Optional[CommunicationServices] = None
    if cfg.attach_message_bus:
        communication = create_message_bus(cfg.message_bus_config)

    return ControlInputStack(
        controller=controller,
        communication=communication,
        broadcast_address=cfg.broadcast_address,
    )


__all__ = [
    "ControlInputStack",
    "ControlInputStackConfig",
    "bootstrap_control_inputs",
]
