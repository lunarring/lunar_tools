from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

from lunar_tools.comms import CommunicationServices, MessageBusConfig, create_message_bus
from lunar_tools.image_gen import ImageGenerators, create_image_generators
from lunar_tools.platform.logging import create_logger
from lunar_tools.presentation.display_window import GridRenderer, Renderer

RenderType = Union[Renderer, GridRenderer]


@dataclass
class DisplayStackConfig:
    """
    Configuration for bootstrapping a display pipeline.

    Attributes:
        width: Window width used when ``use_grid`` is False.
        height: Window height used when ``use_grid`` is False.
        backend: Renderer backend (``"gl"``, ``"opencv"``, ``"pygame"``, etc.).
        window_title: Title shown on the renderer window.
        use_grid: When True, create a ``GridRenderer`` instead of a basic ``Renderer``.
        grid_rows: Number of rows for the grid renderer.
        grid_cols: Number of columns for the grid renderer.
        tile_shape_hw: Height/width of each tile. Defaults to evenly splitting the
            window dimensions across rows/columns when not provided.
        attach_message_bus: When True, initialise the communications message bus.
        message_bus_config: Optional override for the message bus configuration.
        include_image_generators: When True, attach the vision service registry so
            presentation code can source frames from model providers.
        include_glif: Forwarded to :func:`create_image_generators` to control GLIF.
    """

    width: int = 1280
    height: int = 720
    backend: Optional[str] = None
    window_title: str = "Lunar Display"

    use_grid: bool = False
    grid_rows: int = 1
    grid_cols: int = 1
    tile_shape_hw: Optional[Tuple[int, int]] = None

    attach_message_bus: bool = False
    message_bus_config: Optional[MessageBusConfig] = None

    include_image_generators: bool = False
    include_glif: bool = True


@dataclass
class DisplayStack:
    """
    Bundle of renderer, optional communications services, and optional generators.
    """

    renderer: RenderType
    communication: Optional[CommunicationServices] = None
    generators: Optional[ImageGenerators] = None

    def close(self) -> None:
        """Attempt to gracefully stop attached services."""
        logger = create_logger(__name__ + ".DisplayStack")
        if self.communication:
            try:
                self.communication.message_bus.stop_all()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to stop message bus cleanly: %s", exc)
            for endpoint_name in ("osc_receiver", "zmq_endpoint"):
                endpoint = getattr(self.communication, endpoint_name, None)
                if endpoint:
                    try:
                        endpoint.stop()
                    except Exception as exc:  # pragma: no cover
                        logger.warning("Failed to stop %s: %s", endpoint_name, exc)


def _resolve_tile_shape(
    *,
    config: DisplayStackConfig,
) -> Tuple[int, int]:
    if config.tile_shape_hw:
        return config.tile_shape_hw
    height = max(1, config.height // max(1, config.grid_rows))
    width = max(1, config.width // max(1, config.grid_cols))
    return height, width


def _create_renderer(config: DisplayStackConfig) -> RenderType:
    if config.use_grid:
        tile_shape = _resolve_tile_shape(config=config)
        return GridRenderer(
            config.grid_rows,
            config.grid_cols,
            tile_shape,
            backend=config.backend,
            window_title=config.window_title,
        )
    return Renderer(
        width=config.width,
        height=config.height,
        backend=config.backend,
        window_title=config.window_title,
    )


def bootstrap_display_stack(
    config: Optional[DisplayStackConfig] = None,
) -> DisplayStack:
    """
    Construct the display stack, wiring in optional communication and vision services.
    """
    cfg = config or DisplayStackConfig()

    communication: Optional[CommunicationServices] = None
    if cfg.attach_message_bus:
        communication = create_message_bus(cfg.message_bus_config)

    generators: Optional[ImageGenerators] = None
    if cfg.include_image_generators:
        generators = create_image_generators(include_glif=cfg.include_glif)

    renderer = _create_renderer(cfg)

    return DisplayStack(
        renderer=renderer,
        communication=communication,
        generators=generators,
    )


__all__ = [
    "DisplayStack",
    "DisplayStackConfig",
    "bootstrap_display_stack",
]
