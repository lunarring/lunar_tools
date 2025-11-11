from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

from lunar_tools.comms import CommunicationServices, MessageBusConfig, create_message_bus
from lunar_tools.image_gen import ImageGenerators, create_image_generators
from lunar_tools.platform.logging import create_logger
from lunar_tools.presentation.movie import MovieSaver, MovieSaverThreaded

MovieWriter = Union[MovieSaver, MovieSaverThreaded]


@dataclass
class MovieStackConfig:
    """
    Configuration for bootstrapping movie writing utilities.

    Attributes:
        output_path: Destination file path.
        fps: Target frames-per-second.
        shape_hw: Height/width of frames; optional for delayed initialisation.
        crf: Constant rate factor for ffmpeg-based encoders.
        codec: Video codec name.
        preset: ffmpeg preset name.
        pix_fmt: Pixel format string.
        threaded: When True, use ``MovieSaverThreaded`` for background writes.
        attach_message_bus: When True, include :func:`create_message_bus`.
        message_bus_config: Optional override for message bus settings.
        include_image_generators: When True, bootstrap vision providers so movie
            pipelines can render frames from prompts.
        include_glif: Forwarded to :func:`create_image_generators`.
    """

    output_path: str
    fps: int = 24
    shape_hw: Optional[Sequence[int]] = None
    crf: int = 21
    codec: str = "libx264"
    preset: str = "fast"
    pix_fmt: str = "yuv420p"
    threaded: bool = False

    attach_message_bus: bool = False
    message_bus_config: Optional[MessageBusConfig] = None

    include_image_generators: bool = False
    include_glif: bool = True


@dataclass
class MovieStack:
    """
    Bundle of movie writer, optional communication services, and optional vision providers.
    """

    writer: MovieWriter
    communication: Optional[CommunicationServices] = None
    generators: Optional[ImageGenerators] = None

    def close(self) -> None:
        """Stop communication endpoints that were bootstrapped for the stack."""
        if not self.communication:
            return
        logger = create_logger(__name__ + ".MovieStack")
        try:
            self.communication.message_bus.stop_all()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.warning("Failed to stop message bus cleanly: %s", exc)
        endpoint = self.communication.zmq_endpoint
        if endpoint:
            try:
                endpoint.stop()
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to stop zmq endpoint: %s", exc)


def _create_writer(cfg: MovieStackConfig) -> MovieWriter:
    if cfg.threaded:
        return MovieSaverThreaded(
            fp_out=cfg.output_path,
            fps=cfg.fps,
            shape_hw=list(cfg.shape_hw) if cfg.shape_hw is not None else None,
            crf=cfg.crf,
            codec=cfg.codec,
            preset=cfg.preset,
            pix_fmt=cfg.pix_fmt,
        )
    return MovieSaver(
        fp_out=cfg.output_path,
        fps=cfg.fps,
        shape_hw=list(cfg.shape_hw) if cfg.shape_hw is not None else None,
        crf=cfg.crf,
        codec=cfg.codec,
        preset=cfg.preset,
        pix_fmt=cfg.pix_fmt,
    )


def bootstrap_movie_stack(config: MovieStackConfig) -> MovieStack:
    """
    Construct a movie writing stack aligned with the service-first architecture.
    """
    communication: Optional[CommunicationServices] = None
    if config.attach_message_bus:
        communication = create_message_bus(config.message_bus_config)

    generators: Optional[ImageGenerators] = None
    if config.include_image_generators:
        generators = create_image_generators(include_glif=config.include_glif)

    writer = _create_writer(config)

    return MovieStack(
        writer=writer,
        communication=communication,
        generators=generators,
    )


__all__ = [
    "MovieStack",
    "MovieStackConfig",
    "bootstrap_movie_stack",
]
