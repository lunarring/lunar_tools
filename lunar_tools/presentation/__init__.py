"""Presentation layer: UI components, CLI entry points, and sample applications."""

from .audio_stack import AudioConversationController, AudioStackConfig, bootstrap_audio_stack
from .control_input import KeyboardInput, MetaInput
from .display_window import GridRenderer, Renderer
from .fontrender import PopupInput, add_text_to_image
from .movie import (
    MovieReader,
    MovieSaver,
    MovieSaverThreaded,
    add_sound,
    add_subtitles_to_video,
    concatenate_movies,
    fill_up_frames_linear_interpolation,
    interpolate_between_images,
)
from .realtime_voice import RealTimeVoice

__all__ = [
    "AudioConversationController",
    "AudioStackConfig",
    "KeyboardInput",
    "MetaInput",
    "GridRenderer",
    "Renderer",
    "PopupInput",
    "add_text_to_image",
    "MovieReader",
    "MovieSaver",
    "MovieSaverThreaded",
    "add_sound",
    "add_subtitles_to_video",
    "bootstrap_audio_stack",
    "concatenate_movies",
    "fill_up_frames_linear_interpolation",
    "interpolate_between_images",
    "RealTimeVoice",
]
