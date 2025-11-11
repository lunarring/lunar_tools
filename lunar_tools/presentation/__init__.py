"""Presentation layer: UI components, CLI entry points, and sample applications.

Attributes are loaded lazily so that optional extras (e.g. ``pip install
lunar_tools[display]``) are only required when the corresponding feature is
accessed. Missing extras raise informative errors via the `_optional` helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from lunar_tools._optional import optional_import_attr

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

FeatureInfo = Tuple[str, str, str, Tuple[str, ...]]

_FEATURE_REGISTRY: Dict[str, FeatureInfo] = {
    "AudioConversationController": (
        "lunar_tools.presentation.audio_stack",
        "AudioConversationController",
        "audio presentation helpers",
        ("audio",),
    ),
    "AudioStackConfig": (
        "lunar_tools.presentation.audio_stack",
        "AudioStackConfig",
        "audio presentation helpers",
        ("audio",),
    ),
    "bootstrap_audio_stack": (
        "lunar_tools.presentation.audio_stack",
        "bootstrap_audio_stack",
        "audio presentation helpers",
        ("audio",),
    ),
    "KeyboardInput": (
        "lunar_tools.presentation.control_input",
        "KeyboardInput",
        "input device helpers",
        ("inputs",),
    ),
    "MetaInput": (
        "lunar_tools.presentation.control_input",
        "MetaInput",
        "input device helpers",
        ("inputs",),
    ),
    "GridRenderer": (
        "lunar_tools.presentation.display_window",
        "GridRenderer",
        "display rendering utilities",
        ("display",),
    ),
    "Renderer": (
        "lunar_tools.presentation.display_window",
        "Renderer",
        "display rendering utilities",
        ("display",),
    ),
    "PopupInput": (
        "lunar_tools.presentation.fontrender",
        "PopupInput",
        "font rendering utilities",
        ("display",),
    ),
    "add_text_to_image": (
        "lunar_tools.presentation.fontrender",
        "add_text_to_image",
        "font rendering utilities",
        ("display",),
    ),
    "MovieReader": (
        "lunar_tools.presentation.movie",
        "MovieReader",
        "video utilities",
        ("video",),
    ),
    "MovieSaver": (
        "lunar_tools.presentation.movie",
        "MovieSaver",
        "video utilities",
        ("video",),
    ),
    "MovieSaverThreaded": (
        "lunar_tools.presentation.movie",
        "MovieSaverThreaded",
        "video utilities",
        ("video",),
    ),
    "add_sound": (
        "lunar_tools.presentation.movie",
        "add_sound",
        "video utilities",
        ("video",),
    ),
    "add_subtitles_to_video": (
        "lunar_tools.presentation.movie",
        "add_subtitles_to_video",
        "video utilities",
        ("video",),
    ),
    "concatenate_movies": (
        "lunar_tools.presentation.movie",
        "concatenate_movies",
        "video utilities",
        ("video",),
    ),
    "fill_up_frames_linear_interpolation": (
        "lunar_tools.presentation.movie",
        "fill_up_frames_linear_interpolation",
        "video utilities",
        ("video",),
    ),
    "interpolate_between_images": (
        "lunar_tools.presentation.movie",
        "interpolate_between_images",
        "video utilities",
        ("video",),
    ),
    "RealTimeVoice": (
        "lunar_tools.presentation.realtime_voice",
        "RealTimeVoice",
        "realtime voice interface",
        ("audio",),
    ),
}


def _load_feature(name: str) -> Any:
    module_path, attr_name, feature, extras = _FEATURE_REGISTRY[name]
    return optional_import_attr(
        module_path,
        attr_name,
        feature=feature,
        extras=extras,
    )


def __getattr__(name: str) -> Any:
    if name in _FEATURE_REGISTRY:
        value = _load_feature(name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> Iterable[str]:
    return sorted(set(globals()) | set(__all__))
