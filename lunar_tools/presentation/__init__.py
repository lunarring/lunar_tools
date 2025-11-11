"""Presentation layer: UI components, CLI entry points, and sample applications."""

from importlib import import_module
from typing import Any, Dict, Tuple

from lunar_tools._optional import optional_import_attr

_EXPORTS: Dict[str, Tuple[str, str, str | None]] = {
    "AudioConversationController": ("lunar_tools.presentation.audio_stack", "AudioConversationController", None),
    "AudioStackConfig": ("lunar_tools.presentation.audio_stack", "AudioStackConfig", None),
    "bootstrap_audio_stack": ("lunar_tools.presentation.audio_stack", "bootstrap_audio_stack", None),
    "KeyboardInput": ("lunar_tools.presentation.control_input", "KeyboardInput", "inputs"),
    "MetaInput": ("lunar_tools.presentation.control_input", "MetaInput", "inputs"),
    "GridRenderer": ("lunar_tools.presentation.display_window", "GridRenderer", "display"),
    "Renderer": ("lunar_tools.presentation.display_window", "Renderer", "display"),
    "PopupInput": ("lunar_tools.presentation.fontrender", "PopupInput", "display"),
    "add_text_to_image": ("lunar_tools.presentation.fontrender", "add_text_to_image", "display"),
    "MovieReader": ("lunar_tools.presentation.movie", "MovieReader", "video"),
    "MovieSaver": ("lunar_tools.presentation.movie", "MovieSaver", "video"),
    "MovieSaverThreaded": ("lunar_tools.presentation.movie", "MovieSaverThreaded", "video"),
    "add_sound": ("lunar_tools.presentation.movie", "add_sound", "video"),
    "add_subtitles_to_video": ("lunar_tools.presentation.movie", "add_subtitles_to_video", "video"),
    "concatenate_movies": ("lunar_tools.presentation.movie", "concatenate_movies", "video"),
    "fill_up_frames_linear_interpolation": ("lunar_tools.presentation.movie", "fill_up_frames_linear_interpolation", "video"),
    "interpolate_between_images": ("lunar_tools.presentation.movie", "interpolate_between_images", "video"),
    "RealTimeVoice": ("lunar_tools.presentation.realtime_voice", "RealTimeVoice", "audio"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute, extra = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    if extra is None:
        module = import_module(module_name)
        value = getattr(module, attribute)
    else:
        value = optional_import_attr(
            module_name,
            attribute,
            feature=name,
            extras=extra,
        )

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
