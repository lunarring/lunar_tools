from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

from ._optional import optional_import_attr

_EXPORTS: Dict[str, Tuple[str, str, str | None]] = {
    "AudioRecorder": ("lunar_tools.audio", "AudioRecorder", "audio"),
    "SoundPlayer": ("lunar_tools.audio", "SoundPlayer", "audio"),
    "Speech2Text": ("lunar_tools.audio", "Speech2Text", "audio"),
    "Text2SpeechElevenlabs": ("lunar_tools.audio", "Text2SpeechElevenlabs", "audio"),
    "Text2SpeechOpenAI": ("lunar_tools.audio", "Text2SpeechOpenAI", "audio"),
    "RealTimeTranscribe": ("lunar_tools.audio", "RealTimeTranscribe", "audio"),
    "RealTimeVoice": ("lunar_tools.realtime_voice", "RealTimeVoice", "audio"),
    "AudioStackConfig": ("lunar_tools.presentation", "AudioStackConfig", None),
    "AudioConversationController": ("lunar_tools.presentation", "AudioConversationController", None),
    "bootstrap_audio_stack": ("lunar_tools.presentation", "bootstrap_audio_stack", None),
    "WebCam": ("lunar_tools.cam", "WebCam", "camera"),
    "OSCReceiver": ("lunar_tools.comms", "OSCReceiver", "comms"),
    "OSCSender": ("lunar_tools.comms", "OSCSender", "comms"),
    "ZMQPairEndpoint": ("lunar_tools.comms", "ZMQPairEndpoint", "comms"),
    "get_local_ip": ("lunar_tools.comms", "get_local_ip", None),
    "KeyboardInput": ("lunar_tools.presentation.control_input", "KeyboardInput", "inputs"),
    "MetaInput": ("lunar_tools.presentation.control_input", "MetaInput", "inputs"),
    "MidiInput": ("lunar_tools.midi", "MidiInput", "inputs"),
    "GridRenderer": ("lunar_tools.presentation.display_window", "GridRenderer", "display"),
    "Renderer": ("lunar_tools.presentation.display_window", "Renderer", "display"),
    "PopupInput": ("lunar_tools.presentation.fontrender", "PopupInput", "display"),
    "add_text_to_image": ("lunar_tools.presentation.fontrender", "add_text_to_image", "display"),
    "HealthReporter": ("lunar_tools.health_reporting", "HealthReporter", None),
    "FPSTracker": ("lunar_tools.fps_tracker", "FPSTracker", None),
    "Dalle3ImageGenerator": ("lunar_tools.image_gen", "Dalle3ImageGenerator", "imaging"),
    "GlifAPI": ("lunar_tools.image_gen", "GlifAPI", "imaging"),
    "SDXL_LCM": ("lunar_tools.image_gen", "SDXL_LCM", "imaging"),
    "SDXL_TURBO": ("lunar_tools.image_gen", "SDXL_TURBO", "imaging"),
    "FluxImageGenerator": ("lunar_tools.image_gen", "FluxImageGenerator", "imaging"),
    "OpenAIWrapper": ("lunar_tools.llm", "OpenAIWrapper", "llm"),
    "Gemini": ("lunar_tools.llm", "Gemini", "llm"),
    "Deepseek": ("lunar_tools.llm", "Deepseek", "llm"),
    "create_logger": ("lunar_tools.platform.logging", "create_logger", None),
    "dynamic_print": ("lunar_tools.platform.logging", "dynamic_print", None),
    "read_api_key": ("lunar_tools.platform.config", "read_api_key", None),
    "MovieReader": ("lunar_tools.presentation.movie", "MovieReader", "video"),
    "MovieSaver": ("lunar_tools.presentation.movie", "MovieSaver", "video"),
    "MovieSaverThreaded": ("lunar_tools.presentation.movie", "MovieSaverThreaded", "video"),
    "add_sound": ("lunar_tools.presentation.movie", "add_sound", "video"),
    "add_subtitles_to_video": ("lunar_tools.presentation.movie", "add_subtitles_to_video", "video"),
    "concatenate_movies": ("lunar_tools.presentation.movie", "concatenate_movies", "video"),
    "fill_up_frames_linear_interpolation": ("lunar_tools.presentation.movie", "fill_up_frames_linear_interpolation", "video"),
    "interpolate_between_images": ("lunar_tools.presentation.movie", "interpolate_between_images", "video"),
    "FrequencyFilter": ("lunar_tools.torch_utils", "FrequencyFilter", "display"),
    "GaussianBlur": ("lunar_tools.torch_utils", "GaussianBlur", "display"),
    "MedianBlur": ("lunar_tools.torch_utils", "MedianBlur", "display"),
    "interpolate_spherical": ("lunar_tools.torch_utils", "interpolate_spherical", "display"),
    "resize": ("lunar_tools.torch_utils", "resize", "display"),
    "interpolate_linear": ("lunar_tools.utils", "interpolate_linear", None),
    "exception_handler": ("lunar_tools.utils", "exception_handler", None),
    "get_os_type": ("lunar_tools.utils", "get_os_type", None),
    "NumpyArrayBuffer": ("lunar_tools.utils", "NumpyArrayBuffer", None),
    "scale_variable": ("lunar_tools.utils", "scale_variable", None),
    "SimpleNumberBuffer": ("lunar_tools.utils", "SimpleNumberBuffer", None),
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
