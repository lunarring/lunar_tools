from importlib import import_module
from typing import TYPE_CHECKING

_LAZY_EXPORTS = {
    "AudioRecorder": ".audio",
    "SoundPlayer": ".audio",
    "Speech2Text": ".audio",
    "Text2SpeechElevenlabs": ".audio",
    "Text2SpeechOpenAI": ".audio",
    "RealTimeTranscribe": ".audio",
    "RealTimeVoice": ".realtime_voice",
    "WebCam": ".cam",
    "OSCReceiver": ".comms",
    "OSCSender": ".comms",
    "SimpleWebRTCSignalingServer": ".comms",
    "WebRTCDataChannel": ".comms",
    "ZMQPairEndpoint": ".comms",
    "get_local_ip": ".comms",
    "KeyboardInput": ".control_input",
    "MetaInput": ".control_input",
    "MidiInput": ".midi",
    "GridRenderer": ".display_window",
    "Renderer": ".display_window",
    "PopupInput": ".fontrender",
    "add_text_to_image": ".fontrender",
    "HealthReporter": ".health_reporting",
    "FPSTracker": ".fps_tracker",
    "Dalle3ImageGenerator": ".image_gen",
    "GlifAPI": ".image_gen",
    "SDXL_LCM": ".image_gen",
    "SDXL_TURBO": ".image_gen",
    "FluxImageGenerator": ".image_gen",
    "OpenAIWrapper": ".llm",
    "Gemini": ".llm",
    "Deepseek": ".llm",
    "LogPrint": ".logprint",
    "dynamic_print": ".logprint",
    "MovieReader": ".movie",
    "MovieSaver": ".movie",
    "MovieSaverThreaded": ".movie",
    "add_sound": ".movie",
    "add_subtitles_to_video": ".movie",
    "concatenate_movies": ".movie",
    "fill_up_frames_linear_interpolation": ".movie",
    "interpolate_between_images": ".movie",
    "FrequencyFilter": ".torch_utils",
    "GaussianBlur": ".torch_utils",
    "MedianBlur": ".torch_utils",
    "interpolate_spherical": ".torch_utils",
    "resize": ".torch_utils",
    "exception_handler": ".utils",
    "get_os_type": ".utils",
    "interpolate_linear": ".utils",
    "read_all_api_keys_from_lunar_config": ".utils",
    "read_api_key": ".utils",
    "read_api_key_from_lunar_config": ".utils",
    "save_api_key_to_lunar_config": ".utils",
    "NumpyArrayBuffer": ".utils",
    "scale_variable": ".utils",
    "SimpleNumberBuffer": ".utils",
}

__all__ = tuple(_LAZY_EXPORTS)

if TYPE_CHECKING:
    from .audio import (
        AudioRecorder,
        RealTimeTranscribe,
        SoundPlayer,
        Speech2Text,
        Text2SpeechElevenlabs,
        Text2SpeechOpenAI,
    )
    from .realtime_voice import RealTimeVoice
    from .cam import WebCam
    from .comms import (
        OSCReceiver,
        OSCSender,
        SimpleWebRTCSignalingServer,
        WebRTCDataChannel,
        ZMQPairEndpoint,
        get_local_ip,
    )
    from .control_input import KeyboardInput, MetaInput
    from .midi import MidiInput
    from .display_window import GridRenderer, Renderer
    from .fontrender import PopupInput, add_text_to_image
    from .health_reporting import HealthReporter
    from .fps_tracker import FPSTracker
    from .image_gen import (
        Dalle3ImageGenerator,
        FluxImageGenerator,
        GlifAPI,
        SDXL_LCM,
        SDXL_TURBO,
    )
    from .llm import Deepseek, Gemini, OpenAIWrapper
    from .logprint import LogPrint, dynamic_print
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
    from .torch_utils import (
        FrequencyFilter,
        GaussianBlur,
        MedianBlur,
        interpolate_spherical,
        resize,
    )
    from .utils import (
        NumpyArrayBuffer,
        SimpleNumberBuffer,
        exception_handler,
        get_os_type,
        interpolate_linear,
        read_all_api_keys_from_lunar_config,
        read_api_key,
        read_api_key_from_lunar_config,
        save_api_key_to_lunar_config,
        scale_variable,
    )


def __getattr__(name: str):
    try:
        module = import_module(_LAZY_EXPORTS[name], __name__)
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(__all__) | set(globals().keys()))
