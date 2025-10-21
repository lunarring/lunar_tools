from __future__ import annotations

import sys
import types


class _NumpyStub(types.ModuleType):
    __lunar_stub__ = True

    def __init__(self) -> None:
        super().__init__("numpy")
        self.int16 = "int16"
        self.float32 = "float32"
        self.uint8 = "uint8"
        self.bool_ = bool
        self.number = float
        self.ndarray = list
        self.random = types.SimpleNamespace(
            randint=lambda low, high=None, size=None, dtype=None: self._random_int(low, high, size),
            rand=self._random_float,
        )

    # Minimal helpers used by audio code or guards; they intentionally return plain lists.
    def zeros(self, shape, dtype=None):
        size = 1
        for dim in (shape if isinstance(shape, (list, tuple)) else [shape]):
            size *= dim
        return [0 for _ in range(size)]

    def empty(self, shape, dtype=None):
        return self.zeros(shape, dtype=dtype)

    def concatenate(self, seq):
        result = []
        for item in seq:
            result.extend(list(item))
        return result

    def frombuffer(self, buffer, dtype=None):
        return list(buffer)
    def dtype(self, name):
        return name

    def any(self, data):
        return any(data)

    def array(self, data):
        return list(data)

    def clip(self, data, low, high):
        return [min(max(value, low), high) for value in data]

    def hstack(self, seq):
        return self.concatenate(seq)

    # Random helpers ---------------------------------------------------------
    def _random_int(self, low, high=None, size=None):
        high = high if high is not None else low
        low = 0 if high is low else low
        if size is None:
            return low
        if isinstance(size, int):
            size = (size,)
        total = 1
        for dim in size:
            total *= dim
        return [low for _ in range(total)]

    def _random_float(self, *size):
        if not size:
            return 0.0
        total = 1
        for dim in size:
            total *= dim
        return [0.0 for _ in range(total)]

    def __getattr__(self, name):
        placeholder_types = {
            "object_": object,
            "bool_": bool,
            "number": float,
            "float32": "float32",
            "uint8": "uint8",
        }
        if name in placeholder_types:
            value = placeholder_types[name]
        else:
            value = types.SimpleNamespace(__name__=name)
        setattr(self, name, value)
        return value


if "numpy" not in sys.modules:
    sys.modules["numpy"] = _NumpyStub()


def _register_stub(module_name: str, **attrs) -> None:
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    for name, value in attrs.items():
        setattr(module, name, value)
    sys.modules[module_name] = module


class _StubSoundDeviceRecorder:
    def __init__(self, *args, **kwargs) -> None:
        self.output_filename = None
        self.is_recording = False

    def start_recording(self, output_filename=None, max_time=None) -> None:
        self.is_recording = True
        self.output_filename = output_filename

    def stop_recording(self) -> None:
        self.is_recording = False


class _StubSpeech2Text:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def start_recording(self, *args, **kwargs) -> None:
        pass

    def stop_recording(self, *args, **kwargs):
        return "transcript"

    def translate(self, audio_filepath: str) -> str:
        return "transcript"


class _StubTTS:
    def __init__(self, *args, **kwargs) -> None:
        self.generated: list[tuple[str, str | None]] = []

    def generate(self, text: str, output_filename: str | None = None) -> str:
        self.generated.append((text, output_filename))
        return output_filename or "generated.mp3"

    def play(self, text: str) -> None:  # pragma: no cover - compatibility shim
        self.generate(text)

    def stop(self) -> None:
        pass


class _StubRealtime:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self, timeout=None) -> None:
        pass

    def get_text(self) -> str:
        return ""


class _StubSoundPlayer:
    def __init__(self, *args, **kwargs) -> None:
        self.play_calls: list[tuple[str, dict]] = []

    def play(self, file_path: str, **kwargs) -> None:
        self.play_calls.append((file_path, dict(kwargs)))

    def stop(self) -> None:
        self.play_calls.append(("stop", {}))


class _StubOutputStream:
    def __init__(self, callback=None, **kwargs) -> None:
        self.callback = callback

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class _StubInputStream:
    def __init__(self, channels: int, samplerate: int, dtype: str = "int16", **kwargs) -> None:
        self.channels = channels
        self.samplerate = samplerate
        self.dtype = dtype
        self.read_available = 0

    def start(self) -> None:
        pass

    def read(self, frames: int):
        data = [[0 for _ in range(self.channels)] for _ in range(frames)]
        return data, False

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


_register_stub(
    "lunar_tools.adapters.audio.simpleaudio_player",
    SoundPlayer=_StubSoundPlayer,
)
_register_stub(
    "lunar_tools.adapters.audio.sounddevice_recorder",
    SoundDeviceRecorder=_StubSoundDeviceRecorder,
)
_register_stub(
    "lunar_tools.adapters.audio.openai_transcribe",
    Speech2Text=_StubSpeech2Text,
)
_register_stub(
    "lunar_tools.adapters.audio.openai_tts",
    Text2SpeechOpenAI=_StubTTS,
)
_register_stub(
    "lunar_tools.adapters.audio.elevenlabs_tts",
    Text2SpeechElevenlabs=_StubTTS,
)
_register_stub(
    "lunar_tools.adapters.audio.deepgram_transcribe",
    RealTimeTranscribe=_StubRealtime,
)

_register_stub(
    "sounddevice",
    InputStream=_StubInputStream,
    OutputStream=_StubOutputStream,
)
