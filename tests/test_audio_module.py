from __future__ import annotations

from unittest.mock import MagicMock

from lunar_tools import audio as audio_module


def test_audio_recorder_alias_points_to_sounddevice(monkeypatch):
    sentinel_recorder = object()
    class DummySpeech2Text:
        def __init__(self, *args, **kwargs):
            pass

        def start_recording(self, *args, **kwargs):
            pass

        def stop_recording(self, *args, **kwargs):
            return "hi"

        def translate(self, audio_filepath):
            return "hi"

    class DummyTTS:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, text, output_filename=None):
            return output_filename or "dummy.mp3"

    monkeypatch.setattr(audio_module, "SoundDeviceRecorder", lambda **kwargs: sentinel_recorder)
    monkeypatch.setattr(audio_module, "Speech2Text", DummySpeech2Text)
    monkeypatch.setattr(audio_module, "Text2SpeechOpenAI", DummyTTS)
    monkeypatch.setattr(audio_module, "Text2SpeechElevenlabs", DummyTTS)
    monkeypatch.setattr(audio_module, "RealTimeTranscribe", MagicMock())
    services = audio_module.create_audio_services(include_elevenlabs=False, include_realtime_transcription=False)
    assert services.recorder_service._recorder is sentinel_recorder  # type: ignore[attr-defined]


def test_audio_services_exposes_adapter(monkeypatch):
    class DummySpeech2Text:
        def __init__(self, *args, **kwargs):
            pass

        def start_recording(self, *args, **kwargs):
            pass

        def stop_recording(self, *args, **kwargs):
            return "hi"

        def translate(self, audio_filepath):
            return "hi"

    class DummyTTS:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, text, output_filename=None):
            return output_filename or "dummy.mp3"

    monkeypatch.setattr(audio_module, "SoundDeviceRecorder", MagicMock())
    monkeypatch.setattr(audio_module, "Speech2Text", DummySpeech2Text)
    monkeypatch.setattr(audio_module, "Text2SpeechOpenAI", DummyTTS)
    monkeypatch.setattr(audio_module, "Text2SpeechElevenlabs", DummyTTS)
    monkeypatch.setattr(audio_module, "RealTimeTranscribe", MagicMock())

    services = audio_module.create_audio_services(include_elevenlabs=False, include_realtime_transcription=False)

    assert isinstance(services.speech_to_text_adapter, DummySpeech2Text)
    transcript = services.speech_to_text.stop()
    assert transcript == "hi"
