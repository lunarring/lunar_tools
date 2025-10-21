from __future__ import annotations

from unittest.mock import MagicMock

from lunar_tools.audio import AudioServices, create_audio_services
from lunar_tools.presentation.audio_stack import AudioConversationController, AudioStackConfig, bootstrap_audio_stack
from lunar_tools.services.audio.recorder_service import RecorderService
from lunar_tools.services.audio.speech_to_text_service import SpeechToTextService
from lunar_tools.services.audio.tts_service import TextToSpeechService
from tests.fakes.audio import (
    FakeRecorderPort,
    FakeSoundPlaybackPort,
    FakeSpeechSynthesisPort,
    FakeSpeechToTextPort,
)


def test_speech_to_text_service_record_and_translate():
    fake_port = FakeSpeechToTextPort(transcript="transcribed text")
    service = SpeechToTextService(fake_port, wait_fn=lambda _: None)

    transcript = service.record_and_translate(
        output_filename="recording.mp3",
        max_time=1.2,
        minimum_duration=0.25,
    )

    assert transcript == "transcribed text"
    assert fake_port.started is True
    assert fake_port.stopped is True
    assert fake_port.output_filename == "recording.mp3"
    assert fake_port.max_time == 1.2
    assert fake_port.minimum_duration == 0.25


def test_speech_to_text_service_context_manager():
    fake_port = FakeSpeechToTextPort()
    service = SpeechToTextService(fake_port, wait_fn=lambda _: None)

    with service.recording(output_filename="session.mp3", max_time=0.5):
        assert fake_port.started is True
        assert fake_port.output_filename == "session.mp3"
        assert fake_port.max_time == 0.5

    assert fake_port.stopped is True


def test_text_to_speech_service_routes_to_playback():
    synthesizer = FakeSpeechSynthesisPort()
    playback = FakeSoundPlaybackPort()
    service = TextToSpeechService(synthesizer, playback=playback)

    path = service.play("hello", output_filename="tts.mp3", pan=0.1)

    assert path == "tts.mp3"
    assert synthesizer.generated == [("hello", "tts.mp3")]
    assert playback.play_calls == [("tts.mp3", {"pan": 0.1})]


def test_text_to_speech_service_synthesize_optional_playback():
    synthesizer = FakeSpeechSynthesisPort()
    playback = FakeSoundPlaybackPort()
    service = TextToSpeechService(synthesizer, playback=playback)

    path = service.synthesize("greetings", play=True, playback_kwargs={"pan": -0.25})

    assert path == "fake-tts.mp3"
    assert playback.play_calls == [("fake-tts.mp3", {"pan": -0.25})]


def test_audio_services_factory_uses_service_layer(monkeypatch):
    from lunar_tools import audio as audio_module

    class DummyRecorder:
        def __init__(self, *args, **kwargs):
            self.output_filename = None
            self.is_recording = False

        def start_recording(self, output_filename=None, max_time=None):
            self.is_recording = True
            self.output_filename = output_filename

        def stop_recording(self):
            self.is_recording = False

    class DummySpeechToTextAdapter:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def start_recording(self, output_filename=None, max_time=None):
            self.calls.append(("start", output_filename, max_time))

        def stop_recording(self, minimum_duration=0.4):
            self.calls.append(("stop", minimum_duration))
            return "dummy transcript"

        def translate(self, audio_filepath):
            self.calls.append(("translate", audio_filepath))
            return "dummy transcript"

    class DummyTTSAdapter:
        def __init__(self, *args, **kwargs):
            self.generated = []

        def generate(self, text, output_filename=None):
            self.generated.append((text, output_filename))
            return output_filename or "generated.mp3"

    class DummyRealtimeTranscribe:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(audio_module, "SoundDeviceRecorder", DummyRecorder)
    monkeypatch.setattr(audio_module, "Speech2Text", DummySpeechToTextAdapter)
    monkeypatch.setattr(audio_module, "Text2SpeechOpenAI", DummyTTSAdapter)
    monkeypatch.setattr(audio_module, "Text2SpeechElevenlabs", DummyTTSAdapter)
    monkeypatch.setattr(audio_module, "RealTimeTranscribe", DummyRealtimeTranscribe)

    services = create_audio_services(
        include_elevenlabs=False,
        include_realtime_transcription=False,
    )

    assert isinstance(services.recorder_service, RecorderService)
    assert isinstance(services.speech_to_text, SpeechToTextService)
    assert isinstance(services.openai_tts, TextToSpeechService)
    assert services.speech_to_text_adapter.__class__ is DummySpeechToTextAdapter

    # Ensure synthesizer service returns the generated file path.
    assert services.openai_tts.synthesize("hello world") == "generated.mp3"


def test_bootstrap_audio_stack_selects_preferred_tts(monkeypatch):
    fake_recorder_port = FakeRecorderPort()
    recorder_service = RecorderService(fake_recorder_port)

    fake_stt_port = FakeSpeechToTextPort(transcript="hello")
    speech_to_text_service = SpeechToTextService(fake_stt_port, wait_fn=lambda _: None)

    openai_synth_port = FakeSpeechSynthesisPort()
    eleven_synth_port = FakeSpeechSynthesisPort()

    openai_service = TextToSpeechService(openai_synth_port)
    eleven_service = TextToSpeechService(eleven_synth_port)

    services_bundle = AudioServices(
        recorder_service=recorder_service,
        speech_to_text=speech_to_text_service,
        speech_to_text_adapter=fake_stt_port,
        openai_tts=openai_service,
        elevenlabs_tts=eleven_service,
        realtime_transcription=None,
    )

    monkeypatch.setattr("lunar_tools.presentation.audio_stack.create_audio_services", lambda **kwargs: services_bundle)

    playback_events = []

    openai_service.set_playback = MagicMock(wraps=openai_service.set_playback)  # type: ignore[assignment]
    eleven_service.set_playback = MagicMock(wraps=eleven_service.set_playback)  # type: ignore[assignment]

    def fake_attach_playback(services_arg, *, enable_playback, blocking_playback, logger_name):
        class FakePlaybackAdapter:
            def play(self, file_path: str, **kwargs) -> None:
                playback_events.append((file_path, dict(kwargs)))

            def stop(self) -> None:
                playback_events.append(("stop", {}))

        playback_adapter = FakePlaybackAdapter()
        services_arg.openai_tts.set_playback(playback_adapter)
        services_arg.elevenlabs_tts.set_playback(playback_adapter)

    monkeypatch.setattr("lunar_tools.presentation.audio_stack._attach_playback_adapter", fake_attach_playback)

    config = AudioStackConfig(
        include_elevenlabs=True,
        preferred_tts="elevenlabs",
        enable_playback=True,
        blocking_playback=True,
    )

    services, synthesiser = bootstrap_audio_stack(config)

    assert services is services_bundle
    assert synthesiser is eleven_service
    assert openai_service.set_playback.call_count == 1
    assert eleven_service.set_playback.call_count == 1

    controller = AudioConversationController(services, synthesiser=eleven_service)
    controller.respond("hi", play=True)

    assert eleven_synth_port.generated[0][0] == "hi"
    assert playback_events[0][0] == "fake-tts.mp3"
