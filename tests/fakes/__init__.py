"""
Utility fakes for service-layer tests.
"""

from .audio import FakeRecorderPort, FakeSpeechSynthesisPort, FakeTranscriptionPort
from .comms import FakeMessageReceiver, FakeMessageSender
from .llm import FakeLanguageModelPort
from .vision import FakeImageGeneratorPort

__all__ = [
    "FakeRecorderPort",
    "FakeSpeechSynthesisPort",
    "FakeTranscriptionPort",
    "FakeMessageReceiver",
    "FakeMessageSender",
    "FakeLanguageModelPort",
    "FakeImageGeneratorPort",
]
