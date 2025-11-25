import io
import time
import builtins
import threading

import pytest
from lunar_tools.realtime_voice import RealTimeVoice

from examples.voice.realtime_voice_example import run_realtime_voice_example

class DummyRTV(RealTimeVoice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def start(self):
        print("Dummy start triggered")
    def pause(self):
        print("Dummy pause triggered")
    def resume(self):
        print("Dummy resume triggered")
    def mute_mic(self):
        print("Dummy mute triggered")
    def unmute_mic(self):
        print("Dummy unmute triggered")
    def inject_message(self, message: str):
        print(f"Dummy inject triggered with message: {message}")
    def update_instructions(self, new_instructions: str):
        print(f"Dummy update instructions triggered with: {new_instructions}")
    def stop(self):
        print("Dummy stop triggered")

def test_realtime_voice_keyboard_interactive(monkeypatch, capsys):
    # Sequence of inputs: s, p, r, m (mute), m (unmute), i (inject), <message>, u (update), <new instructions>, q (quit)
    inputs = iter([
        's',
        'p',
        'r',
        'm',
        'm',
        'i',
        'Test message injection',
        'u',
        'New instructions',
        'q'
    ])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    import examples.voice.realtime_voice_example as rve
    original_rtv = rve.RealTimeVoice
    rve.RealTimeVoice = DummyRTV

    rve.run_realtime_voice_example()

    rve.RealTimeVoice = original_rtv

    captured = capsys.readouterr().out
    assert "Dummy start triggered" in captured
    assert "Dummy pause triggered" in captured
    assert "Dummy resume triggered" in captured
    assert "Dummy mute triggered" in captured
    assert "Dummy unmute triggered" in captured
    assert "Dummy inject triggered with message: Test message injection" in captured
    assert "Dummy update instructions triggered with: New instructions" in captured
    assert "Dummy stop triggered" in captured
