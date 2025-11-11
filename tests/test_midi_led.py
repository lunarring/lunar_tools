import pytest

try:
    import lunar_tools.midi as midi_module
except Exception as exc:  # pragma: no cover - optional dependency guard
    pytest.skip(f"Skipping MIDI LED tests: {exc}")


def test_led_velocity_mapping():
    assert midi_module._led_velocity(True) == 127
    assert midi_module._led_velocity(False) == 0
    # Non-boolean truthy/falsey values follow the same rule
    assert midi_module._led_velocity(1) == 127
    assert midi_module._led_velocity(0) == 0


def test_set_led_uses_note_on_and_off():
    class FakeOut:
        def __init__(self):
            self.messages = []

        def write(self, payload):
            self.messages.append(payload)

    midi = midi_module.MidiInput.__new__(midi_module.MidiInput)
    midi.simulate_device = False
    midi.id_config = {"A3": [1, "button"]}
    midi.button_down = 144
    midi.button_release = 128
    midi.midi_out = FakeOut()

    midi.set_led("A3", True)
    midi.set_led("A3", False)

    assert midi.midi_out.messages == [
        [[[144, 1, 127, 0], 0]],
        [[[128, 1, 0, 0], 0]],
    ]
