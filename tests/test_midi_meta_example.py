from __future__ import annotations

import types

import examples.midi_meta_example as midi_example


def test_midi_meta_example_main(monkeypatch, tmp_path):
    recorded = {}

    class DummyStack:
        def __init__(self):
            self.device_name = "dummy"
            self.closed = False

        def poll_and_broadcast(self, controls):
            recorded["controls"] = controls
            return {name: idx for idx, name in enumerate(controls, 1)}

        def close(self):
            self.closed = True

    def fake_bootstrap(config):
        recorded["config"] = config
        return DummyStack()

    monkeypatch.setattr(midi_example, "bootstrap_control_inputs", fake_bootstrap)
    monkeypatch.setattr(midi_example, "time", types.SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda s: None))

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"control_input": {"use_meta": false}, "controls": {"a": {"keyboard": "x"}}}',
        encoding="utf-8",
    )

    exit_code = midi_example.main(["--config", str(config_path), "--run-seconds", "0"])

    assert exit_code == 0
    assert isinstance(recorded["config"], midi_example.ControlInputStackConfig)
    assert recorded["controls"]["a"]["keyboard"] == "x"
