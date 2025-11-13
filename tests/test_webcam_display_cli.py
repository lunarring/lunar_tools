from __future__ import annotations

import types

import importlib
import sys
import types

sys.modules.setdefault("cv2", types.SimpleNamespace())


class _DummyTorch:
    Tensor = type("DummyTensor", (), {})

    def __init__(self):
        self.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.nn = types.SimpleNamespace(
            functional=types.SimpleNamespace(interpolate=lambda x, **kwargs: x)
        )

    def from_numpy(self, array):
        return array

    def clamp(self, tensor, min=0, max=1):
        return tensor

    def ones(self, *args, **kwargs):
        return 1

    def cat(self, seq, dim=None):
        return seq[0]

    def rand(self, *args, **kwargs):
        return self.Tensor()


dummy_torch = _DummyTorch()
dummy_torch.__spec__ = types.SimpleNamespace()

sys.modules.setdefault("torch", dummy_torch)

webcam_display = importlib.import_module("lunar_tools.presentation.webcam_display")


def test_webcam_display_main(monkeypatch, tmp_path):
    frames = []

    class DummyFrame:
        shape = (480, 640, 3)

    class DummyCamera:
        def __init__(self, *args, **kwargs):
            self.do_mirror = False
            self.shift_colors = True
            self.frames = [DummyFrame(), DummyFrame()]

        def get_img(self):
            return self.frames.pop(0) if self.frames else None

    class DummyRenderer:
        def __init__(self):
            self.rendered = []

        def render(self, frame):
            self.rendered.append(frame)

    class DummyStack:
        def __init__(self):
            self.renderer = DummyRenderer()
            self.communication = None
            self.closed = False

        def close(self):
            self.closed = True

    captured = {}

    def fake_bootstrap(config):
        captured["config"] = config
        stack = DummyStack()
        captured["stack"] = stack
        return stack

    config_path = tmp_path / "config.json"
    config_path.write_text('{"camera": {"mirror": true}}', encoding="utf-8")

    monkeypatch.setattr(webcam_display, "lt", types.SimpleNamespace(WebCam=DummyCamera))
    monkeypatch.setattr(webcam_display, "bootstrap_display_stack", fake_bootstrap)

    exit_code = webcam_display.main(
        ["--config", str(config_path), "--run-seconds", "0", "--print-fps", "--fps-interval", "1"]
    )

    assert exit_code == 0
    stack = captured["stack"]
    assert isinstance(stack.renderer.rendered, list)
    assert captured["config"].width == 640
    assert captured["config"].height == 480
    assert stack.closed is True
