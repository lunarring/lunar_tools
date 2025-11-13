from __future__ import annotations

from pathlib import Path
from typing import Any

import sys
import types

sys.modules.setdefault("cv2", types.SimpleNamespace())

import examples.movie_example as movie_example


def test_movie_example_main(monkeypatch, tmp_path):
    written_frames: list[Any] = []

    class DummyWriter:
        def __init__(self, path: Path):
            self.path = path

        def write_frame(self, frame):
            written_frames.append(frame)

        def finalize(self):
            self.path.write_text("video", encoding="utf-8")

    class DummyStack:
        def __init__(self, path: Path):
            self.writer = DummyWriter(path)
            self.communication = None
            self._closed = False

        def close(self):
            self._closed = True

    captured = {}

    def fake_bootstrap(config):
        captured["config"] = config
        stack = DummyStack(Path(config.output_path))
        captured["stack"] = stack
        return stack

    class DummyReader:
        def __init__(self, *args, **kwargs):
            self._calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_next_frame(self):
            if self._calls < 2:
                self._calls += 1
                return types.SimpleNamespace(shape=(6, 10, 3))
            return None

    monkeypatch.setattr(movie_example, "bootstrap_movie_stack", fake_bootstrap)
    monkeypatch.setattr(
        movie_example,
        "render_gallery_frame",
        lambda *args, **kwargs: types.SimpleNamespace(shape=(6, 10, 3)),
    )
    monkeypatch.setattr(movie_example.lt, "MovieReader", DummyReader)

    output_path = tmp_path / "demo.mp4"
    exit_code = movie_example.main(
        [
            "--output",
            str(output_path),
            "--fps",
            "2",
            "--seconds",
            "1",
            "--width",
            "10",
            "--height",
            "6",
            "--stars",
            "1",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert len(written_frames) == 2
    assert captured["stack"]._closed is True
