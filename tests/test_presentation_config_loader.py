from __future__ import annotations

import json
from pathlib import Path

import pytest

from lunar_tools.presentation.config_loader import load_config_file


def test_load_config_file_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    payload = {"realtime_voice": {"instructions": "Hello"}}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = load_config_file(config_path)
    assert result == payload


def test_load_config_file_yaml(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "audio_stack:\n  enable_playback: true\nrealtime_voice:\n  instructions: Test\n",
        encoding="utf-8",
    )

    result = load_config_file(config_path)
    assert result["audio_stack"]["enable_playback"] is True
    assert result["realtime_voice"]["instructions"] == "Test"


def test_load_config_file_invalid_extension(tmp_path: Path) -> None:
    config_path = tmp_path / "config.txt"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config_file(config_path)
