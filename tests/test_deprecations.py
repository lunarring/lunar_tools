from __future__ import annotations

import warnings

import lunar_tools as lt


def _capture_warning(name: str) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(lt, name)
    return caught


def test_audio_recorder_deprecation():
    warnings_list = _capture_warning("AudioRecorder")
    assert any("AudioRecorder is deprecated" in str(item.message) for item in warnings_list)


def test_realtime_voice_deprecation():
    warnings_list = _capture_warning("RealTimeVoice")
    assert any("RealTimeVoice is deprecated" in str(item.message) for item in warnings_list)
