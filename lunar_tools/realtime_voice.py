from __future__ import annotations

import warnings

from lunar_tools.presentation.realtime_voice import AudioPlayerAsync, RealTimeVoice, TranscriptEntry


warnings.warn(
    "`lunar_tools.realtime_voice` is deprecated; import from `lunar_tools.presentation.realtime_voice` instead.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["AudioPlayerAsync", "RealTimeVoice", "TranscriptEntry"]
