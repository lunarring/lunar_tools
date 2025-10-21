"""Deprecated video utilities.

Import from ``lunar_tools.presentation.movie`` instead.
"""

from __future__ import annotations

import warnings

from lunar_tools.presentation.movie import (
    MovieReader,
    MovieSaver,
    MovieSaverThreaded,
    add_sound,
    add_subtitles_to_video,
    concatenate_movies,
    fill_up_frames_linear_interpolation,
    interpolate_between_images,
)

warnings.warn(
    "Import from 'lunar_tools.presentation.movie' instead of "
    "'lunar_tools.movie'; the legacy module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "MovieReader",
    "MovieSaver",
    "MovieSaverThreaded",
    "add_sound",
    "add_subtitles_to_video",
    "concatenate_movies",
    "fill_up_frames_linear_interpolation",
    "interpolate_between_images",
]
