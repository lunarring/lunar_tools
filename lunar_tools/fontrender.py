"""Deprecated font rendering helpers.

Import from ``lunar_tools.presentation.fontrender`` instead.
"""

from __future__ import annotations

import warnings

from lunar_tools.presentation.fontrender import PopupInput, add_text_to_image

warnings.warn(
    "Import from 'lunar_tools.presentation.fontrender' instead of "
    "'lunar_tools.fontrender'; the legacy module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PopupInput", "add_text_to_image"]
