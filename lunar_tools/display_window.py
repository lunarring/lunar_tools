"""Deprecated entry points for display utilities.

Import from ``lunar_tools.presentation.display_window`` instead.
"""

from __future__ import annotations

import warnings

from lunar_tools.presentation.display_window import (
    GridRenderer,
    OpenGLException,
    Renderer,
    SDLException,
)

warnings.warn(
    "Import from 'lunar_tools.presentation.display_window' instead of "
    "'lunar_tools.display_window'; the legacy module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GridRenderer", "Renderer", "SDLException", "OpenGLException"]
