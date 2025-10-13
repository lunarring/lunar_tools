"""
Backward-compatible shim for legacy imports.

Prefer importing from ``lunar_tools.platform.logging`` instead::

    from lunar_tools.platform.logging import create_logger, dynamic_print
"""

from __future__ import annotations

import warnings

from lunar_tools.platform.logging import create_logger, dynamic_print, ColorFormatter


warnings.warn(
    "Importing from 'lunar_tools.logprint' is deprecated; import from "
    "'lunar_tools.platform.logging' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["create_logger", "dynamic_print", "ColorFormatter"]
