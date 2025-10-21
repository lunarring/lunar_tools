"""Deprecated control input helpers.

Import from ``lunar_tools.presentation.control_input`` instead.
"""

from __future__ import annotations

import warnings

from lunar_tools.presentation.control_input import KeyboardInput, MetaInput

warnings.warn(
    "Import from 'lunar_tools.presentation.control_input' instead of "
    "'lunar_tools.control_input'; the legacy module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["KeyboardInput", "MetaInput"]
