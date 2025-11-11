"""Numerical helpers for interpolation and range scaling."""

from __future__ import annotations

import numpy as np


def scale_variable(
    variable: float,
    min_input: float,
    max_input: float,
    min_output: float,
    max_output: float,
) -> float:
    """Scale ``variable`` from one range into another."""
    clipped = np.clip(variable, min_input, max_input)
    return min_output + (clipped - min_input) * (max_output - min_output) / (max_input - min_input)


def interpolate_linear(p0, p1, fract_mixing: float):
    r"""Linearly interpolate between ``p0`` and ``p1``."""
    reconvert_uint8 = False
    if isinstance(p0, np.ndarray) and p0.dtype == "uint8":
        reconvert_uint8 = True
        p0 = p0.astype(np.float64)
    if isinstance(p1, np.ndarray) and p1.dtype == "uint8":
        reconvert_uint8 = True
        p1 = p1.astype(np.float64)

    interp = (1 - fract_mixing) * p0 + fract_mixing * p1

    if reconvert_uint8:
        interp = np.clip(interp, 0, 255).astype(np.uint8)
    return interp


__all__ = ["scale_variable", "interpolate_linear"]
