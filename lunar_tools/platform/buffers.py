"""Buffer helpers shared across the platform layer."""

from __future__ import annotations

from collections import deque

import numpy as np


class SimpleNumberBuffer:
    """Fixed-length buffer that optionally normalises values on read."""

    def __init__(self, buffer_size: int = 500, normalize: bool = False) -> None:
        self.buffer_size = buffer_size
        self.buffer: deque[float] = deque(maxlen=self.buffer_size)
        self.default_return_value = 0
        self.normalize = normalize

    def append(self, value: float) -> None:
        self.buffer.append(value)

    def get_buffer(self) -> np.ndarray:
        buffer_array = np.array(self.buffer)
        if self.normalize and buffer_array.size:
            min_val = np.min(buffer_array)
            max_val = np.max(buffer_array)
            if min_val != max_val:
                buffer_array = (buffer_array - min_val) / (max_val - min_val)
            else:
                buffer_array = np.full_like(buffer_array, 0.5)
        return buffer_array

    def get_last_value(self) -> float:
        return self.buffer[-1] if self.buffer else self.default_return_value

    def set_buffer_size(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.buffer = deque(self.buffer, maxlen=buffer_size)

    def set_normalize(self, normalize: bool) -> None:
        self.normalize = normalize


class NumpyArrayBuffer:
    """Fixed-length buffer that tracks numpy arrays with consistent shape."""

    def __init__(self, buffer_size: int = 500, default_return_value: np.ndarray | None = None) -> None:
        self.buffer_size = buffer_size
        self.default_return_value = default_return_value
        self.buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.array_shape: tuple[int, ...] | None = None

    def append(self, array: np.ndarray) -> None:
        if self.array_shape is None:
            self.array_shape = array.shape
        if array.shape != self.array_shape:
            raise ValueError(f"Array shape {array.shape} does not match buffer shape {self.array_shape}")
        self.buffer.append(array)

    def get_last(self) -> np.ndarray | None:
        return self.buffer[-1] if self.buffer else self.default_return_value

    def set_buffer_size(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.buffer = deque(self.buffer, maxlen=buffer_size)


__all__ = ["SimpleNumberBuffer", "NumpyArrayBuffer"]
