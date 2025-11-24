"""Play generated sine tones from disk and directly from memory."""

from __future__ import annotations

import math
import os
import struct
import tempfile
import wave

import lunar_tools as lt
from pydub import AudioSegment

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional
    np = None


def _generate_pcm_bytes(frequency: float, duration: float, sample_rate: int) -> bytes:
    """Return 16-bit little-endian PCM bytes of a sine wave."""
    amplitude = 0.3  # avoid clipping
    sample_count = int(duration * sample_rate)

    if np is not None:
        t = np.arange(sample_count) / sample_rate
        samples = amplitude * np.sin(2 * np.pi * frequency * t)
        return (samples * 32767).astype("<i2").tobytes()

    # Fallback without numpy
    frames = []
    for i in range(sample_count):
        sample = amplitude * math.sin(2 * math.pi * frequency * i / sample_rate)
        frames.append(struct.pack("<h", int(sample * 32767)))
    return b"".join(frames)


def _write_wave_file(path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    """Write PCM bytes into a mono WAV container."""
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def _bytes_to_audiosegment(pcm_bytes: bytes, sample_rate: int) -> AudioSegment:
    """Wrap PCM data in an AudioSegment for in-memory playback."""
    return AudioSegment(
        data=pcm_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )


def main():
    sample_rate = 44100
    player = lt.SoundPlayer(blocking_playback=True)

    # --- Example 1: Save to disk and play via existing API ---
    file_bytes = _generate_pcm_bytes(frequency=440.0, duration=1.5, sample_rate=sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        _write_wave_file(tmp_path, file_bytes, sample_rate)
        print("Playing 440 Hz tone from a temporary WAV file...")
        player.play_sound(tmp_path)
    finally:
        os.unlink(tmp_path)

    # --- Example 2: Inject audio without saving to disk ---
    inline_bytes = _generate_pcm_bytes(frequency=660.0, duration=1.0, sample_rate=sample_rate)
    inline_segment = _bytes_to_audiosegment(inline_bytes, sample_rate)

    print("Playing 660 Hz tone directly from memory...")
    player.play_audiosegment(inline_segment)


if __name__ == "__main__":
    main()
