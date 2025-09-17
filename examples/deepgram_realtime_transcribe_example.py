#!/usr/bin/env python3
"""
Example: Real-time transcription using Deepgram via lunar_tools.audio.RealTimeTranscribe.

Requirements:
- deepgram-sdk installed (`pip install deepgram-sdk`)
- DEEPGRAM_API_KEY in environment or configured via lunar_tools.utils.read_api_key

Run:
  python examples/deepgram_realtime_transcribe_example.py
"""

import time


def main() -> int:
    try:
        from lunar_tools.audio import RealTimeTranscribe
    except Exception as e:
        print(f"Failed to import RealTimeTranscribe: {e}")
        return 1

    try:
        rtt = RealTimeTranscribe(auto_start=True, ready_timeout=10.0)
    except ImportError:
        print("Deepgram SDK not installed. Install 'deepgram-sdk' to run this example.")
        return 1
    except Exception as e:
        print(f"Failed to initialize RealTimeTranscribe: {e}")
        return 1

    if rtt.is_ready():
        print("Start talking! Press Ctrl+C to stop...")
    else:
        print("Deepgram not ready yet. Waiting for connection events...")

    try:
        time_silence = 2
        while True:
            full_text = rtt.get_text()
            recent_text = " ".join(rtt.get_chunks(silence_duration=time_silence))
            print(f"Transcript so far: {full_text}")
            print(f"Transcript since {time_silence}s silence: {recent_text}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        rtt.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

