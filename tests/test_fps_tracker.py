import time
import pytest
from lunar_tools.fps_tracker import FPSTracker

def test_segment_duration():
    # Set update_interval to 0 so update() always triggers display update.
    tracker = FPSTracker(update_interval=0)
    tracker.start_segment("test")
    # Sleep for 100ms
    time.sleep(0.1)
    # Call update to record the segment duration
    tracker.update()
    # Check that the segment "test" was recorded with duration near 100ms.
    assert "test" in tracker.segments, "Segment 'test' not found in segments."
    duration = tracker.segments["test"]
    # Allow a tolerance of 50ms to account for timing variability
    assert abs(duration - 100) < 50, f"Segment duration {duration} is not within expected range of 100ms."

def test_fps_calculation():
    tracker = FPSTracker(update_interval=0)
    # Simulate 5 frames with a consistent delay of 0.05 seconds between frames.
    for _ in range(5):
        time.sleep(0.05)
        tracker.update()
    fps = tracker.get_fps()
    # The expected fps should be around 20. Allow a tolerance.
    assert 18 <= fps <= 22, f"FPS value {fps} is not within the expected range (18-22 FPS)."

def test_segment_cleared_after_print():
    tracker = FPSTracker(update_interval=0)
    tracker.start_segment("seg1")
    time.sleep(0.05)
    tracker.update()
    # Call print_fps, which should print and clear segments if update_display occurred.
    tracker.print_fps()
    assert len(tracker.segments) == 0, "Segments were not cleared after printing FPS."

if __name__ == "__main__":
    pytest.main()