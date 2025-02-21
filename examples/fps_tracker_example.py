import time
import random
from lunar_tools.fps_tracker import FPSTracker

def main():
    tracker = FPSTracker()

    # Loop through 100 iterations to simulate 100 frames/segments.
    for i in range(100):
        tracker.start_segment(f"Frame {i+1}")
        time.sleep(random.uniform(0.005, 0.300))
        tracker.update()

    # Print the final FPS and segment durations.
    tracker.print_fps()

if __name__ == "__main__":
    main()