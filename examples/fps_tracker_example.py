import time
import random
from lunar_tools.fps_tracker import FPSTracker

def main():
    tracker = FPSTracker()

    # Loop through 100 iterations to simulate 100 frames/segments.
    for i in range(100):
        tracker.start_segment(f"processing segment 1")
        time.sleep(random.uniform(0.005, 0.300))
        tracker.start_segment(f"processing segment 2")
        time.sleep(random.uniform(0.005, 0.300))

        # Print the final FPS and segment durations.
        tracker.print_fps()

if __name__ == "__main__":
    main()
