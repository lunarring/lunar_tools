import time
import random
from lunar_tools.fps_tracker import FPSTracker

def main():
    tracker = FPSTracker()

    # Start Segment 1 and simulate work with a random sleep between 5ms and 300ms.
    tracker.start_segment("Segment 1")
    time.sleep(random.uniform(0.005, 0.300))

    # Start Segment 2 and simulate work.
    tracker.start_segment("Segment 2")
    time.sleep(random.uniform(0.005, 0.300))

    # Start Segment 3 and simulate work.
    tracker.start_segment("Segment 3")
    time.sleep(random.uniform(0.005, 0.300))

    # Call print_fps to update FPS and print segment durations.
    tracker.print_fps()

if __name__ == "__main__":
    main()