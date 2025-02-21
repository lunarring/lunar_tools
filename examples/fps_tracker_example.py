import time
import random
from lunar_tools.fps_tracker import FPSTracker

def main():
    tracker = FPSTracker(update_interval=0.5)
    segment_count = 0
    start_time = time.time()
    # Run the simulation for 5 seconds
    while time.time() - start_time < 5:
        # Sleep for a random duration between 5ms and 300ms
        time.sleep(random.uniform(0.005, 0.300))
        
        # Randomly start a new segment with a 10% chance
        if random.random() < 0.1:
            segment_count += 1
            tracker.start_segment(f"Segment {segment_count}")
        
        # Call update every frame and, if it's time, print the FPS and segment timings.
        if tracker.update():
            print(tracker.get_colored_fps_string(), flush=True)

if __name__ == "__main__":
    main()