import time
import random
from lunar_tools.fps_tracker import FPSTracker

def main():
    tracker = FPSTracker(update_interval=0.5)
    segment_names = [f"Segment {i}" for i in range(1, 6)]
    total_duration = 10.0
    segment_interval = total_duration / 5  # 2 seconds per segment
    
    start_time = time.time()
    next_segment_index = 0
    # Immediately start the first segment.
    tracker.start_segment(segment_names[next_segment_index])
    next_segment_index += 1

    while time.time() - start_time < total_duration:
        time.sleep(random.uniform(0.005, 0.300))
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Start a new segment at fixed intervals.
        if next_segment_index < len(segment_names) and elapsed >= next_segment_index * segment_interval:
            tracker.start_segment(segment_names[next_segment_index])
            next_segment_index += 1
        
        if tracker.update():
            print(tracker.get_colored_fps_string(), flush=True)
    
    # Final update call to flush and print the last segment's timing.
    tracker.update()
    print(tracker.get_colored_fps_string(), flush=True)

if __name__ == "__main__":
    main()