import time
from collections import deque
from typing import Dict
from lunar_tools.logprint import dynamic_print

class FPSTracker:
    """A class to track and display FPS (Frames Per Second) with color-coded output and segment timing."""
    
    def __init__(self, buffer_size=30, update_interval=0.5):
        """
        Initialize the FPS tracker.
        
        Args:
            buffer_size (int): Number of frames to keep in rolling window for FPS calculation
            update_interval (float): How often to update the FPS display in seconds
        """
        self.fps_buffer = deque(maxlen=buffer_size)
        self.last_time = time.time()
        self.fps_update_interval = update_interval
        self.last_fps_update = time.time()
        self.current_fps = 0
        
        # Segment timing
        self.segments: Dict[str, float] = {}
        self.current_segment_start = None
        self.current_segment_name = None
        self._should_clear_segments = False
        
    def start_segment(self, name: str):
        """Start timing a new segment and end the previous one if it exists."""
        current_time = time.time()
        
        if self.current_segment_name is not None:
            # Record the previous segment's duration
            duration = current_time - self.current_segment_start
            self.segments[self.current_segment_name] = duration * 1000  # Convert to milliseconds
        
        self.current_segment_name = name
        self.current_segment_start = current_time
        
    def update(self):
        """
        Update FPS calculation. Call this method once per frame.
        Returns True if FPS display should be updated this frame.
        """
        current_time = time.time()
        
        # Handle the last segment if it exists
        if self.current_segment_name is not None:
            duration = current_time - self.current_segment_start
            self.segments[self.current_segment_name] = duration * 1000
            self.current_segment_name = None
            self.current_segment_start = None
        
        dt = current_time - self.last_time
        self.last_time = current_time
        self.fps_buffer.append(dt)
        
        should_update_display = current_time - self.last_fps_update >= self.fps_update_interval
        if should_update_display:
            if len(self.fps_buffer) > 0:
                self.current_fps = len(self.fps_buffer) / sum(self.fps_buffer)
            self.last_fps_update = current_time
            self._should_clear_segments = True
            
        return should_update_display
    
    def get_fps(self):
        """Get the current FPS value."""
        return self.current_fps
    
    def get_colored_fps_string(self):
        """Get a color-coded string representation of the current FPS."""
        # Clear the line first
        base_str = f"\033[2K\r\033[92mFPS: {self.current_fps:.1f}"  # Always green, with line clear
        
        # Add segment timings if any
        if self.segments:
            base_str += " | "
            segments_str = " | ".join(f"{name}: {ms:.1f}ms" for name, ms in self.segments.items())
            base_str += segments_str
            
        # Add a space at the end to ensure clean overwrite
        return base_str + " \033[0m"
    
    def print_fps(self):
        self.update()
        dynamic_print(self.get_colored_fps_string())
        if self._should_clear_segments:
            self.segments.clear()
            self._should_clear_segments = False 