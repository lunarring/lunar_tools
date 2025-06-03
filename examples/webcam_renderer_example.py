#!/usr/bin/env python3
"""
Webcam Renderer Example

This example demonstrates how to capture images from a webcam and display them
in real-time using the lunar_tools renderer. It combines the WebCam input 
functionality with the Renderer output functionality.

Features:
- Real-time webcam capture
- Live display using OpenGL renderer
- FPS tracking for performance monitoring
- Graceful shutdown on keyboard interrupt

Usage:
    python webcam_renderer_example.py

Controls:
    - Press Ctrl+C to exit
"""

import lunar_tools as lt
import time


def main():
    print("Starting webcam renderer example...")
    print("Press Ctrl+C to exit")
    
    # Initialize webcam
    try:
        cam = lt.WebCam()
        print("✓ Webcam initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize webcam: {e}")
        return
    
    # Get initial image to determine dimensions
    try:
        initial_img = cam.get_img()
        if initial_img is None:
            print("✗ Failed to capture initial image from webcam")
            return
        
        height, width = initial_img.shape[:2]
        print(f"✓ Webcam resolution: {width}x{height}")
    except Exception as e:
        print(f"✗ Failed to get webcam image: {e}")
        return
    
    # Initialize renderer with webcam dimensions
    try:
        renderer = lt.Renderer(width=width, height=height)
        print("✓ Renderer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize renderer: {e}")
        return
    
    # Initialize FPS tracker for performance monitoring
    fps_tracker = lt.FPSTracker()
    
    print("\n🎥 Starting live webcam feed...")
    print("Note: You may need to disable VSYNC for better performance")
    print("On Ubuntu: Run nvidia-settings > Screen 0 > OpenGL > "
          "Sync to VBlank -> off")
    
    try:
        while True:
            # Start timing for capture
            fps_tracker.start_segment("capture")
            
            # Capture image from webcam
            img = cam.get_img()
            if img is None:
                print("⚠ Warning: Failed to capture frame, skipping...")
                time.sleep(0.01)  # Small delay to prevent busy waiting
                continue
            
            # Start timing for render
            fps_tracker.start_segment("render")
            
            # Display the image using the renderer
            renderer.render(img)
            
            # Print FPS and timing information
            fps_tracker.print_fps()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
    finally:
        print("✓ Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main() 