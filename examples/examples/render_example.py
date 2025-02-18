#!/usr/bin/env python3
import time
import numpy as np
from lunar_tools.display_window import Renderer

# Initialize Renderer with opencv backend for simplicity
renderer = Renderer(width=480, height=320, backend='opencv', do_fullscreen=False)

print("Starting render example. Press ESC to exit.")

# Create a dummy image: random color image
dummy_image = np.random.randint(0, 256, (320, 480, 3), dtype=np.uint8)

while True:
    # Render the dummy image and capture peripheral events.
    peripheral_event = renderer.render(dummy_image)
    
    # For opencv backend, pressed_key_code is set from cv2.waitKey
    if peripheral_event and peripheral_event.pressed_key_code == 27:  # ESC key
        print("ESC pressed. Exiting render loop.")
        break

    time.sleep(0.1)
