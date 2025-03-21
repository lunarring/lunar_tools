import time
import numpy as np
from PIL import Image
from lunar_tools.display_window import Renderer

def main():
    # Set the SDL_VIDEO_FULLSCREEN_DISPLAY to use screen id 1 when in fullscreen mode.
    # The Renderer class will set the environment variable in its pygame_setup method if do_fullscreen is True.
    sz = (400, 600)  # (height, width)
    renderer = Renderer(width=sz[1], height=sz[0], backend='pygame', do_fullscreen=True, screen_id=1, window_title="Pygame Renderer Example")
    
    while renderer.running:
        # Generate a random image with random colors
        image = np.random.randint(0, 256, (sz[0], sz[1], 3), dtype=np.uint8)
        
        # Render the image using the pygame backend
        peripheralEvent = renderer.render(image)
        
        # Check for key press events to exit (using the ESC key which has keycode 27)
        if peripheralEvent and peripheralEvent.pressed_key_code is not None:
            if peripheralEvent.pressed_key_code == 27:
                break
        
        time.sleep(0.03)

if __name__ == '__main__':
    main()
    