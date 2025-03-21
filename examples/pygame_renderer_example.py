import sys
import time
import numpy as np
import pygame
from lunar_tools.display_window import Renderer

def main():
    # Create a full-screen renderer on display id 1 with the pygame backend.
    renderer = Renderer(width=800, height=600, backend='pygame', do_fullscreen=True, display_id=1)
    
    # Create a dummy black image.
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Render the image once.
    renderer.render(image)
    
    # Wait a few seconds to view the window, then exit gracefully.
    time.sleep(3)
    pygame.quit()
    sys.exit(0)

if __name__ == '__main__':
    main()