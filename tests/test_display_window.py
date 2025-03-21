import os
import pytest
from lunar_tools.display_window import Renderer

def test_pygame_screen_id_env_variable():
    # Specify a non-default screen id (e.g., 1)
    screen_id = 1
    # Instantiate Renderer with pygame backend in fullscreen mode and the given screen_id
    renderer = Renderer(width=800, height=600, window_title="Test", do_fullscreen=True, screen_id=screen_id, backend='pygame')
    # Verify the environment variable is correctly set before display mode is set
    assert os.environ.get("SDL_VIDEO_FULLSCREEN_DISPLAY") == str(screen_id)
    # Clean up by stopping the renderer (if needed)
    renderer.running = False
    # Quit pygame to release resources
    import pygame
    pygame.quit()
    
if __name__ == "__main__":
    pytest.main()