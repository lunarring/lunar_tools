import os
import pygame
import pytest
from lunar_tools.display_window import Renderer

def test_pygame_fullscreen_screen_id():
    # Ensure the environment variable is not preset.
    if "SDL_VIDEO_FULLSCREEN_DISPLAY" in os.environ:
        del os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"]
    # Instantiate Renderer with pygame backend, fullscreen enabled and a non-default screen_id.
    renderer = Renderer(width=640, height=480, backend="pygame", do_fullscreen=True, screen_id=1, window_title="Test Window")
    # Verify that the SDL_VIDEO_FULLSCREEN_DISPLAY environment variable has been set to '1'.
    assert os.environ.get("SDL_VIDEO_FULLSCREEN_DISPLAY") == "1"
    # Cleanup
    pygame.quit()