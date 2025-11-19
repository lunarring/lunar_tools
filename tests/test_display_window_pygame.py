import sys
import pytest
import pygame
import numpy as np

if getattr(np, "__lunar_stub__", False) or not hasattr(np, "ndarray"):
    pytest.skip("Display window tests require functional numpy.", allow_module_level=True)

from lunar_tools.presentation.display_window import Renderer

def test_pygame_display_id(monkeypatch):
    captured = {}

    def fake_set_mode(size, flags=0, depth=0, display=0, vsync=0):
        captured['display'] = display
        # Return a dummy surface object
        class DummySurface:
            pass
        return DummySurface()

    monkeypatch.setattr(pygame.display, "set_mode", fake_set_mode)
    
    # Create a Renderer instance with the pygame backend and a specific display id.
    renderer = Renderer(width=640, height=480, backend='pygame', display_id=2)
    # The pygame_setup should have been called during initialization.
    assert captured.get('display', None) == 2
    # Clean up to avoid side effects in other tests
    pygame.quit()

if __name__ == '__main__':
    pytest.main(["-v", __file__])
