import pygame
import sys
import pytest
import warnings
from lunar_tools.display_window import Renderer

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

def test_pygame_display_fallback(monkeypatch):
    captured = {}

    def fake_set_mode(size, flags=0, depth=0, display=0, vsync=0):
        captured['display'] = display
        # Return a dummy surface object
        class DummySurface:
            pass
        return DummySurface()

    monkeypatch.setattr(pygame.display, "set_mode", fake_set_mode)
    
    with pytest.warns(UserWarning, match="Display ID 3 is not available"):
        # Create a Renderer instance with available_displays not including the requested display_id.
        renderer = Renderer(width=800, height=600, backend='pygame', do_fullscreen=True, display_id=3, available_displays=[0,1,2])
    # The Renderer should fallback to display_id 0.
    assert captured.get('display', None) == 0
    pygame.quit()

if __name__ == '__main__':
    pytest.main(["-v", __file__])