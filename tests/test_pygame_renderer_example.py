import sys
import pygame
import pytest
from lunar_tools.display_window import Renderer

def test_pygame_renderer_fullscreen(monkeypatch):
    captured = {}

    def fake_set_mode(size, flags=0, depth=0, display=0, vsync=0):
        captured['display'] = display
        captured['flags'] = flags
        # Return a dummy surface object
        class DummySurface:
            pass
        return DummySurface()

    monkeypatch.setattr(pygame.display, "set_mode", fake_set_mode)
    
    # Create a Renderer instance with the pygame backend in fullscreen mode on display id 1.
    renderer = Renderer(width=800, height=600, backend='pygame', do_fullscreen=True, display_id=1)
    
    # Renderer initializes the window during setup, check captured display id
    assert captured.get('display', None) == 1
    
    # Clean up to avoid side-effects.
    pygame.quit()

if __name__ == '__main__':
    pytest.main(["-v", __file__])