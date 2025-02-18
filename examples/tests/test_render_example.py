import unittest
from lunar_tools.display_window import Renderer, PeripheralEvent

class DummyRenderer(Renderer):
    def __init__(self):
        # Do not initialize a real window; just stub minimal attributes.
        self.backend = 'dummy'
        self.running = True
    
    def render(self, image):
        # Simulate a PeripheralEvent with ESC key pressed.
        pe = PeripheralEvent()
        pe.pressed_key_code = 27
        return pe

class TestRenderExample(unittest.TestCase):
    def test_exit_condition(self):
        renderer = DummyRenderer()
        dummy_image = None  # The dummy renderer ignores the image.
        event = renderer.render(dummy_image)
        self.assertEqual(event.pressed_key_code, 27)

if __name__ == '__main__':
    unittest.main()
