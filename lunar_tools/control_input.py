from pynput import keyboard
from pygame import midi
import time

class KeyboardInput:
    """ A class to track keyboard inputs. """

    def __init__(self):
        """ Initializes the keyboard listener and a set to store pressed keys. """
        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        """ Adds a pressed key to the set of pressed keys. """
        key_name = self.get_key_name(key)
        self.pressed_keys.add(key_name)

    def on_release(self, key):
        """ Handles key release events. Currently does nothing. """
        pass

    def get_key_name(self, key):
        """ Returns the character of the key if available, else returns the key name. """
        if hasattr(key, 'char'):
            return key.char
        else:
            return key.name

    def detect(self, key):
        """ Checks if a specific key has been pressed and removes it from the set if found. """
        key = key.lower()
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
            return True
        return False


class MidiInput:
    
    def __init__(self,
                 device_id_input=None,
                 device_id_output=None,
                 ):
    
        self.device_id_input = device_id_input
        self.device_id_output = device_id_output
        self.init_device_config()
        self.init_midi()


    def init_device_config(self):
        self.name_device = "LPD8 MIDI"        
        
    def auto_determine_device_id(self, is_input):
        dev_count = midi.get_count()
        device_id = None
        for i in range(dev_count):
            dev_info = midi.get_device_info(i)
            if self.name_device in dev_info[1].decode():
                if is_input and dev_info[2] == 1:
                    device_id = i
                elif not is_input and dev_info[3] == 1:
                    device_id = i
        assert device_id, f"Could nbt auto determine device_id for {is_input}"
        if is_input:
            self.device_id_input = device_id
        else:
            self.device_id_output = device_id
        
    def check_device_id(self, is_input):
        if is_input:
            dev_info = midi.get_device_info(self.device_id_input)
        else:
            dev_info = midi.get_device_info(self.device_id_output)
            
        assert self.name_device in dev_info[1].decode(), f"Device mismatch: name_device={self.name_device} and get_device_info={dev_info[1].decode()}"
        
    def init_midi(self):
        # Have a small safety loop for init
        for i in range(5):
            midi.quit()
            time.sleep(0.01)
            midi.init()
            
        # Set the device_ids
        if self.device_id_input is None:
            self.auto_determine_device_id(is_input=True)
        if self.device_id_output is None:
            self.auto_determine_device_id(is_input=False)
            
        # Check the device_ids
        self.check_device_id(is_input=True)
        self.check_device_id(is_input=False)
            


    
    def update(self):
        inputs = self.midi_in.read(1)
        print(inputs)


if __name__ == "__main__":
    self = MidiInput()
    
    # while True:
    #     self.update()
        
