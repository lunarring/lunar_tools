from pynput import keyboard
from pygame import midi
import numpy as np
import time
import os
import yaml
import threading
import pkg_resources

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
    """ A class to track midi inputs. """
    def __init__(self,
                 device_name="akai_lpd8",
                 allow_fail=True,
                 device_id_input=None,
                 device_id_output=None,
                 enforce_local_config=False,
                 ):
        self.simulate_device = False
        self.device_name = device_name
        self.allow_fail = allow_fail
        self.device_id_input = device_id_input
        self.device_id_output = device_id_output
        self.init_device_config(enforce_local_config)
        self.init_midi()
        self.reset_all_leds()
        
    def init_device_config(self, enforce_local_config):
        # Determine the path to the YAML file
        config_filename = f"{self.device_name}.yml"
        if enforce_local_config:
            config_path = os.path.join("midi_configs", config_filename)
        else:
            config_path = pkg_resources.resource_filename('lunar_tools', f'midi_configs/{config_filename}')

        # Load the configuration from the YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Set the device configuration from the YAML file
        self.name_device = config['name_device']
        self.button_down = config['button_down']
        self.button_release = config['button_release']
        self.dict_name_control = config['controls']

        # Reverse for last lookup
        self.reverse_control_name = {v[0]: k for k, v in self.dict_name_control.items()}

        # Initialize the last_value, last_time_scanned, last_time_retrieved dicts
        self.last_value = {}
        self.last_time_scanned = {}
        self.last_time_retrieved = {}
        self.nmb_button_down = {}
        for key in self.dict_name_control:
            control_type = self.dict_name_control[key][1]
            self.last_value[key] = False if control_type == "button" else 0.0
            self.last_time_scanned[key] = 0
            self.last_time_retrieved[key] = 0
            self.nmb_button_down[key] = 0 if control_type == "button" else None

        
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
        if not self.allow_fail:
            assert device_id, f"Could not auto determine device_id for {is_input}"
        if is_input:
            self.device_id_input = device_id
        else:
            self.device_id_output = device_id
        
    def check_device_id(self, is_input):
        if self.allow_fail and self.device_id_input is None:
            self.simulate_device = True
            return
        elif not self.allow_fail and self.device_id_input is None:
            raise ValueError("Device init failed! If you want to simulate, set allow_fail=True")
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
        
        # Init midi in and out
        if not self.simulate_device:
            self.midi_in = midi.Input(self.device_id_input)
            self.midi_out = midi.Output(self.device_id_output)

    def get_control_name(self, idx_control):
        if idx_control in self.reverse_control_name:
            return self.reverse_control_name[idx_control]
        else:
            return None

    def scan_inputs(self):
        if self.simulate_device:
            return
        # Gather all inputs that arrived in the meantime
        while True:
            input_last = self.midi_in.read(1)
            if input_last == []:
                break
            type_control = input_last[0][0][0]
            idx_control = input_last[0][0][1]
            val_control = input_last[0][0][2]
            
            name_control = self.get_control_name(idx_control)
            
            # Process the inputs
            if self.dict_name_control[name_control][1] == "slider":
                self.last_value[name_control] = val_control / 127.0
                self.last_time_scanned[name_control] = time.time()
                
                
            elif self.dict_name_control[name_control][1] == "button":
                if type_control == self.button_down:
                    self.last_time_scanned[name_control] = time.time()
                    self.nmb_button_down[name_control] += 1
                    self.last_value[name_control] = True
                else:
                    self.last_value[name_control] = False
                    
            
    def get(self, name_control, val_min=0, val_max=1, val_default=False, button_mode='was_pressed'):
        # Asserts
        if name_control not in self.dict_name_control:
            print(f"Warning! {name_control} is unknown. Returning val_default")
            return val_default
        # button mode correct if button
        assert button_mode in ['is_pressed', 'was_pressed', 'toggle']
        
        # Scan new inputs
        self.scan_inputs()
        
        # Process slider
        if self.dict_name_control[name_control][1] == "slider":
            if val_default is False:
                val_default = 0.5 * (val_min + val_max)
            if self.last_time_scanned[name_control] == 0:
                val_return = val_default
            else:
                val_return = val_min + (val_max-val_min) * self.last_value[name_control]
        
        # Process button
        elif self.dict_name_control[name_control][1] == "button":
            if button_mode == 'is_pressed':
                val_return = self.last_value[name_control]
            elif button_mode == "was_pressed":
                val_return = self.last_time_scanned[name_control] > self.last_time_retrieved[name_control]
            elif button_mode == "toggle":
                val_return = np.mod(self.nmb_button_down[name_control]+1,2) == 0
                # Set LED
                self.set_led(name_control, val_return)
                
        self.last_time_retrieved[name_control] = time.time()
        
        return val_return
        
    def set_led(self, name_control, state):
        if self.simulate_device:
            return
        assert name_control in self.dict_name_control
        assert self.dict_name_control[name_control][1] == "button"
        self.midi_out.write([[[self.button_down, self.dict_name_control[name_control][0], state, 0], 0]])
        
    def reset_all_leds(self):
        for name_control in self.dict_name_control:
            if self.dict_name_control[name_control][1] == "button":
                self.set_led(name_control, False)
    
    
if __name__ == "__main__":
    import lunar_tools as lt
    import time
    akai_lpd8 = lt.MidiInput(device_name="akai_lpd8")
    
    while True:
        time.sleep(0.1)
        a0 = akai_lpd8.get("A0", button_mode='toggle') # toggle switches the state with every press between on and off
        b0 = akai_lpd8.get("B0", button_mode='is_pressed') # is_pressed checks if the button is pressed down at the moment
        c0 = akai_lpd8.get("C0", button_mode='was_pressed') # was_pressed checks if the button was pressed since we checked last time
        e0 = akai_lpd8.get("E0", val_min=3, val_max=6) # e0 is a slider float between val_min and val_max
        print(f"a0: {a0}, b0: {b0}, c0: {c0}, e0: {e0}")
        
        

#%%

