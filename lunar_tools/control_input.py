from pynput import keyboard
from pygame import midi
import numpy as np
import time
import os
import yaml
import threading
import pkg_resources
import inspect
import json

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
        self.init_vars()
        self.init_midi()
        self.reset_all_leds()
        self.autodetect_varname = True
        
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
        self.id_config = config['controls']

        # Reverse for last lookup
        self.reverse_control_name = {v[0]: k for k, v in self.id_config.items()}

    def init_vars(self):
        # Initializes all variables
        self.id_value = {}
        self.id_last_time_scanned = {}
        self.id_last_time_retrieved = {}
        self.id_nmb_button_down = {}
        self.id_nmb_scan_cycles = {}
        self.id_name = {}
        for key in self.id_config:
            control_type = self.id_config[key][1]
            self.id_value[key] = False if control_type == "button" else 0.0
            self.id_last_time_scanned[key] = 0
            self.id_last_time_retrieved[key] = 0
            self.id_nmb_button_down[key] = 0 if control_type == "button" else None
            self.id_nmb_scan_cycles[key] = 0

        
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
        
        if self.name_device not in dev_info[1].decode():
            print(f"Device mismatch: name_device={self.name_device} and get_device_info={dev_info[1].decode()}")
            return False
        else:
            return True
        
        
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
        assert(self.check_device_id(is_input=True))
        assert(self.check_device_id(is_input=False))
        
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
            
            id_control = self.get_control_name(idx_control)
            
            # Process the inputs
            if self.id_config[id_control][1] == "slider":
                self.id_value[id_control] = val_control / 127.0
                self.id_last_time_scanned[id_control] = time.time()
                
                
            elif self.id_config[id_control][1] == "button":
                if type_control == self.button_down:
                    self.id_last_time_scanned[id_control] = time.time()
                    self.id_nmb_button_down[id_control] += 1
                    self.id_value[id_control] = True
                else:
                    self.id_value[id_control] = False
    
            
    def get(self, id_control, val_min=0, val_max=1, val_default=False, button_mode='was_pressed'):
        # Asserts
        if id_control not in self.id_config:
            print(f"Warning! {id_control} is unknown. Returning val_default")
            return val_default
        # Assert that button mode is correct if button
        assert button_mode in ['is_pressed', 'was_pressed', 'toggle']
        if val_default:
            assert val_default >= val_min and val_default <= val_max
        
        if self.autodetect_varname:
            if self.id_nmb_scan_cycles[id_control] <= 2:
            # self.id_nmb_scan_cycles
            # if id_control not in self.id_name:
                # Inspecting the stack to find the variable name
                frame = inspect.currentframe()
                try:
                    outer_frame = frame.f_back
                    call_line = outer_frame.f_lineno
                    source_lines, starting_line = inspect.getsourcelines(outer_frame)
                    line_index = call_line - starting_line
                    call_code = source_lines[line_index-1].strip()
        
                    # Extracting the variable name
                    variable_name = call_code.split('=')[0].strip()
                except Exception:
                    variable_name = "autodetect failed"
                finally:
                    del frame  # Prevent reference cycles
                
                if self.id_nmb_scan_cycles[id_control] == 1:
                    assert variable_name == self.id_name[id_control], f"Double assignment for {id_control}: {variable_name} and {self.id_name[id_control]}"
                self.id_nmb_scan_cycles[id_control] += 1
                self.id_name[id_control] = variable_name
        
        # Scan new inputs
        try:
            self.scan_inputs()
        except Exception as e:
            print(f"scan_inputs raised: {e}")
        
        # Process slider
        if self.id_config[id_control][1] == "slider":
            if val_default is False:
                val_default = 0.5 * (val_min + val_max)
            if self.id_last_time_scanned[id_control] == 0:
                val_return = val_default
            else:
                val_return = val_min + (val_max-val_min) * self.id_value[id_control]
        
        # Process button
        elif self.id_config[id_control][1] == "button":
            if button_mode == 'is_pressed':
                val_return = self.id_value[id_control]
            elif button_mode == "was_pressed":
                val_return = self.id_last_time_scanned[id_control] > self.id_last_time_retrieved[id_control]
            elif button_mode == "toggle":
                val_return = np.mod(self.id_nmb_button_down[id_control]+1,2) == 0
                # Set LED
                self.set_led(id_control, val_return)
                
        self.id_last_time_retrieved[id_control] = time.time()
        
        return val_return
        
    def set_led(self, id_control, state):
        if self.simulate_device:
            return
        assert id_control in self.id_config
        assert self.id_config[id_control][1] == "button"
        self.midi_out.write([[[self.button_down, self.id_config[id_control][0], state, 0], 0]])
        
    def reset_all_leds(self):
        for id_control in self.id_config:
            if self.id_config[id_control][1] == "button":
                self.set_led(id_control, False)
                
    def show(self):
        """
        shows the assignemnet of the id_controls on the midi device
        """
        # Extract letters and numbers
        letters = sorted(set(key[0] for key in self.id_config.keys()))
        max_num = max(int(key[1]) for key in self.id_config.keys())
        
        # Determine the maximum width of each column
        max_widths = {letter: max(len(self.id_name.get(f"{letter}{num}", '-')) for num in range(max_num + 1)) for letter in letters}
        
        # Print header row
        header_row = '   ' + ' '.join(letter.center(max_widths[letter]) for letter in letters)
        print(header_row)
        print('   ' + '+'.join('-' * max_widths[letter] for letter in letters))
        
        # Create the grid with left header
        for num in range(max_num + 1):
            row = f"{num} |"
            for letter in letters:
                key = f"{letter}{num}"
                row += self.id_name.get(key, '-').center(max_widths[letter]) + '|'
            print(row)
                    
if __name__ == "__main__":
    import lunar_tools as lt
    import time
    akai_lpd8 = MidiInput(device_name="akai_lpd8")
    
    while True:
        time.sleep(0.1)
        variable1 = akai_lpd8.get("A0", button_mode='toggle') # toggle switches the state with every press between on and off
        do_baba = akai_lpd8.get("B1", button_mode='is_pressed') # is_pressed checks if the button is pressed down at the moment
        strange_effect = akai_lpd8.get("C0", button_mode='was_pressed') # was_pressed checks if the button was pressed since we checked last time
        supermorph = akai_lpd8.get("E1", val_min=3, val_max=6, val_default=5) # e0 is a slider float between val_min and val_max
        print(f"variable1: {variable1}, do_baba: {do_baba}, strange_effect: {strange_effect}, supermorph: {supermorph}")
        
    akai_lpd8.show()

