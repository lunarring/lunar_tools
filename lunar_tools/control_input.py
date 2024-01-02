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
import subprocess
import re
import usb.core     # pip install pyusb
from sys import platform

def get_midi_device_vendor_product_ids(name):
    # Initialize the result dictionary
    midi_mix_ids = {}

    try:
        # Run the lsusb command and capture the output
        lsusb_output = subprocess.check_output(['lsusb'], text=True)

        # Regular expression to match the vendor and product IDs of device called <name>
        regex = f'ID (\w+:\w+).+{name}'

        # Find all matches
        matches = re.findall(regex, lsusb_output)

        for match in matches:
            # Split the match into vendor and product IDs
            vendor_id, product_id = match.split(':')

            # Add to the dictionary
            return {'vendor_id': int(vendor_id, 16), 'product_id': int(product_id, 16)}

        return midi_mix_ids
    except subprocess.CalledProcessError as e:
        print(f"Error executing lsusb: {e}")
        return {}
    
def check_midi_device_connected_pyusb(device_code):
    if len(device_code) > 0:
        # Look for all devices connected
        devices = usb.core.find(find_all=True)
    
        # Iterate through each device and check if <device_code> device connected
        for device in devices:
            if device.idVendor == device_code['vendor_id'] and device.idProduct == device_code['product_id']:
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
                 do_auto_reconnect=False
                 ):
        
        # get operating system
        if (platform == "linux" or platform == "linux2"):
            self.os_name = 'linux'
        elif platform == "darwin":
            self.os_name = 'macos'
        else:
            self.os_name = 'windows'
        self.do_auto_reconnect = do_auto_reconnect
        
        self.simulate_device = False
        self.device_name = device_name
        self.allow_fail = allow_fail
        self.device_id_input = device_id_input
        self.device_id_output = device_id_output
        self.init_device_config(enforce_local_config)
        self.init_device_hardware_code()
        self.init_vars()
        self.init_midi()
        self.reset_all_leds()
        self.autodetect_varname = True
        self.autoshow_names = True
        
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
        self.reverse_control_name = {(v[0], v[1]): k for k, v in self.id_config.items()}
        
    def init_device_hardware_code(self):
        if self.os_name == 'linux':
            self.device_hardware_code = get_midi_device_vendor_product_ids(self.name_device)

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

    def get_control_name(self, idx_control, type_control):
        if type_control == self.button_down or type_control == self.button_release:
            str_type_control = "button"
        else:
            str_type_control = "slider"
        
        if (idx_control, str_type_control) in self.reverse_control_name:
            return self.reverse_control_name[(idx_control, str_type_control)]
        else:
            return None

    def scan_inputs(self):
        if self.simulate_device:
            return
        
        # Gather all inputs that arrived in the meantime
        while True:
            if self.os_name == 'linux' and self.do_auto_reconnect:
                time.sleep(1e-3)
                is_midi_device_connected = check_midi_device_connected_pyusb(self.device_hardware_code)
                print(f'{is_midi_device_connected}')
                if not is_midi_device_connected:
                    print(f'{self.device_name} has disconnected. trying to reconnect...')
                    self.init_midi()                
            else:
                is_midi_device_connected = True
                
            if is_midi_device_connected:
                input_last = self.midi_in.read(1)
            else:
                break
                
            if input_last == []:
                break
            type_control = input_last[0][0][0]
            idx_control = input_last[0][0][1]
            val_control = input_last[0][0][2]
            
            alpha_num = self.get_control_name(idx_control, type_control)
            
            # Process the inputs
            if self.id_config[alpha_num][1] == "slider":
                self.id_value[alpha_num] = val_control / 127.0
                self.id_last_time_scanned[alpha_num] = time.time()
                
                
            elif self.id_config[alpha_num][1] == "button":
                if type_control == self.button_down:
                    self.id_last_time_scanned[alpha_num] = time.time()
                    self.id_nmb_button_down[alpha_num] += 1
                    self.id_value[alpha_num] = True
                else:
                    self.id_value[alpha_num] = False
    
            
    def get(self, alpha_num, val_min=None, val_max=None, val_default=False, button_mode=None):
        # Asserts
        if alpha_num not in self.id_config:
            print(f"Warning! {alpha_num} is unknown. Returning val_default")
            return val_default
        # Assert that button mode is correct if button
        if val_default:
            assert val_default >= val_min and val_default <= val_max
        # Assert correct type
        if self.id_config[alpha_num][1] == "button":
            assert val_min is None and val_max is None, f"{alpha_num} is a button cannot have val_min/val_max"
        if self.id_config[alpha_num][1] == "slider":
            assert button_mode is None, f"{alpha_num} is a slider cannot have button_mode"
        
        # Set default args
        if val_min is None:
            val_min = 0
        if val_max is None:
            val_max = 1
        if button_mode is None:
            button_mode = 'was_pressed'
        assert button_mode in ['is_pressed', 'was_pressed', 'toggle']
        
        if self.autodetect_varname:
            if self.id_nmb_scan_cycles[alpha_num] <= 2:
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
                
                if self.id_nmb_scan_cycles[alpha_num] == 1:
                    assert variable_name == self.id_name[alpha_num], f"Double assignment for {alpha_num}: {variable_name} and {self.id_name[alpha_num]}"
                self.id_nmb_scan_cycles[alpha_num] += 1
                self.id_name[alpha_num] = variable_name
            
            if self.id_nmb_scan_cycles[alpha_num] == 2 and self.autoshow_names:
                self.show()
                self.autoshow_names = False
        
        # Scan new inputs
        try:
            # so far only linux supported for auto device reconnect/disconnect handling
            self.scan_inputs()
        except Exception as e:
            print(f"scan_inputs raised: {e}")
        
        # Process slider
        if self.id_config[alpha_num][1] == "slider":
            if val_default is False:
                val_default = 0.5 * (val_min + val_max)
            if self.id_last_time_scanned[alpha_num] == 0:
                val_return = val_default
            else:
                val_return = val_min + (val_max-val_min) * self.id_value[alpha_num]
        
        # Process button
        elif self.id_config[alpha_num][1] == "button":
            if button_mode == 'is_pressed':
                val_return = self.id_value[alpha_num]
            elif button_mode == "was_pressed":
                val_return = self.id_last_time_scanned[alpha_num] > self.id_last_time_retrieved[alpha_num]
            elif button_mode == "toggle":
                val_return = np.mod(self.id_nmb_button_down[alpha_num]+1,2) == 0
                # Set LED
                self.set_led(alpha_num, val_return)
                
        self.id_last_time_retrieved[alpha_num] = time.time()
        
        return val_return
        
    def set_led(self, alpha_num, state):
        if self.simulate_device:
            return
        assert alpha_num in self.id_config
        assert self.id_config[alpha_num][1] == "button"
        self.midi_out.write([[[self.button_down, self.id_config[alpha_num][0], state, 0], 0]])
        
    def reset_all_leds(self):
        for alpha_num in self.id_config:
            if self.id_config[alpha_num][1] == "button":
                self.set_led(alpha_num, False)
                
    def show(self):
        """
        shows the assignemnet of the alpha_nums on the midi device
        """
        # Extract letters and numbers
        letters = sorted(set(key[0] for key in self.id_config.keys()))
        max_num = max(int(key[1]) for key in self.id_config.keys())
        
        # Determine the maximum width of each column
        max_widths = {letter: max(len(self.id_name.get(f"{letter}{num}", '-')) for num in range(max_num + 1)) for letter in letters}
        
        # Print header row
        header_row = '   ' + ' '.join(letter.center(max_widths[letter]) for letter in letters)
        print('\n')
        print(header_row)
        print('   ' + '+'.join('-' * max_widths[letter] for letter in letters))
        
        # Create the grid with left header
        for num in range(max_num + 1):
            row = f"{num} |"
            for letter in letters:
                key = f"{letter}{num}"
                row += self.id_name.get(key, '-').center(max_widths[letter]) + '|'
            print(row)
        print('\n')
        
        
#%%
class KeyboardInput:
    """ A class to track keyboard inputs, including emulated sliders. """

    def __init__(self):
        """ Initializes the keyboard listener and dictionaries to store pressed keys and their states. """
        self.pressed_keys = {}
        self.key_last_time_pressed = {}
        self.key_last_time_released = {}
        self.key_press_count = {}
        self.was_pressed_flags = {}
        self.active_slider = None
        self.slider_values = {}
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        """ Adds a pressed key to the dictionary of pressed keys and updates its state. """
        key_name = self.get_key_name(key)
        self.pressed_keys[key_name] = True
        self.key_last_time_pressed[key_name] = time.time()
        self.key_press_count[key_name] = self.key_press_count.get(key_name, 0) + 1
        self.was_pressed_flags[key_name] = False

        # Activate slider or adjust its value
        if key_name in self.slider_values:
            self.active_slider = key_name
        elif self.active_slider:
            slider_info = self.slider_values[self.active_slider]
            if key == keyboard.Key.up:
                self.slider_values[self.active_slider]['value'] = min(
                    slider_info['val_max'], 
                    slider_info['value'] + slider_info['step']
                )
            elif key == keyboard.Key.down:
                self.slider_values[self.active_slider]['value'] = max(
                    slider_info['val_min'], 
                    slider_info['value'] - slider_info['step']
                )

    def on_release(self, key):
        """ Handles key release events and updates the state of the key. """
        key_name = self.get_key_name(key)
        self.pressed_keys[key_name] = False
        self.key_last_time_released[key_name] = time.time()
        self.was_pressed_flags[key_name] = True

    def get_key_name(self, key):
        """ Returns the character of the key if available, else returns the key name. """
        if hasattr(key, 'char'):
            return key.char
        else:
            return key.name

    def get(self, key, val_min=None, val_max=None, val_default=None, button_mode=None, step=0.1):
        """ Checks the state of a specific key based on the requested mode (button or slider). """
        key = key.lower()
        if val_min is not None and val_max is not None:
            # Slider mode
            if key not in self.slider_values:
                self.slider_values[key] = {
                    'val_min': val_min, 'val_max': val_max, 'step': step,
                    'value': val_default if val_default is not None else (val_min + val_max) / 2
                }
            return self.slider_values[key]['value']
        else:
            # Button mode
            if button_mode == 'is_pressed':
                return self.pressed_keys.get(key, False)
            elif button_mode == 'was_pressed':
                was_pressed = self.was_pressed_flags.get(key, False)
                if was_pressed:
                    self.was_pressed_flags[key] = False
                return was_pressed
            elif button_mode == 'toggle':
                return np.mod(self.key_press_count.get(key, 0), 2) == 1
            else:
                raise ValueError(f"Invalid button mode: {button_mode}")




#%%

# Example of usage
if __name__ == "__main__":
    keyboard_input = KeyboardInput()
    # ... In some update loop
    while True:
        time.sleep(0.1)
        a = keyboard_input.get('a', button_mode='is_pressed')
        s = keyboard_input.get('s', button_mode='was_pressed')
        d = keyboard_input.get('d', button_mode='toggle')
        x = keyboard_input.get('x', val_min=3, val_max=6)
        print(f"{a} {s} {d} {x}")
        
        
                    
if __name__ == "__main__midi":
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
                    
if __name__ == "__main__midimix":
    import lunar_tools as lt
    import time
    akai_lpd8 = MidiInput(device_name="akai_midimix")
    
    while True:
        time.sleep(0.1)
        a0 = akai_lpd8.get("A0", val_min=3, val_max=6, val_default=5)
        print(a0)
        
    
