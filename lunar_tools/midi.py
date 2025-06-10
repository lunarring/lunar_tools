try:
    from pygame import midi
except Exception as e:
    print(f"IMPORT FAIL: {e}")
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
import platform
from datetime import datetime


def get_midi_device_vendor_product_ids(system_device_name):
    # Initialize the result dictionary
    vendor_product_ids = {}

    try:
        os_system = platform.system()
        if os_system == 'Linux':
            # Run the lsusb command for Linux
            usb_output = subprocess.check_output(['lsusb'], text=True)
            regex = f'ID (\\w+:\\w+).+{system_device_name}'
        elif os_system == 'Darwin':
            # Run the system_profiler command for macOS
            usb_output = subprocess.check_output(['system_profiler', 'SPUSBDataType'], text=True)
            regex = f'{system_device_name}.*?\\n.*?Product ID: (0x\\w+)\\n.*?Vendor ID: (0x\\w+)'
        elif os_system == 'Windows':
            # Use WMIC command to retrieve USB device details on Windows
            usb_output = subprocess.check_output(['wmic', 'path', 'Win32_USBHub', 'get', 'DeviceID'], text=True, stderr=subprocess.DEVNULL)
            regex = f'{system_device_name}.*?VID_([0-9A-Fa-f]{{4}})&PID_([0-9A-Fa-f]{{4}})'
        else:
            print("Unsupported operating system.")
            return vendor_product_ids

        # Find all matches
        matches = re.findall(regex, usb_output, re.IGNORECASE | re.DOTALL)

        for match in matches:
            if os_system == 'Linux':
                # Split the match into vendor and product IDs for Linux
                vendor_id, product_id = match.split(':')
            elif os_system == 'Darwin':
                # Assign vendor and product IDs for macOS
                product_id, vendor_id = match
            elif os_system == 'Windows':
                # On Windows, match returns VID and PID groups
                vendor_id, product_id = match

            # Convert hexadecimal to integer and add to the dictionary
            vendor_product_ids = {'vendor_id': int(vendor_id, 16), 'product_id': int(product_id, 16)}
            break

        return vendor_product_ids

    except Exception as e:
        print(f"An error occurred: {e}")
        return vendor_product_ids

    
def check_midi_device_connected_pyusb(device_code):
    if len(device_code) > 0:
        # Look for all devices connected
        devices = usb.core.find(find_all=True)
    
        # Iterate through each device and check if <device_code> device connected
        for device in devices:
            if device.idVendor == device_code['vendor_id'] and device.idProduct == device_code['product_id']:
                return True

    return False    


def check_any_midi_device_connected():
    config_path = pkg_resources.resource_filename('lunar_tools', 'midi_configs')
    for filename in os.listdir(config_path):
        if filename.endswith(".yml"):
            with open(os.path.join(config_path, filename), 'r') as file:
                config = yaml.safe_load(file)
                device_name = config['name_device']
                device_ids = get_midi_device_vendor_product_ids(device_name)
                if len(device_ids) > 0:
                    return filename.split(".yml")[0]
    return None

#%%
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
        if platform.system() == 'Linux':
            self.os_name = 'linux'
        elif platform.system() == 'Darwin':
            self.os_name = 'macos'
        elif platform.system() == 'Windows':
            self.os_name = 'windows'
        else:
            raise NotImplementedError("Only Linux, MacOS and Windows supported at the moment.")            
        
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
        self.valid_button_modes = ['held_down', 'released_once', 'toggle', 'pressed_once']
        
        
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
        self.system_device_name = config['name_device']
        self.button_down = config['button_down']
        self.button_release = config['button_release']
        self.id_config = config['controls']

        # Reverse for last lookup
        self.reverse_control_name = {(v[0], v[1]): k for k, v in self.id_config.items()}
        
    def init_device_hardware_code(self):
        self.device_hardware_code = get_midi_device_vendor_product_ids(self.system_device_name)

    def init_vars(self):
        # Initializes all variables
        self.id_value = {}
        self.id_last_time_scanned = {}
        self.id_last_time_retrieved = {}
        self.id_nmb_button_down = {}
        self.id_nmb_scan_cycles = {}
        self.id_name = {}
        self.id_val_min = {}  # Store val_min for each control
        self.id_val_max = {}  # Store val_max for each control
        self.id_val_default = {}  # Store val_default for each control
        self.released_once_flags = {}  # Adding a new dictionary to track released_once flags
        for key in self.id_config:
            control_type = self.id_config[key][1]
            self.released_once_flags[key] = False 
            self.id_value[key] = False if control_type == "button" else 0.0
            self.id_last_time_scanned[key] = 0
            self.id_last_time_retrieved[key] = 0
            self.id_nmb_button_down[key] = 0 if control_type == "button" else None
            self.id_nmb_scan_cycles[key] = 0
            # Initialize scaling parameters
            if control_type == "slider":
                self.id_val_min[key] = 0
                self.id_val_max[key] = 1
                self.id_val_default[key] = 0.5
            else:
                self.id_val_min[key] = None
                self.id_val_max[key] = None
                self.id_val_default[key] = False

    def compare_device_names(self, dev_info):
        if self.system_device_name in dev_info[1].decode():
            return True
        elif dev_info[1].decode() in self.system_device_name:
            return True
        else:
            return False
        
    def auto_determine_device_id(self, is_input):
        dev_count = midi.get_count()
        device_id = None
        for i in range(dev_count):
            dev_info = midi.get_device_info(i)
            if self.compare_device_names(dev_info):
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
            print("Simulating midi device!")
            return
        elif not self.allow_fail and self.device_id_input is None:
            raise ValueError("Device init failed! If you want to simulate, set allow_fail=True")
        if is_input:
            dev_info = midi.get_device_info(self.device_id_input)
        else:
            dev_info = midi.get_device_info(self.device_id_output)
        
        if not self.compare_device_names(dev_info):
            print(f"Device mismatch: name_device={self.system_device_name} and get_device_info={dev_info[1].decode()}")
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
        is_valid_input = self.check_device_id(is_input=True)
        is_valid_output = self.check_device_id(is_input=False)
        if not self.simulate_device: 
            assert is_valid_input
            assert is_valid_output
        
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
            if self.os_name in ['linux', 'windows'] and self.do_auto_reconnect:
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
                    self.released_once_flags[alpha_num] = True 
    
            
    def get(self, alpha_num, val_min=None, val_max=None, val_default=False, button_mode=None, variable_name=None):
        """
        Retrieves the value of a control input based on its alphanumeric identifier.

        Parameters:
        - alpha_num (str): The alphanumeric identifier of the control input.
        - val_min (float, optional): The minimum value for sliders. Defaults to None.
        - val_max (float, optional): The maximum value for sliders. Defaults to None.
        - val_default (bool or float, optional): The default value for the control input. For buttons, it's a boolean indicating pressed state; for sliders, it's their position. Defaults to False.
        - button_mode (str, optional): The mode of button operation. Can be one of ['held_down', 'released_once', 'toggle', 'pressed_once']. Defaults to None.
        - variable_name (str, optional): The name of the variable to override the autodetection. Defaults to None.

        Returns:
        - The current value of the control input based on its configuration and the provided parameters.
        """
        # Asserts
        if alpha_num not in self.id_config:
            print(f"Warning! {alpha_num} is unknown. Returning val_default")
            return val_default
        # Assert that button mode is correct if button
        if val_default and button_mode is None:
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
            button_mode = 'released_once'
        assert button_mode in self.valid_button_modes
        
        # Store scaling parameters for later use in get_parameters_dict
        if alpha_num in self.id_config:
            if self.id_config[alpha_num][1] == "slider":
                self.id_val_min[alpha_num] = val_min
                self.id_val_max[alpha_num] = val_max
                self.id_val_default[alpha_num] = val_default
            else:  # button
                self.id_val_default[alpha_num] = val_default
        
        if self.autodetect_varname and variable_name is None:
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
            
        elif variable_name is not None:
            self.id_name[alpha_num] = variable_name
    
        if self.id_nmb_scan_cycles[alpha_num] == 2 and self.autoshow_names:
            self.show()
            self.autoshow_names = False
        # Scan new inputs
        try:
            # Auto device reconnect/disconnect handling for Linux and Windows
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
            if val_default:
                assert button_mode == "toggle", "if val_default is set True, button_mode must be 'toggle'"
            if button_mode == 'held_down':
                val_return = self.id_value[alpha_num]
            elif button_mode == "pressed_once":
                val_return = self.id_value[alpha_num] and self.id_last_time_scanned[alpha_num] > self.id_last_time_retrieved[alpha_num]
                if val_return:
                    self.id_value[alpha_num] = False  # Reset the value after being processed
            elif button_mode == "released_once":
                val_return = self.released_once_flags.get(alpha_num, False)
                if val_return:
                    self.released_once_flags[alpha_num] = False # Reset the last time scanned after being processed
            elif button_mode == "toggle":
                if val_default is None:
                    val_default = False
                val_return = np.mod(self.id_nmb_button_down[alpha_num]+1, 2) == val_default
                # Set LED
                self.set_led(alpha_num, val_return)
                
        self.id_last_time_retrieved[alpha_num] = time.time()
        
        return val_return
    
    def get_parameters_dict(self, **misc):
        """ Return a dictionary with all assigned parameters and their values. Additional information can be passed as kwargs"""
        parameters = []
        for id_, name in self.id_name.items():
            if self.id_config[id_][1] == "slider":
                # Scale the raw value using stored min/max parameters
                if self.id_last_time_scanned[id_] == 0:
                    # No input received yet, use default value
                    scaled_value = self.id_val_default[id_]
                else:
                    # Scale the raw value (0-1) to the desired range
                    val_min = self.id_val_min[id_]
                    val_max = self.id_val_max[id_]
                    scaled_value = val_min + (val_max - val_min) * self.id_value[id_]
                parameters.append({'id': id_, 'name': name, 'value': scaled_value})
            else:
                # For buttons, return the boolean value as-is
                value = self.id_value[id_]
                parameters.append({'id': id_, 'name': name, 'value': value})
        if misc:
            parameters.append(misc)
        return parameters    

        
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
        
    def print_values(self):
        print('id\tname\tvalue\n')
        for id_, name in self.id_name.items():
            value = self.id_value[id_]
            print(f'{id_}\t{name}\t{value}')
        print('\n')

       


#%%

       
                    
if __name__ == "__main__":
    import time
    akai_lpd8 = MidiInput(device_name="akai_lpd8")
    
    while True:
        time.sleep(0.2)
        # released_once checks if the button was pressed since we checked last time
        z = akai_lpd8.get("A0", button_mode='toggle', val_default=True) 
        # released_once checks if the button was pressed since we checked last time
        x = akai_lpd8.get("B0", button_mode='released_once') 
        #  e0 is a slider float between val_min and val_max
        slider = akai_lpd8.get("E0", val_min=3, val_max=6, val_default=5) #
        print(f"z: {z}, x: {x} slider: {slider}")
        
    akai_lpd8.show()
                    
if __name__ == "__main__x":
    import lunar_tools as lt
    import time
    akai_midimix = MidiInput(device_name="akai_midimix")
    
    
    while True:
        time.sleep(0.1)
        a0 = akai_midimix.get("A0", val_min=3, val_max=6, val_default=5)
        ba = akai_midimix.get("A3", button_mode="toggle")
        print(ba)

