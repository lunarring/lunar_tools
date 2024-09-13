try:
    from pynput import keyboard
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
from lunar_tools.midi import check_any_midi_device_connected
from lunar_tools.midi import MidiInput

import warnings

warnings.warn(
    "The MetaInput and KeyboardInput classes are deprecated and will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2
)

     
#%%
class MetaInput:
    """ Automatically selects the control method based on what is plugged in. Using keyboard as fallback. """
    
    def __init__(self,
                 force_device = None
                 ):
        self.valid_get_args = ["val_min", "val_max", "val_default", "button_mode", "variable_name"]
        self.autoshow_names = True
        self.id_name = {}
        self.id_nmb_scan_cycles = {}
        if force_device:
            if "keyb" in force_device:
                device_name = "keyboard"
            else:
                device_name = force_device
        else:
            device_name = check_any_midi_device_connected()
            if device_name is None:
                device_name = "keyboard"
        
        self.device_name = device_name
        
        if self.device_name == "keyboard":
            self.control_device = KeyboardInput()
        else:
            self.control_device = MidiInput(self.device_name)
        self.last_values = {}
        print(f"MetaInput: using exclusively {self.device_name}")
        
    def get(self, **kwargs):
        return_val = None
        device_control_key = f"{self.device_name}"
        if device_control_key in kwargs:
            valid_kwargs = {k: v for k, v in kwargs.items() if k in self.valid_get_args}

            alpha_num = kwargs[device_control_key]
            # Autodetect varname feature
            if self.id_nmb_scan_cycles.get(alpha_num, 0) <= 2:
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
                
                if self.id_nmb_scan_cycles.get(alpha_num, 0) == 1:
                    assert variable_name == self.id_name.get(alpha_num, ""), f"Double assignment for {alpha_num}: {variable_name} and {self.id_name.get(alpha_num, '')}"
                self.id_nmb_scan_cycles[alpha_num] = self.id_nmb_scan_cycles.get(alpha_num, 0) + 1
                self.id_name[alpha_num] = variable_name
            else:
                variable_name = self.id_name[alpha_num]
        
            if self.id_nmb_scan_cycles[alpha_num] == 2 and self.autoshow_names:
                self.control_device.show()
                self.autoshow_names = False
            
            return_val = self.control_device.get(alpha_num, variable_name=variable_name, **valid_kwargs)
            self.last_values[alpha_num] = return_val
        else:
            if "button_mode" in kwargs:
                if "val_default" in kwargs:
                    return_val = kwargs['val_default']
                else:
                    return_val = False
            else: 
                if "val_default" in kwargs:
                    return_val = kwargs['val_default']
                elif "val_min" in kwargs and "val_max" in kwargs:
                    return_val = (kwargs['val_max'] - kwargs['val_min'])/2
                else:
                    return_val = 0.0
        
        
        return return_val
            # raise ValueError(f"Device '{self.device_name}' not specified in arguments, and it is the active connected device.")

    def show(self):
        self.control_device.show()

    def print_state(self):
        self._display_or_save()

    def show_state(self, filename):
        self._display_or_save(filename)

    def _display_or_save(self, filename=None):
        output_lines = [f"Device: {self.device_name}"]
        for key in sorted(self.id_name.keys()):
            output_lines.append(f"{key} | {self.id_name[key]} | {self.last_values[key]}")
        if filename:
            with open(filename, 'w') as file:
                file.write('\n'.join(output_lines))
        else:
            print('\n'.join(output_lines))
#%%
class KeyboardInput:
    """ A class to track keyboard inputs, including emulated sliders. """

    def __init__(self):
        """ Initializes the keyboard listener and dictionaries to store pressed keys and their states. """
        self.pressed_keys = {}
        self.key_last_time_pressed = {}
        self.key_last_time_released = {}
        self.id_name = {}
        self.id_nmb_scan_cycles = {}
        self.key_press_count = {}
        self.released_once_flags = {}
        self.pressed_once_flags = {}  # Tracks if key was pressed down once
        self.pressed_once_ready = {}  # Tracks if key is ready to be pressed down once again
        self.active_slider = None
        self.slider_values = {}
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.nmb_steps = 64
        self.valid_button_modes = ['held_down', 'released_once', 'toggle', 'pressed_once']

    def on_press(self, key):
        """ Adds a pressed key to the dictionary of pressed keys and updates its state. """
        key_name = self.get_key_name(key)
        self.pressed_keys[key_name] = True
        self.key_last_time_pressed[key_name] = time.time()
        self.key_press_count[key_name] = self.key_press_count.get(key_name, 0) + 1
        self.released_once_flags[key_name] = False

        if self.pressed_once_ready.get(key_name, True):  # Check if ready for pressed_once
            self.pressed_once_flags[key_name] = True
            self.pressed_once_ready[key_name] = False   # Reset readiness

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
        # print(f"was released: {key} {self.key_last_time_released[key_name]}")
        self.released_once_flags[key_name] = True
        self.pressed_once_ready[key_name] = True  # Set ready for next pressed_once

    def get_key_name(self, key):
        """ Returns the character of the key if available, else returns the key name. """
        if hasattr(key, 'char'):
            return key.char
        else:
            return key.name

    def get(self, key, val_min=None, val_max=None, val_default=None, button_mode=None, variable_name=None):
        """ Checks the state of a specific key based on the requested mode (button or slider). """
        key = key.lower()
    
        # Autodetect variable name if not provided
        if variable_name is None:
            if self.id_nmb_scan_cycles.get(key, 0) <= 2:
                frame = inspect.currentframe()
                try:
                    outer_frame = frame.f_back
                    call_line = outer_frame.f_lineno
                    source_lines, starting_line = inspect.getsourcelines(outer_frame)
                    line_index = call_line - starting_line
                    call_code = source_lines[line_index - 1].strip()
    
                    # Extracting the variable name
                    variable_name = call_code.split('=')[0].strip()
                except Exception:
                    variable_name = "autodetect failed"
                finally:
                    del frame  # Prevent reference cycles
    
                if self.id_nmb_scan_cycles.get(key, 0) == 1:
                    if button_mode in ['pressed_once', 'released_once']:
                        variable_name += f" ({button_mode})"
                    else:
                        assert variable_name == self.id_name.get(key, ""), f"Double assignment for {key}: {variable_name} and {self.id_name.get(key, '')}"
                self.id_nmb_scan_cycles[key] = self.id_nmb_scan_cycles.get(key, 0) + 1
                self.id_name[key] = variable_name
    
        # Assertions to ensure correct parameter usage
        if val_min is not None and val_max is not None:
            assert button_mode is None, "Button mode should not be provided for slider usage"
            step = (val_max - val_min) / self.nmb_steps
            if key not in self.slider_values:
                self.slider_values[key] = {
                    'val_min': val_min, 'val_max': val_max, 'step': step,
                    'value': val_default if val_default is not None else (val_min + val_max) / 2
                }
            return self.slider_values[key]['value']
        elif button_mode is not None:
            assert val_min is None and val_max is None, "val_min and val_max should not be provided for button usage"
            assert button_mode in self.valid_button_modes, "Invalid button mode"
            if val_default:
                assert button_mode == "toggle", "if val_default is set True, button_mode must be 'toggle'"
            if button_mode == 'held_down':
                return self.pressed_keys.get(key, False)
            elif button_mode == 'released_once':
                released_once = self.released_once_flags.get(key, False)
                if released_once:
                    self.released_once_flags[key] = False
                return released_once
            elif button_mode == 'pressed_once':
                pressed_once = self.pressed_once_flags.get(key, False)
                if pressed_once:
                    self.pressed_once_flags[key] = False
                return pressed_once
            elif button_mode == 'toggle':
                toggle_state = np.mod(self.key_press_count.get(key, 0), 2) == 1
                if val_default:
                    toggle_state = not toggle_state
                return toggle_state
        else:
            raise ValueError("Invalid parameters: provide either val_min and val_max for a slider or button_mode for a button")
    
    def show(self):
        for key in sorted(self.id_name):
            print(f"{key}: {self.id_name[key]}")


# Example of usage

if __name__ == "__main__":
    keyboard_input = KeyboardInput()
    # ... In some update loop
    while True:
        time.sleep(0.1)
        x1 = keyboard_input.get('s', button_mode='toggle', val_default=True)
        # x2 = keyboard_input.get('s', button_mode='released_once')
        print(f"x1 {x1}")

        
    
