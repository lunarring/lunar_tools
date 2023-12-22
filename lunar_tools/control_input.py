from pynput import keyboard
from pygame import midi
import numpy as np
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
    """ A class to track midi inputs. """
    def __init__(self,
                 device_id_input=None,
                 device_id_output=None,
                 ):
    
        self.device_id_input = device_id_input
        self.device_id_output = device_id_output
        self.init_device_config()
        self.init_midi()


    def init_device_config(self):
        # THIS SHOULD BE EXTERNAL
        self.name_device = "LPD8 MIDI"
        self.button_down = 144
        self.button_release = 128
        self.control_name = {}
        self.control_name["A0"] = [40, "button"]    
        self.control_name["A1"] = [36, "button"]    
        self.control_name["B0"] = [41, "button"]    
        self.control_name["B1"] = [37, "button"]    
        self.control_name["C0"] = [42, "button"]    
        self.control_name["C1"] = [38, "button"]    
        self.control_name["D0"] = [43, "button"]    
        self.control_name["D1"] = [39, "button"]    
        self.control_name["E0"] = [1, "slider"]       
        self.control_name["E1"] = [5, "slider"]       
        self.control_name["F0"] = [2, "slider"]       
        self.control_name["F1"] = [6, "slider"]       
        self.control_name["G0"] = [3, "slider"]       
        self.control_name["G1"] = [7, "slider"]       
        self.control_name["H0"] = [4, "slider"]    
        self.control_name["H1"] = [8, "slider"]
        # -----
        
        # Reverse for last lookup
        self.reverse_control_name = {v[0]: k for k, v in self.control_name.items()}

        # Initialize the last_value last_time_scanned last_time_retrieved dicts
        self.last_value = {}
        self.last_time_scanned = {}
        self.last_time_retrieved = {}
        self.nmb_button_down = {}
        for key in self.control_name:
            if self.control_name[key][1] == "button":
                self.last_value[key] = False
            else:
                self.last_value[key] = 0.0
            self.last_time_scanned[key] = 0
            self.last_time_retrieved[key] = 0
            self.nmb_button_down[key] = 0
            
                
        

        
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
        
        # Init midi in and out
        self.midi_in = midi.Input(self.device_id_input)
        self.midi_out = midi.Output(self.device_id_output)

    def get_control_name(self, idx_control):
        if idx_control in self.reverse_control_name:
            return self.reverse_control_name[idx_control]
        else:
            return None

    def scan_inputs(self):
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
            if self.control_name[name_control][1] == "slider":
                self.last_value[name_control] = val_control / 127.0
                self.last_time_scanned[name_control] = time.time()
                
            elif self.control_name[name_control][1] == "button":
                if type_control == self.button_down:
                    self.last_time_scanned[name_control] = time.time()
                    self.nmb_button_down[name_control] += 1
                    self.last_value[name_control] = True
                else:
                    self.last_value[name_control] = False
                    
                
            
    def get(self, name_control, val_min=0, val_max=1, val_default=False, button_mode='was_pressed'):
        # Asserts
        if name_control not in self.control_name:
            print(f"Warning! {name_control} is unknown. Returning val_default")
            return val_default
        # button mode correct if button
        assert button_mode in ['is_held', 'was_pressed', 'toggle']
        
        # Process slider
        if self.control_name[name_control][1] == "slider":
            if val_default is False:
                val_default = 0.5 * (val_min + val_max)
            if self.last_time_scanned[name_control] == 0:
                val_return = val_default
            else:
                val_return = val_min + (val_max-val_min) * self.last_value[name_control]
        
        # Process button
        elif self.control_name[name_control][1] == "button":
            if button_mode == 'is_held':
                val_return = self.last_value[name_control]
            elif button_mode == "was_pressed":
                val_return = self.last_time_scanned[name_control] > self.last_time_retrieved[name_control]
            elif button_mode == "toggle":
                val_return = np.mod(self.nmb_button_down[name_control]+1,2) == 0
                # Set LED
                self.midi_out.write([[[self.button_down, self.control_name['A0'][0], val_return, 0], 0]])
                
                
        self.last_time_retrieved[name_control] = time.time()
        
        return val_return
        
        
if __name__ == "__main__":
    self = MidiInput()
    
    while True:
        time.sleep(0.1)
        self.scan_inputs()
        x = self.get("A0", button_mode='toggle')
        print(x)
        

#%%

