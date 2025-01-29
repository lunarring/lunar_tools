import os
import platform
import numpy as np
from threading import Thread
import time
from collections import deque

def exception_handler(func):
    """
    A decorator that wraps a function to provide exception handling.

    This decorator catches any exceptions raised by the decorated function,
    prints an error message with the function name and the exception details,
    and prevents the exception from propagating further.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: A wrapped function that includes exception handling.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Exception when running {func.__name__}: {e}")
    return wrapper

def get_os_type():
    os_name = platform.system()
    if os_name == "Darwin":
        return "MacOS"
    elif os_name == "Linux":
        return "Linux"
    elif os_name == "Windows":
        return "Windows"
    else:
        raise ValueError("unsupported OS")

def get_config_path():
    os_type = get_os_type()
    if os_type in ["MacOS", "Linux"]:
        return os.path.expanduser("~/.lunar_tools_env_vars")
    elif os_type == "Windows":
        return os.path.join(os.environ['USERPROFILE'], '.lunar_tools_env_vars')
    else:
        raise ValueError("Unsupported OS")

def read_all_api_keys_from_lunar_config():
    config_path = get_config_path()
    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r') as file:
        lines = file.readlines()
        return dict(line.strip().split('=') for line in lines if line.strip())

def read_api_key_from_lunar_config(key_name):
    keys = read_all_api_keys_from_lunar_config()
    return keys.get(key_name)

def save_api_key_to_lunar_config(key_name, key_value):
    keys = read_all_api_keys_from_lunar_config()
    keys[key_name] = key_value

    config_path = get_config_path()

    with open(config_path, 'w') as file:
        for k, v in keys.items():
            file.write(f"{k}={v}\n")
            
    print(f"saved API KEY '{key_name}={key_value} in {get_config_path()}")

def read_api_key(key_name):
    """Retrieve API key directly from environment variables"""
    return os.getenv(key_name)


def delete_api_key_from_lunar_config(key_name):
    keys = read_all_api_keys_from_lunar_config()
    if key_name in keys:
        del keys[key_name]

        config_path = get_config_path()

        with open(config_path, 'w') as file:
            for k, v in keys.items():
                file.write(f"{k}={v}\n")


class SimpleNumberBuffer:
    """
    A class used to manage a buffer of numerical values with optional normalization.

    Attributes
    ----------
    buffer_size : int
        The maximum size of the buffer.
    buffer : deque
        The buffer storing numerical values.
    default_return_value : int
        The default value to return when the buffer is empty.
    normalize : bool
        A flag indicating whether to normalize the buffer values.

    Methods
    -------
    append(value)
        Appends a value to the buffer.
    get_buffer()
        Returns the buffer as a numpy array, optionally normalized.
    get_last_value()
        Returns the last value in the buffer or the default return value if the buffer is empty.
    set_buffer_size(buffer_size)
        Sets a new buffer size and adjusts the buffer accordingly.
    set_normalize(normalize)
        Sets the normalization flag.
    """

    def __init__(self, buffer_size=500, normalize=False):
        """
        Initializes the SimpleNumberBuffer with a specified buffer size and normalization flag.

        Parameters
        ----------
        buffer_size : int, optional
            The maximum size of the buffer (default is 500).
        normalize : bool, optional
            A flag indicating whether to normalize the buffer values (default is False).
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.default_return_value = 0
        self.normalize = normalize

    def append(self, value):
        """
        Appends a value to the buffer.

        Parameters
        ----------
        value : float
            The numerical value to append to the buffer.
        """
        self.buffer.append(value)

    def get_buffer(self):
        """
        Returns the buffer as a numpy array, optionally normalized.

        Returns
        -------
        numpy.ndarray
            The buffer as a numpy array. If normalization is enabled, the values are scaled between 0 and 1.
        """
        buffer_array = np.array(self.buffer)
        if self.normalize:
            min_val = np.min(buffer_array)
            max_val = np.max(buffer_array)
            if min_val != max_val:
                buffer_array = (buffer_array - min_val) / (max_val - min_val)
            else:
                buffer_array = np.full_like(buffer_array, 0.5)
        return buffer_array

    def get_last_value(self):
        """
        Returns the last value in the buffer or the default return value if the buffer is empty.

        Returns
        -------
        float
            The last value in the buffer or the default return value if the buffer is empty.
        """
        return self.buffer[-1] if len(self.buffer) > 0 else self.default_return_value

    def set_buffer_size(self, buffer_size):
        """
        Sets a new buffer size and adjusts the buffer accordingly.

        Parameters
        ----------
        buffer_size : int
            The new maximum size of the buffer.
        """
        self.buffer_size = buffer_size
        self.buffer = deque(self.buffer, maxlen=buffer_size)

    def set_normalize(self, normalize):
        """
        Sets the normalization flag.

        Parameters
        ----------
        normalize : bool
            A flag indicating whether to normalize the buffer values.
        """
        self.normalize = normalize


def scale_variable(variable, min_input, max_input, min_output, max_output):
    """
    Scales the input variable from the input range [min_input, max_input] to the output range [min_output, max_output].

    Parameters:
    variable (float): The input variable to be scaled.
    min_input (float): The minimum value of the input range.
    max_input (float): The maximum value of the input range.
    min_output (float): The minimum value of the output range.
    max_output (float): The maximum value of the output range.

    Returns:
    float: The scaled variable.
    """
    # Clip the variable between min_input and max_input using np.clip
    variable = np.clip(variable, min_input, max_input)
    # Scale the variable between min_output and max_output
    scaled_variable = min_output + (variable - min_input) * (max_output - min_output) / (max_input - min_input)
    return scaled_variable


class NumpyArrayBuffer:
    def __init__(self, buffer_size=500, default_return_value=None):
        """
        Initializes the NumpyArrayBuffer with a specified buffer size.

        Parameters
        ----------
        buffer_size : int
            The maximum size of the buffer.
        default_return_value : numpy array, optional
            The default return value if the buffer is empty.
        """
        self.buffer_size = buffer_size
        self.default_return_value = default_return_value
        self.buffer = deque(maxlen=buffer_size)
        self.array_shape = None

    def append(self, array):
        """
        Appends a new numpy array to the buffer. Sets the array shape if not already set.

        Parameters
        ----------
        array : numpy array
            The numpy array to be appended to the buffer.
        """
        if self.array_shape is None:
            self.array_shape = array.shape
        if array.shape == self.array_shape:
            self.buffer.append(array)
        else:
            raise ValueError(f"Array shape {array.shape} does not match buffer shape {self.array_shape}")

    def get_last(self):
        """
        Returns the last numpy array in the buffer or the default return value if the buffer is empty.

        Returns
        -------
        numpy array
            The last numpy array in the buffer or the default return value if the buffer is empty.
        """
        return self.buffer[-1] if len(self.buffer) > 0 else self.default_return_value

    def set_buffer_size(self, buffer_size):
        """
        Sets a new buffer size and adjusts the buffer accordingly.

        Parameters
        ----------
        buffer_size : int
            The new maximum size of the buffer.
        """
        self.buffer_size = buffer_size
        self.buffer = deque(self.buffer, maxlen=buffer_size)


def interpolate_linear(p0, p1, fract_mixing):
    r"""
    Helper function to mix two variables using standard linear interpolation.
    Args:
        p0:
            First tensor / np.ndarray for interpolation
        p1:
            Second tensor / np.ndarray  for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a linear mix between both.
    """
    reconvert_uint8 = False
    if type(p0) is np.ndarray and p0.dtype == 'uint8':
        reconvert_uint8 = True
        p0 = p0.astype(np.float64)

    if type(p1) is np.ndarray and p1.dtype == 'uint8':
        reconvert_uint8 = True
        p1 = p1.astype(np.float64)

    interp = (1 - fract_mixing) * p0 + fract_mixing * p1

    if reconvert_uint8:
        interp = np.clip(interp, 0, 255).astype(np.uint8)

    return interp

def get_time_ms():
    return int(round(time.time() * 1000))

class MultiThreader():
    def __init__(self, runfunc, sleeptime=0.5, idx=None):
        """
        Helper class to make threading easier
        runfunc is the function you want to run in the thread
        sleeptime is the delay between executions
        note to self: if more arguments shall be passed, please refactor and use that args= things.
        """
        
        self.t_refreshlast_slow = 0
        self.running = True
        self.runfunc = runfunc
        self.sleeptime = sleeptime
        self.thread = Thread(target = self.thread_loop, args = (1, ))
        self.thread.daemon = True
        self.thread.start()


    def thread_loop(self, arg):
        refreshrate_slow = 10
        
        while self.running:
            t_now = get_time_ms()

            t_diff_slow = t_now - self.t_refreshlast_slow
            if t_diff_slow > refreshrate_slow:
                self.t_refreshlast_slow = t_now
                self.runfunc()

            time.sleep(self.sleeptime) 
    
    def stop(self):
        self.running = False
        self.thread.join()

if __name__ == "__main__":
    # Example usage
    # save_api_key_to_lunar_config('ELEVEN_API_KEY', "bba")
    # # Reading a specific key
    
    api_key_value = read_api_key('ELEVEN_API_KEY')
    delete_api_key_from_lunar_config('ELEVEN_API_KEY')

