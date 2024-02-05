import os
import platform
import numpy as np

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
    # First, try to get the API key from the environment variables
    api_key = os.getenv(key_name)

    if not api_key:
        # If not found, try to get the key from lunar config
        api_key = read_api_key_from_lunar_config(key_name)

    if not api_key:
        # If still not found, prompt the user to input the key
        print(f"API key for {key_name} not found. Please paste your API key:")
        api_key = input().strip()
        save_api_key_to_lunar_config(key_name, api_key)

    return api_key


def delete_api_key_from_lunar_config(key_name):
    keys = read_all_api_keys_from_lunar_config()
    if key_name in keys:
        del keys[key_name]

        config_path = get_config_path()

        with open(config_path, 'w') as file:
            for k, v in keys.items():
                file.write(f"{k}={v}\n")

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

if __name__ == "__main__":
    # Example usage
    # save_api_key_to_lunar_config('ELEVEN_API_KEY', "bba")
    # # Reading a specific key
    api_key_value = read_api_key('ELEVEN_API_KEY')
    delete_api_key_from_lunar_config('ELEVEN_API_KEY')
