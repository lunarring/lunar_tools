import os
import platform

def get_os_type():
    os_name = platform.system()
    if os_name == "Darwin":
        return "MacOS"
    elif os_name == "Linux":
        dist_name, _, _ = platform.linux_distribution()
        if dist_name.lower() in ["ubuntu"]:
            return "Ubuntu"
        else:
            raise ValueError("unsupported OS")
    elif os_name == "Windows":
        return "Windows"
    else:
        raise ValueError("unsupported OS")

def get_config_path():
    os_type = get_os_type()
    if os_type in ["MacOS", "Ubuntu"]:
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


if __name__ == "__main__":
    # Example usage
    save_api_key_to_lunar_config('OPENAI_API_KEY', "bba")
    # # Reading a specific key
    api_key_value = read_api_key('OPENAI_API_KEY')
