#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform
import os 

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
    else:
        raise ValueError("unsupported OS")