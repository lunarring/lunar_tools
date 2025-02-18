import torch
import numpy as np
from lunar_tools.torch_utils import resize

def test_resize_tensor_channel_first():
    # Test tensor with shape [3, 100, 120] (channel-first)
    t1 = torch.rand(3, 100, 120)
    output = resize(t1, resizing_factor=1)
    # Expect output shape to be [100, 120, 3]
    assert output.shape == (100, 120, 3), f"Expected (100, 120, 3), got {output.shape}"

def test_resize_tensor_channel_last():
    # Test tensor with shape [150, 200, 3] (channel-last)
    t2 = torch.rand(150, 200, 3)
    output = resize(t2, resizing_factor=1)
    # Expect output shape to remain [150, 200, 3]
    assert output.shape == (150, 200, 3), f"Expected (150, 200, 3), got {output.shape}"
