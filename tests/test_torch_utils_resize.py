import torch
import numpy as np
import pytest
from lunar_tools.torch_utils import resize

def test_resize_channel_first():
    # Test with a torch tensor in channel-first format (3, H, W)
    input_tensor = torch.rand(3, 32, 32)
    output = resize(input_tensor, size=(32, 32), resample_method='nearest')
    # Verify that output is in channel-last format: (H, W, 3)
    assert output.shape == (32, 32, 3), f"Expected shape (32,32,3), got {output.shape}"

def test_resize_channel_last():
    # Test with a torch tensor already in channel-last format (H, W, 3)
    input_tensor = torch.rand(32, 32, 3)
    output = resize(input_tensor, size=(32, 32), resample_method='nearest')
    # For nearest interpolation with same size, the data should remain (nearly) unchanged.
    np.testing.assert_allclose(output.numpy(), input_tensor.numpy(), err_msg="Channel-last tensor data changed on resize")
    assert output.shape == (32, 32, 3), f"Expected shape (32,32,3), got {output.shape}"

def test_resize_empty_tensor():
    # Test with an empty tensor (0, H, W)
    input_tensor = torch.rand(0, 32, 32)
    output = resize(input_tensor, size=(0, 32), resample_method='nearest')
    assert output.shape == input_tensor.shape, f"Expected shape {input_tensor.shape}, got {output.shape}"
