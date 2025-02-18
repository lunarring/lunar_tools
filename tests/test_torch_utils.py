import torch
import pytest
from lunar_tools.torch_utils import resize

def test_resize_single_image():
    # Create a 3-dimensional tensor (C, H, W)
    input_tensor = torch.rand(3, 100, 200)
    output = resize(input_tensor, resizing_factor=2)
    assert output.shape == (3, 200, 400), f"Expected shape (3, 200, 400) but got {output.shape}"

def test_resize_batch_image_error():
    # Create a 4-dimensional tensor (N, C, H, W)
    input_tensor = torch.rand(1, 3, 100, 200)
    with pytest.raises(ValueError, match="resize\\(\\) does not support batched images; expected a 3-dimensional tensor."):
        resize(input_tensor, resizing_factor=2)

def test_resize_channels_last():
    # Create a channels-last tensor (H, W, C)
    img = torch.rand(100, 150, 3)
    output = resize(img, size=(200, 300))
    assert output.shape == (200, 300, 3), f"Expected shape (200, 300, 3) but got {output.shape}"
def test_resize_channels_first():
    # Create a channels-first tensor (C, H, W)
    img = torch.rand(3, 100, 150)
    output = resize(img, size=(200, 300))
    assert output.shape == (3, 200, 300), f"Expected shape (3, 200, 300) but got {output.shape}"

def test_resize_tensor_channels_last_small():
    # Test a channels-last tensor with 1 channel (H, W, C)
    input_tensor = torch.rand(100, 200, 1)
    output = resize(input_tensor, resizing_factor=2)
    assert output.shape == (200, 400, 1), f"Expected shape (200, 400, 1) but got {output.shape}"

    # Test a channels-last tensor with 2 channels (H, W, C)
    input_tensor = torch.rand(100, 200, 2)
    output = resize(input_tensor, resizing_factor=2)
    assert output.shape == (200, 400, 2), f"Expected shape (200, 400, 2) but got {output.shape}"

def test_resize_tensor_channels_first_large():
    # For a tensor that is already channels-first with more than 3 channels, no swap should occur.
    input_tensor = torch.rand(4, 100, 200)
    output = resize(input_tensor, resizing_factor=2)
    assert output.shape == (4, 200, 400), f"Expected shape (4, 200, 400) but got {output.shape}"
