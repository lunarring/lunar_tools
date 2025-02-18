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
