#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

def get_binary_kernel2d(
    window_size: tuple[int, int] | int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a binary kernel to extract the patches.
    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    if isinstance(window_size, int):
        ky = kx = window_size
    else:
        ky, kx = window_size
        
    window_range = kx * ky
    kernel = torch.zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)

class GaussianBlur(nn.Module):
    r"""Blurs an image using a Gaussian filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.
        sigma (float): standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> tx = torch.rand(2, 4, 5, 7)
        >>> blur = GaussianBlur((3, 3), 1.0)
        >>> output = blur(tx)
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    def __init__(self, kernel_size: Tuple[int, int], sigma: float) -> None:
        super(GaussianBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: float = sigma
        self.padding: Tuple[int, int] = self._compute_zero_padding(kernel_size)
        self.kernel: torch.Tensor = self._get_gaussian_kernel2d(kernel_size, sigma)

    @staticmethod
    def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        r"""Utility function that computes zero padding tuple."""
        computed: List[int] = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    @staticmethod
    def _get_gaussian_kernel2d(kernel_size: Tuple[int, int], sigma: float) -> torch.Tensor:
        """Create a Gaussian kernel for 2D convolution."""
        kx = torch.linspace(-kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[0])
        ky = torch.linspace(-kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[1])
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        kernel = torch.exp(-(kx**2 + ky**2) / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.shape)
        return kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))

        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))

        b, c, h, w = input.shape

        # Apply Gaussian blur
        features: torch.Tensor = F.conv2d(
            input.reshape(b * c, 1, h, w), self.kernel.to(input), padding=self.padding, stride=1)
        features = features.view(b, c, h, w)  # BxCxHxW

        return features    
    
class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = MedianBlur((3, 3))
        >>> output = blur(input)
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(MedianBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.padding: Tuple[int, int] = self._compute_zero_padding(kernel_size)
        self.kernel: torch.Tensor = get_binary_kernel2d(kernel_size)

    @staticmethod
    def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        r"""Utility function that computes zero padding tuple."""
        computed: List[int] = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))

        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))

        b, c, h, w = input.shape

        # map the local window to single vector
        features: torch.Tensor = F.conv2d(
            input.reshape(b * c, 1, h, w), self.kernel.to(input), padding=self.padding, stride=1)
        features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        median: torch.Tensor = torch.median(features, dim=2)[0]

        return median
    
    
def resize(input_img, resizing_factor=None, size=None, resample_method='bicubic', force_device=None):
    """
    This function is used to convert a numpy image array, a PIL image, or a tensor to a tensor and resize it by a given downscaling factor or to a given size.

    Args:
        input_img (np.ndarray, PIL.Image, or torch.Tensor): The input image to be converted and resized.
        size (tuple, optional): The desired size (height, width) for the output tensor. If provided, this overrides the downscaling_factor.
        resizing_factor (float, optional): The factor by which to downscale the input tensor. Defaults to None. Ignored if size is provided.
        resample_method (str, optional): The resampling method used for resizing. Defaults to 'bilinear'. Options: 'bilinear', 'nearest', 'bicubic', 'lanczos'.
        force_device (str, optional): The device to which the tensor should be moved. If not provided, the device of the input_img is used.

    Returns:
        np.ndarray, PIL.Image, or torch.Tensor: The converted and resized image.
    """
    supported_resample_methods = ['bilinear', 'nearest', 'bicubic', 'lanczos']
    
    if resample_method not in supported_resample_methods:
        raise ValueError(f"Unsupported resample method: {resample_method}. Choose from 'bilinear', 'nearest', 'bicubic', 'lanczos'.")
    
    input_type = type(input_img)
    input_dtype = None
    
    if force_device is not None:
        device = force_device
    elif input_type == torch.Tensor:
        device = input_img.device
    else:
        device = 'cpu'
    
    if input_type == np.ndarray:
        input_dtype = input_img.dtype
        input_img = input_img.copy()
        if len(input_img.shape) == 3:
            input_tensor = torch.as_tensor(input_img, dtype=torch.float, device=device).permute(2, 0, 1)
        elif len(input_img.shape) == 2:
            input_tensor = torch.as_tensor(input_img, dtype=torch.float, device=device).unsqueeze(0)
        else:
            raise ValueError("input_type np.ndarray should be 2 or 3 dimensional!")
    elif input_type == Image.Image:
        input_tensor = torch.as_tensor(np.array(input_img), dtype=torch.float, device=device).permute(2, 0, 1)
    elif input_type == torch.Tensor:
        input_dtype = input_img.dtype
        input_tensor = input_img.clone().to(dtype=torch.float, device=device)
    else:
        raise TypeError("input_img should be of type np.ndarray, PIL.Image, or torch.Tensor")
    
    if (size is not None) and (resizing_factor is not None):
        raise ValueError("Provide either size or downscaling_factor, not both.")
    elif (size is None) and (resizing_factor is None):
        raise ValueError("Either size or downscaling_factor must be provided.")
    elif size is not None:
        size = size
    else:
        if len(input_tensor.shape) == 3:
            size = (int(input_tensor.shape[1] * resizing_factor), int(input_tensor.shape[2] * resizing_factor))
        elif len(input_tensor.shape) == 4:
            size = (int(input_tensor.shape[2] * resizing_factor), int(input_tensor.shape[3] * resizing_factor))
    
    resized_tensor = torch.nn.functional.interpolate(input_tensor.unsqueeze(0), size=size, mode=resample_method).squeeze(0)
    
    if input_type == np.ndarray:
        if len(input_img.shape) == 3:
            resized_tensor = resized_tensor.permute(1,2,0).cpu()
        elif len(input_img.shape) == 2:
            resized_tensor = resized_tensor.squeeze(0).cpu()
        resized_array = np.clip(np.round(resized_tensor.numpy()), 0, 255).astype(input_dtype)
        return resized_array
    elif input_type == Image.Image:
        resized_tensor = resized_tensor.permute(1,2,0).cpu()
        resized_array = np.clip(np.round(resized_tensor.numpy()), 0, 255).astype('uint8')
        return Image.fromarray(resized_array, 'RGB')
    else:
        return resized_tensor.to(input_dtype)
    
if __name__ == "__main__":
    torch.manual_seed(0)
    tx = 255*torch.rand(1, 3, 100, 200)
    
    blur = GaussianBlur((3, 3), 3)
    output = blur(tx)

