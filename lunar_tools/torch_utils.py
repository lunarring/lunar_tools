#!/usr/bin/env python3
# -*- coding: utf-8 -*-
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

def _check_torch():
    """Helper to raise informative error when torch is not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch support requires the 'torch' package. Install it via 'pip install torch' "
            "or 'pip install lunar-tools[torch]'."
        )

_RESIZE_RESAMPLE_METHODS = frozenset({"bilinear", "nearest", "bicubic", "lanczos"})

def get_binary_kernel2d(
    window_size: tuple[int, int] | int, *, device: Optional['torch.device'] = None, dtype: 'torch.dtype' = None
) -> 'torch.Tensor':
    """Create a binary kernel to extract the patches.
    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    _check_torch()
    if dtype is None:
        dtype = torch.float32
    if isinstance(window_size, int):
        ky = kx = window_size
    else:
        ky, kx = window_size
        
    window_range = kx * ky
    kernel = torch.zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def interpolate_spherical(p0: 'torch.Tensor', p1: 'torch.Tensor', fract_mixing: float) -> 'torch.Tensor':
    """
    Performs spherical interpolation between two tensors.

    This function performs a spherical linear interpolation (slerp) between two tensors `p0` and `p1`.
    The interpolation is performed in a high-dimensional space, and the result is a tensor that lies
    on the shortest arc between `p0` and `p1` on the unit hypersphere.

    Args:
        p0 (torch.Tensor): The first tensor to interpolate from.
        p1 (torch.Tensor): The second tensor to interpolate to.
        fract_mixing (float): The fraction for the interpolation. A value of 0 returns `p0`, a value of 1 returns `p1`.

    Returns:
        torch.Tensor: The interpolated tensor.

    Note:
        The tensors `p0` and `p1` must have the same shape, and `fract_mixing` must be a scalar.
    """
    _check_torch()

    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    fract_mixing = np.clip(fract_mixing, 0, 1)
    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1 + epsilon, 1 - epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0 * s0 + p1 * s1

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()

    return interp

def interpolate_linear(p0: 'torch.Tensor', p1: 'torch.Tensor', fract_mixing: float) -> 'torch.Tensor':
    """
    Performs linear interpolation between two tensors.

    This function performs a linear interpolation (lerp) between two tensors `p0` and `p1`.
    The interpolation is performed in a high-dimensional space, and the result is a tensor that lies
    on the line between `p0` and `p1`.

    Args:
        p0 (torch.Tensor): The first tensor to interpolate from.
        p1 (torch.Tensor): The second tensor to interpolate to.
        fract_mixing (float): The fraction for the interpolation. A value of 0 returns `p0`, a value of 1 returns `p1`.

    Returns:
        torch.Tensor: The interpolated tensor.

    Note:
        The tensors `p0` and `p1` must have the same shape, and `fract_mixing` must be a scalar.
    """
    _check_torch()

    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    fract_mixing = np.clip(fract_mixing, 0, 1)
    p0 = p0.double()
    p1 = p1.double()

    interp = p0 * (1 - fract_mixing) + p1 * fract_mixing

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()

    return interp


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
        _check_torch()
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
        _check_torch()
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
    _check_torch()
    if resample_method not in _RESIZE_RESAMPLE_METHODS:
        raise ValueError(f"Unsupported resample method: {resample_method}. Choose from 'bilinear', 'nearest', 'bicubic', 'lanczos'.")
    
    is_numpy = isinstance(input_img, np.ndarray)
    is_pil = isinstance(input_img, Image.Image)
    is_tensor = isinstance(input_img, torch.Tensor)
    input_dtype = None
    size_checked = False
    
    if force_device is not None:
        device = torch.device(force_device)
    elif is_tensor:
        device = input_img.device
    else:
        device = 'cpu'
    
    if is_numpy:
        input_dtype = input_img.dtype
        if len(input_img.shape) not in (2, 3):
            raise ValueError("input_type np.ndarray should be 2 or 3 dimensional!")
        if (size is not None) and (resizing_factor is not None):
            raise ValueError("Provide either size or downscaling_factor, not both.")
        elif (size is None) and (resizing_factor is None):
            raise ValueError("Either size or downscaling_factor must be provided.")
        if size is None:
            size = (int(input_img.shape[0] * resizing_factor), int(input_img.shape[1] * resizing_factor))
        size_checked = True
        if size == (input_img.shape[0], input_img.shape[1]):
            resized_array = np.clip(np.round(input_img), 0, 255).astype(input_dtype)
            return resized_array
        if len(input_img.shape) == 3:
            input_tensor = torch.as_tensor(input_img, dtype=torch.float, device=device).permute(2, 0, 1)
        else:
            input_tensor = torch.as_tensor(input_img, dtype=torch.float, device=device).unsqueeze(0)
    elif is_pil:
        input_tensor = torch.as_tensor(np.array(input_img), dtype=torch.float, device=device).permute(2, 0, 1)
    elif is_tensor:
        input_dtype = input_img.dtype
        input_tensor = input_img
        if input_tensor.device != device or input_tensor.dtype != torch.float32:
            input_tensor = input_tensor.to(dtype=torch.float, device=device)
        is_channels_last = False
        if input_tensor.dim() == 3 and input_tensor.shape[2] <= 3:
            is_channels_last = True
            input_tensor = input_tensor.permute(2, 0, 1)
    else:
        raise TypeError("input_img should be of type np.ndarray, PIL.Image, or torch.Tensor")
    
    if not size_checked:
        if (size is not None) and (resizing_factor is not None):
            raise ValueError("Provide either size or downscaling_factor, not both.")
        elif (size is None) and (resizing_factor is None):
            raise ValueError("Either size or downscaling_factor must be provided.")
        if input_tensor.dim() == 4:
            raise ValueError("resize() does not support batched images; expected a 3-dimensional tensor.")
        if size is None:
            size = (int(input_tensor.shape[1] * resizing_factor), int(input_tensor.shape[2] * resizing_factor))
    
    resized_tensor = F.interpolate(input_tensor.unsqueeze(0), size=size, mode=resample_method).squeeze(0)
    
    if is_numpy:
        if len(input_img.shape) == 3:
            resized_tensor = resized_tensor.permute(1,2,0).cpu()
        elif len(input_img.shape) == 2:
            resized_tensor = resized_tensor.squeeze(0).cpu()
        resized_array = resized_tensor.numpy()
        np.rint(resized_array, out=resized_array)
        np.clip(resized_array, 0, 255, out=resized_array)
        return resized_array.astype(input_dtype, copy=False)
    elif is_pil:
        resized_tensor = resized_tensor.permute(1,2,0).cpu()
        resized_array = resized_tensor.numpy()
        np.rint(resized_array, out=resized_array)
        np.clip(resized_array, 0, 255, out=resized_array)
        resized_array = resized_array.astype('uint8', copy=False)
        return Image.fromarray(resized_array, 'RGB')
    else:
        if 'is_channels_last' in locals() and is_channels_last:
            resized_tensor = resized_tensor.permute(1,2,0)
        if resized_tensor.dtype != input_dtype:
            resized_tensor = resized_tensor.to(input_dtype)
        return resized_tensor
    
class FrequencyFilter:
    """
    FrequencyFilter is a class that applies a high pass filter to an image.

    This class creates a high pass filter of a given size and radius, and applies it to an image.
    The high pass filter is created in the frequency domain, and is applied to the image by performing
    a 2D Fast Fourier Transform (FFT) on the image, multiplying the result by the filter, and then
    performing an inverse FFT to obtain the filtered image. The class can also do a low-pass filter.

    Attributes:
        size (tuple): The size of the high pass filter. This should match the size of the images that
            will be filtered.
        radius (float): The radius of the high pass filter. This determines the cutoff frequency of the
            filter.
        device (str): The device on which to perform the computations. Default is 'cpu'.
        high_pass_filter (torch.Tensor): The high pass filter created based on the size and radius.

    """
    def __init__(self, size, radius, device='cpu'):
        _check_torch()
        self.size = size
        self.radius = radius
        self.device = device
        self._init_high_pass_filter()

    def _init_high_pass_filter(self):
        """
        Initializes the high pass filter.

        This method creates a high pass filter based on the size and radius provided during the 
        initialization of the FrequencyFilter class. The filter is created in the frequency domain 
        and is stored as a tensor attribute of the class instance.
        """
        rows, cols = self.size
        crow, ccol = rows // 2, cols // 2
        high_pass_filter = torch.ones((rows, cols), dtype=torch.float32, device=self.device)
        y, x = torch.meshgrid(torch.arange(rows, device=self.device), torch.arange(cols, device=self.device), indexing='ij')
        mask_area = (x - ccol)**2 + (y - crow)**2 <= self.radius**2
        high_pass_filter[mask_area] = 0
        self.high_pass_filter = high_pass_filter
        self.low_pass_filter = 1 - high_pass_filter
        self.high_pass_filter_unshifted = torch.fft.ifftshift(high_pass_filter)
        self.low_pass_filter_unshifted = torch.fft.ifftshift(self.low_pass_filter)


    def apply_highpass(self, image):
        """
        Applies the high pass filter to the input image.

        This method applies the high pass filter to the input image in the frequency domain.
        It performs a 2D Fast Fourier Transform (FFT) on the image, multiplies the result by the filter,
        and then performs an inverse FFT to obtain the filtered image.

        Args:
            image (torch.Tensor): The input image to be filtered. The image should be a 4D tensor with
                shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The filtered image. The image is a 4D tensor with shape (batch_size, channels, height, width).
        """
        filter_torch = self.high_pass_filter_unshifted
        return self._apply_filter(image, filter_torch)

    def apply_lowpass(self, image):
        """
        Applies the low pass filter to the input image.

        This method applies the low pass filter to the input image in the frequency domain.
        It performs a 2D Fast Fourier Transform (FFT) on the image, multiplies the result by the filter,
        and then performs an inverse FFT to obtain the filtered image.

        Args:
            image (torch.Tensor): The input image to be filtered. The image should be a 4D tensor with
                shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The filtered image. The image is a 4D tensor with shape (batch_size, channels, height, width).
        """
        filter_torch = self.low_pass_filter_unshifted
        return self._apply_filter(image, filter_torch)

    def _apply_filter(self, image, filter_torch):
        # Preserve legacy behavior: apply filter to the first batch only.
        image0 = image[0]
        fft_image = torch.fft.fft2(image0, dim=(-2, -1))
        filtered_fft = fft_image * filter_torch
        filtered_image = torch.fft.ifft2(filtered_fft, dim=(-2, -1))
        return torch.real(filtered_image).unsqueeze(0)

# Example for FrequencyFilter
if __name__ == "__main__":
    torch.manual_seed(0)
    tx = 255*torch.rand(1, 3, 100, 200)
    
    high_pass_filter = FrequencyFilter((100, 200), 30)
    output0 = high_pass_filter.apply_lowpass(tx)
    output1 = high_pass_filter.apply_highpass(tx)


# Example for Gaussianblur
if __name__ == "__main__X":
    torch.manual_seed(0)
    tx = 255*torch.rand(1, 3, 100, 200)
    
    blur = GaussianBlur((3, 3), 3)
    output = blur(tx)
