from PIL import Image
import requests
from io import BytesIO
import copy
import base64
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import Optional


try:
    import decord
    decord.bridge.set_bridge("torch")
except ImportError:
    decord = None


MIMO_PLACEHOLDER = "<|mimo_placeholder|>"

# Imagenet's mean and std.
QWEN2VL_PIXEL_MEAN = [123.675, 116.28, 103.53]
QWEN2VL_PIXEL_STD = [58.395, 57.12, 57.375]

# Reshape for broadcasting.
QWEN2VL_PIXEL_MEAN = torch.Tensor(QWEN2VL_PIXEL_MEAN).view(-1, 1, 1)
QWEN2VL_PIXEL_STD = torch.Tensor(QWEN2VL_PIXEL_STD).view(-1, 1, 1)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
])

# Device-aware cache for mean/std tensors
_mean_std_cache = {}


def format_timestamp(timestamp: float):
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    return f"{minutes:02d}:{seconds:02d}"


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if min(height, width) < factor:
        # Keep aspect ratio and resize smaller edge to factor
        if height < width:
            height = factor
            width = int(width * (factor / height))
        else:
            width = factor 
            height = int(height * (factor / width))
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def standardize_image(img):
    """Standardize image pixel values."""
    return (torch.Tensor(np.array(img)).permute(2, 0, 1) - QWEN2VL_PIXEL_MEAN) / QWEN2VL_PIXEL_STD
import torch.nn.functional as F


def standardize_batch(images: torch.Tensor) -> torch.Tensor:
    """
    Standardize a batch of images using device-aware mean/std.
    
    Args:
        images: Tensor of shape (B, C, H, W) in range [0, 255]
    
    Returns:
        Standardized tensor of shape (B, C, H, W)
    """
    device_key = str(images.device)
    if device_key not in _mean_std_cache:
        _mean_std_cache[device_key] = (
            torch.tensor(QWEN2VL_PIXEL_MEAN, device=images.device).view(1, -1, 1, 1),
            torch.tensor(QWEN2VL_PIXEL_STD, device=images.device).view(1, -1, 1, 1),
        )
    mean, std = _mean_std_cache[device_key]
    return (images - mean) / std

def get_visual_transform_batch(
    frames: torch.Tensor,  # (t, c, h, w)
    factor: int,
    min_pixels: int,
    max_pixels: int,
    device: Optional[torch.device] = None,
):
    """
    Batch version of get_visual_transform.
    
    Note: 
    - Input frames should be in range [0, 255] (uint8 or float)
    - standardize_image expects PIL image (H,W,C) in [0,255], converts to numpy array,
      then to tensor, then (img - mean) / std (WITHOUT dividing by 255 first!)
    - So we need to match that: resize, then (img - mean) / std directly
    """
    if device is not None:
        frames = frames.to(device)
    
    t, c, h, w = frames.shape
    
    # Compute target size ONCE
    h_bar, w_bar = smart_resize(h, w, factor, min_pixels, max_pixels)
    
    # Batch resize — no loop, no PIL
    # Convert to float for interpolation
    resized = F.interpolate(
        frames.float(),
        size=(h_bar, w_bar),
        mode='bilinear',
        align_corners=False,
    )
    
    # Batch standardization using device-aware mean/std
    standardized = standardize_batch(resized)
    
    return standardized, w_bar, h_bar

def get_visual_transform(
        img: torch.Tensor | Image.Image, 
        factor: int, 
        min_pixels: int, 
        max_pixels: int,
        device: Optional[torch.device] = None,
    ):
    """
    Transform and resize image using PyTorch's F.interpolate with bilinear mode.
    This ensures consistency with get_visual_transform_batch.
    """
    # Convert to torch tensor if needed
    if isinstance(img, torch.Tensor):
        # Input: (C, H, W) in range [0, 255]
        img_tensor = img.float()
        c, h, w = img_tensor.shape
    elif isinstance(img, Image.Image):
        # PIL Image
        img = img.convert("RGB")
        w, h = img.size
        # Convert PIL to tensor: (H, W, C) -> (C, H, W)
        img_array = np.array(img)  # (H, W, C), [0, 255]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # (C, H, W)
        c = 3
    else:
        raise TypeError(f"Unsupported image type: {type(img)}. Expected torch.Tensor or PIL.Image.Image")
    
    if device is not None:
        img_tensor = img_tensor.to(device)
    
    # Compute target size
    h_bar, w_bar = smart_resize(h, w, factor, min_pixels, max_pixels)
    
    # Resize using F.interpolate with bilinear (same as batch version)
    img_resized = F.interpolate(
        img_tensor.unsqueeze(0),  # Add batch dim
        size=(h_bar, w_bar),
        mode='bilinear',
        align_corners=False,
    )
    
    # Standardize: (img - mean) / std
    img_standardized = standardize_batch(img_resized).squeeze(0)  # (C, H, W)
    
    return img_standardized, w_bar, h_bar


def fetch_image(
    image: Image.Image | str | bytes,
):
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # fix memory leak issue while using BytesIO
            with requests.get(image, stream=True) as response:
                response.raise_for_status()
                with BytesIO(response.content) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                # fix memory leak issue while using BytesIO
                with BytesIO(data) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        else:
            image_obj = Image.open(image)
    else:
        # bytes
        image_obj = Image.open(BytesIO(image))
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    return image
