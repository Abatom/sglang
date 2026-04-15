"""MiMoV2 Omni multimodal processor -- protocol, utilities, and processor."""

import asyncio
import base64
import copy
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pybase64
import requests
import torch
import torch.nn.functional as F
from fastapi import HTTPException
from PIL import Image
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.mimo_v2_omni import MiMoV2OmniForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import smart_nframes
from sglang.srt.utils import ImageData, VideoData
from sglang.utils import logger
from torchcodec.decoders import AudioDecoder
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)

try:
    import decord
    from decord import VideoReader

    decord.bridge.set_bridge("torch")
except ImportError:
    decord = None
    VideoReader = None

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram
except ImportError:
    print("[Warning] torchaudio is not installed, audio inference will not be supported")
    torchaudio = None
    MelSpectrogram = None

# ---------------------------------------------------------------------------
# Protocol data classes
# ---------------------------------------------------------------------------

@dataclass
class ImageInput:
    image: Image.Image | str | bytes | torch.Tensor     # pixels (C, H, W)
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.image, (Image.Image, str, bytes, torch.Tensor)):
            raise ValueError(f"image must be a PIL.Image.Image, str, bytes, or torch.Tensor, but got {type(self.image)}")

@dataclass
class VideoInput:
    video: VideoReader | str | bytes | tuple[torch.Tensor, torch.Tensor]           # pixels (T, C, H, W), timestamps (T)
    # video preprocessor arguments
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    total_max_pixels: Optional[int] = None
    fps: Optional[float] = None
    num_frames: Optional[int] = None
    max_frames: Optional[int] = None
    min_frames: Optional[int] = None
    do_include_last_frame: Optional[bool] = False
    # segment arguments
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    segment_type: Literal["individual", "partial"] = "individual"
    

    def __post_init__(self):
        if not isinstance(self.video, (VideoReader, str, bytes, tuple)):
            raise ValueError(f"video must be a str, bytes, or tuple, but got {type(self.video)}")
        if isinstance(self.video, tuple):
            if len(self.video) != 2:
                raise ValueError(f"video must be a tuple of 2 elements (pixels, timestamps), but got {len(self.video)} elements")
            if not isinstance(self.video[0], torch.Tensor) or not isinstance(self.video[1], torch.Tensor):
                raise ValueError(f"video must be a tuple of Tensors (pixels, timestamps), but got {type(self.video[0])} and {type(self.video[1])}")
            if self.video[0].ndim != 4 or self.video[1].ndim != 1 or self.video[0].shape[0] != self.video[1].shape[0]:
                raise ValueError(f"video must be a tuple of (pixels-TCHW, timestamps-T), but got {self.video[0].shape} and {self.video[1].shape}")
        assert self.segment_type in ["individual", "partial"]
        assert self.segment_type == "partial" or (self.start_time is None and self.end_time is None)
            

@dataclass
class AudioInput:
    """
    if audio is str or bytes, only load it as mel spectrogram.
    if audio is tuple, it is (waveform, original_sr)
    if audio is torch.Tensor, it is tokenized input ids with shape (T, n_vq+).
    if audio is np.ndarray, it is a pre-loaded waveform (1D, already resampled).
    """
    audio: str | bytes | tuple | torch.Tensor | np.ndarray

    def __post_init__(self):
        if not isinstance(self.audio, (str, bytes, tuple, torch.Tensor, np.ndarray)):
            raise ValueError(f"audio must be a str, bytes, tuple, torch.Tensor, or np.ndarray, but got {type(self.audio)}")
        if isinstance(self.audio, tuple):
            if len(self.audio) != 2 or not isinstance(self.audio[0], torch.Tensor) or not isinstance(self.audio[1], (int, float)):
                raise ValueError(f"audio must be a tuple of (waveform-T, original_sr-int/float), but got {len(self.audio)} elements and {type(self.audio[0])} and {type(self.audio[1])}")
            if self.audio[0].ndim != 1:
                raise ValueError(f"waveform must be a 1D tensor, but got {self.audio[0].ndim}D tensor")
            if self.audio[1] <= 0:
                raise ValueError(f"original_sr must be a positive number, but got {self.audio[1]}")
        if isinstance(self.audio, torch.Tensor) and self.audio.ndim != 2:
            raise ValueError(f"audio must be a 2D tensor, but got {self.audio.ndim}D tensor")


@dataclass
class VideoAudioInput:
    video: VideoReader | str | bytes | tuple[torch.Tensor, torch.Tensor]           # pixels (T, C, H, W), timestamps (T)
    audio: str | bytes | torch.Tensor
    # video preprocessor arguments
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    total_max_pixels: Optional[int] = None
    fps: Optional[float] = None
    num_frames: Optional[int] = None
    max_frames: Optional[int] = None
    min_frames: Optional[int] = None
    do_include_last_frame: Optional[bool] = False
    # segment arguments
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    segment_type: Literal["individual", "partial"] = "individual"

    def __post_init__(self):
        if not isinstance(self.video, (VideoReader, str, bytes, tuple)):
            raise ValueError(f"video must be a str, bytes, or tuple, but got {type(self.video)}")
        if isinstance(self.video, tuple):
            if len(self.video) != 2:
                raise ValueError(f"video must be a tuple of 2 elements (pixels, timestamps), but got {len(self.video)} elements")
            if not isinstance(self.video[0], torch.Tensor) or not isinstance(self.video[1], torch.Tensor):
                raise ValueError(f"video must be a tuple of Tensors (pixels, timestamps), but got {type(self.video[0])} and {type(self.video[1])}")
            if self.video[0].ndim != 4 or self.video[1].ndim != 1 or self.video[0].shape[0] != self.video[1].shape[0]:
                raise ValueError(f"video must be a tuple of (pixels-TCHW, timestamps-T), but got {self.video[0].shape} and {self.video[1].shape}")
        assert self.segment_type in ["individual", "partial"]
        assert self.segment_type == "partial" or (self.start_time is None and self.end_time is None)

        if not isinstance(self.audio, (str, bytes, torch.Tensor)):
            raise ValueError(f"audio must be a str, bytes, or torch.Tensor, but got {type(self.audio)}")
        if isinstance(self.audio, torch.Tensor) and self.audio.ndim != 2:
            raise ValueError(f"audio must be a 2D tensor, but got {self.audio.ndim}D tensor")


TextInput = str | list[int]

@dataclass
class MiMoVLInputSample:
    input_ids: torch.Tensor
    labels: Optional[torch.Tensor]
    pixel_values: list[torch.Tensor]                # list of (num_patches, patch_dim)
    pixel_values_videos: list[torch.Tensor]         # list of (num_patches, patch_dim)
    image_thw_grids: list[torch.Tensor]             # list of (3)
    video_thw_grids: list[torch.Tensor]             # list of (3)
    audio_inputs: list[torch.Tensor]                # list of (T_padded//group_size, group_size, n_vq) for audio_input_ids; list of (n_mels, seq_len) for audio_spec
    position_ids: Optional[torch.Tensor] = None
    rope_deltas: Optional[torch.Tensor] = None
    extra: dict = field(default_factory=dict)

@dataclass
class Content:
    type: Literal["text", "image", "video", "audio", "video_audio"]
    content: TextInput | ImageInput | VideoInput | AudioInput | VideoAudioInput
    is_target: Optional[bool] = None

    def __post_init__(self):
        if self.type not in ["text", "image", "video", "audio", "video_audio"]:
            raise ValueError(f"type must be one of text, image, video, audio, video_audio, but got {self.type}")
        if self.type == "text":
            if not isinstance(self.content, (str, list)) or (isinstance(self.content, list) and not all(isinstance(item, int) for item in self.content)):
                raise ValueError(f"content must be a str or a list of ints, but got {type(self.content)}")
        elif self.type == "image":
            if not isinstance(self.content, ImageInput):
                raise ValueError(f"content must be a ImageInput, but got {type(self.content)}")
        elif self.type == "video":
            if not isinstance(self.content, VideoInput):
                raise ValueError(f"content must be a VideoInput, but got {type(self.content)}")
        elif self.type == "audio":
            if not isinstance(self.content, AudioInput):
                raise ValueError(f"content must be a AudioInput, but got {type(self.content)}")
        elif self.type == "video_audio":
            if not isinstance(self.content, VideoAudioInput):
                raise ValueError(f"content must be a VideoAudioInput, but got {type(self.content)}")

@dataclass
class MessageTurn:
    role: Literal["user", "assistant", "system"]
    contents: list[Content]

Conversation = list[MessageTurn]


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------

def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    model_type: str,
    tokens_per_second: Optional[int] = None,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                if model_type == "qwen2_5_vl":
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(
                        -1, llm_grid_h * llm_grid_w
                    )

                    time_tensor = (
                        expanded_range * second_per_grid_t * tokens_per_second
                    )

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()
                elif model_type == "qwen2_vl":
                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                else:
                    raise RuntimeError("Unimplemented")
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, :] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        s = input_ids.shape[1]
        position_ids = torch.arange(s)
        position_ids = (
            position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
        )
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(
            -1, keepdim=True
        )[0]
        mrope_position_deltas = max_position_ids + 1 - s
        return position_ids, mrope_position_deltas


# ---------------------------------------------------------------------------
# VL utilities
# ---------------------------------------------------------------------------

MIMO_PLACEHOLDER = "<|mimo_placeholder|>"

# Imagenet's mean and std, reshaped for broadcasting.
QWEN2VL_PIXEL_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
QWEN2VL_PIXEL_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

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


# ---------------------------------------------------------------------------
# MiMoVLProcessor
# ---------------------------------------------------------------------------

class MiMoVLProcessor:
    def __init__(
            self,
            tokenizer,
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
            temporal_compression_ratio=1,

            video_tokens_per_second=2,
            use_video_timestamps=False,
            video_audio_interleave_length=0,
            use_per_grid_t_timestamps=True,

            audio_kernel_size=3,
            audio_stride_size=2,
            audio_avg_pooler=2,

            audio_sampling_rate=24000,
            audio_nfft=960,
            audio_hop_length=240,
            audio_window_size=960,
            audio_fmin=0,
            audio_fmax=None,
            audio_n_mels=128,

            audio_segment_size=6000,

            audio_channels=8,
            audio_group_size=4,
            audio_input_id_per_second=25,
            audio_zeroemb_idx=4096,

            image_min_pixels=None,
            image_max_pixels=None,
            video_min_pixels=None,
            video_max_pixels=None,
            video_total_max_pixels=None,
            fps=None,
            num_frames=None,
            max_frames=None,
            min_frames=None,

            image_token_id=None,
            video_token_id=None,
            audio_token_id=None,
            vision_start_token_id=None,
            vision_end_token_id=None,
            audio_start_token_id=None,
            audio_end_token_id=None,
            video_start_token_id=None,
            video_end_token_id=None,
            pad_token_id=None,

            rope_type="rope",

            video_process_num_threads=16,
            
            device=None,

            **kwargs
        ):
        self.tokenizer = tokenizer
        self.video_process_num_threads = video_process_num_threads
        
        if device is None:
            self.device = None
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.rope_type = rope_type
        if self.rope_type == "1d":
            self.rope_type = "rope"
        assert self.rope_type in ["rope", "mrope"]

        self.use_video_timestamps = use_video_timestamps
        assert self.use_video_timestamps
        assert not self.use_video_timestamps or self.rope_type == "rope", "use_video_timestamps only supports 1d rope"
        self.video_audio_interleave_length = video_audio_interleave_length
        self.use_per_grid_t_timestamps = False #use_per_grid_t_timestamps
        assert self.video_audio_interleave_length == -1 or self.rope_type == "rope", "video_audio_interleave_length != -1 only supports 1d rope"
        assert self.video_audio_interleave_length == -1 or self.video_audio_interleave_length >= 0

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.pad_token_id = pad_token_id

        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.temporal_compression_ratio = temporal_compression_ratio

        self.video_tokens_per_second = video_tokens_per_second

        self.audio_sampling_rate = audio_sampling_rate
        self.audio_nfft = audio_nfft
        self.audio_hop_length = audio_hop_length
        self.audio_window_size = audio_window_size
        self.audio_fmin = audio_fmin
        self.audio_fmax = audio_fmax
        self.audio_n_mels = audio_n_mels

        self.audio_segment_size = audio_segment_size

        self.audio_kernel_size = audio_kernel_size
        self.audio_stride_size = audio_stride_size
        self.audio_avg_pooler = audio_avg_pooler

        # load mel spectrogram lazily, otherwise it might hang in mimo-omni-dataloader
        self.mel_spectrogram_kwargs = dict(
            sample_rate=audio_sampling_rate,
            n_fft=audio_nfft,
            hop_length=audio_hop_length,
            win_length=audio_window_size,
            f_min=audio_fmin,
            f_max=audio_fmax,
            n_mels=audio_n_mels,
            power=1.0,
            center=True,
        )
        self._mel_spectrogram = None
        self._resamplers = OrderedDict()
        self._resamplers_max = 16

        self.audio_channels = audio_channels
        self.audio_group_size = audio_group_size
        self.audio_input_id_per_second = audio_input_id_per_second
        if isinstance(audio_zeroemb_idx, int):
            self.audio_zeroemb_idxs = torch.tensor([audio_zeroemb_idx] * self.audio_channels, dtype=torch.int32)
        elif isinstance(audio_zeroemb_idx, list):
            if len(audio_zeroemb_idx) == 1:
                self.audio_zeroemb_idxs = torch.tensor(audio_zeroemb_idx * self.audio_channels, dtype=torch.int32)
            elif len(audio_zeroemb_idx) == self.audio_channels:
                self.audio_zeroemb_idxs = torch.tensor(audio_zeroemb_idx, dtype=torch.int32)
            else:
                raise ValueError(f"audio_zeroemb_idx must be a list of 1 or {self.audio_channels} integers, but got {len(audio_zeroemb_idx)}")
        else:
            raise ValueError(f"audio_zeroemb_idx must be an integer or a list of {self.audio_channels} integers, but got {type(audio_zeroemb_idx)}")

        assert image_min_pixels is not None
        assert image_max_pixels is not None
        assert video_min_pixels is not None
        assert video_max_pixels is not None
        assert video_total_max_pixels is not None
        assert fps is not None or num_frames is not None

        self.default_image_processor_kwargs = {
            "min_pixels": image_min_pixels,
            "max_pixels": image_max_pixels,
        }

        self.default_video_processor_kwargs = {
            "min_pixels": video_min_pixels,
            "max_pixels": video_max_pixels,
            "total_max_pixels": video_total_max_pixels,
            "fps": fps,
            "num_frames": num_frames,
            "max_frames": max_frames,
            "min_frames": min_frames,
        }

        self.http_session = requests.Session()
        for k, v in kwargs.items():
            logger.info(f"[Warning] Ignored unknown parameter {k} for MiMoVLProcessor")

    def get_config(self):
        return {
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "temporal_patch_size": self.temporal_patch_size,
            "temporal_compression_ratio": self.temporal_compression_ratio,
            "video_tokens_per_second": self.video_tokens_per_second,
            "use_video_timestamps": self.use_video_timestamps,
            "video_audio_interleave_length": self.video_audio_interleave_length,
            "use_per_grid_t_timestamps": self.use_per_grid_t_timestamps,
            "audio_kernel_size": self.audio_kernel_size,
            "audio_stride_size": self.audio_stride_size,
            "audio_avg_pooler": self.audio_avg_pooler,
            "audio_sampling_rate": self.audio_sampling_rate,
            "audio_nfft": self.audio_nfft,
            "audio_hop_length": self.audio_hop_length,
            "audio_window_size": self.audio_window_size,
            "audio_fmin": self.audio_fmin,
            "audio_fmax": self.audio_fmax,
            "audio_n_mels": self.audio_n_mels,
            "audio_segment_size": self.audio_segment_size,
            "audio_channels": self.audio_channels,
            "audio_group_size": self.audio_group_size,
            "audio_input_id_per_second": self.audio_input_id_per_second,
            "audio_zeroemb_idx": self.audio_zeroemb_idxs.tolist(),
            "image_min_pixels": self.default_image_processor_kwargs["min_pixels"],
            "image_max_pixels": self.default_image_processor_kwargs["max_pixels"],
            "video_min_pixels": self.default_video_processor_kwargs["min_pixels"],
            "video_max_pixels": self.default_video_processor_kwargs["max_pixels"],
            "video_total_max_pixels": self.default_video_processor_kwargs["total_max_pixels"],
            "fps": self.default_video_processor_kwargs["fps"],
            "num_frames": self.default_video_processor_kwargs["num_frames"],
            "max_frames": self.default_video_processor_kwargs["max_frames"],
            "min_frames": self.default_video_processor_kwargs["min_frames"],
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "audio_token_id": self.audio_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
            "audio_start_token_id": self.audio_start_token_id,
            "audio_end_token_id": self.audio_end_token_id,
            "video_start_token_id": self.video_start_token_id,
            "video_end_token_id": self.video_end_token_id,
            "pad_token_id": self.pad_token_id,
            "rope_type": self.rope_type,
            "video_process_num_threads": self.video_process_num_threads,
        }

    def copy_codebase(self, path: str):
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        dst_path = os.path.join(path, "mimo_vl_processor")
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)
                logger.info(f"Successfully copied mimo_vl_processor to {dst_path}")
            else:
                logger.info(f"mimo_vl_processor codebase already exists at {dst_path}, skip copying")
        else:
            logger.info(f"mimo_vl_processor codebase not found at {src_path}, skip copying")
            raise FileNotFoundError(f"mimo_vl_processor codebase not found at {src_path}")

    def save_config(self, path: str):
        config = self.get_config()
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load_from_config(cls, path: str | dict, **kwargs):
        if isinstance(path, str):
            if os.path.isdir(path):
                path = os.path.join(path, "mimo_processor_config.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Processor config file not found at {path}")
            with open(path, "r") as f:
                config = json.load(f)
        else:
            config = path
        config.update(kwargs)
        return cls(**config)

    @property
    def mel_spectrogram(self):
        if self._mel_spectrogram is None:
            self._mel_spectrogram = MelSpectrogram(**self.mel_spectrogram_kwargs)
        return self._mel_spectrogram

    def prepare_image_kwargs(self, image: ImageInput):
        kwargs = {}
        for k in ["min_pixels", "max_pixels"]:
            if getattr(image, k) is not None:
                kwargs[k] = getattr(image, k)
            else:
                kwargs[k] = self.default_image_processor_kwargs[k]
        return kwargs

    def prepare_video_kwargs(self, video: VideoInput | VideoAudioInput):
        kwargs = {}
        for k in ["min_pixels", "max_pixels", "total_max_pixels"]:
            if getattr(video, k) is not None:
                kwargs[k] = getattr(video, k)
            else:
                kwargs[k] = self.default_video_processor_kwargs[k]
        # Priority: num_frames -> fps + max/min_frames -> default_num_frames -> default_fps + default_max/min_frames
        if video.num_frames is not None:
            kwargs["num_frames"] = video.num_frames
        elif video.fps is not None:
            kwargs["fps"] = video.fps
            if video.max_frames is not None:
                kwargs["max_frames"] = video.max_frames
            if video.min_frames is not None:
                kwargs["min_frames"] = video.min_frames
        elif self.default_video_processor_kwargs["num_frames"] is not None:
            kwargs["num_frames"] = self.default_video_processor_kwargs["num_frames"]
        elif self.default_video_processor_kwargs["fps"] is not None:
            kwargs["fps"] = self.default_video_processor_kwargs["fps"]
            if self.default_video_processor_kwargs["max_frames"] is not None:
                kwargs["max_frames"] = self.default_video_processor_kwargs["max_frames"]
            if self.default_video_processor_kwargs["min_frames"] is not None:
                kwargs["min_frames"] = self.default_video_processor_kwargs["min_frames"]
        else:
            raise ValueError("Video sampling strategy not specified")
        return kwargs

    def preprocess_audio(self, audio: str | bytes):
        """
        - Input: audio filename string, bytes, or tuple of (waveform, original_sr)
        - Output:
            - mel spectrogram: torch.Tensor (T, n_mels)
            - number of tokens: int
        """
        assert isinstance(audio, (str, bytes, tuple)), f"audio must be a str, bytes or tuple, but got {type(audio)}"
        if isinstance(audio, tuple):
            waveform, original_sr = audio
        else:
            if isinstance(audio, bytes):
                file = io.BytesIO(audio)
            elif isinstance(audio, str):
                if audio.startswith("data:"):
                    file = io.BytesIO(pybase64.b64decode(audio.split(",")[1], validate=True))
                elif audio.startswith("http://") or audio.startswith("https://"):
                    dl_start = time.perf_counter()
                    timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                    try:
                        response = self.http_session.get(audio, stream=True, timeout=timeout)
                        dl_elapsed_ms = (time.perf_counter() - dl_start) * 1000
                        if dl_elapsed_ms > 1000.0:
                            content_len = len(response.content)
                            logger.warning(
                                f"Slow audio download: {dl_elapsed_ms:.2f}ms, "
                                f"size={content_len / 1024:.1f}KB, url={audio}"
                            )
                        file = io.BytesIO(response.content)
                        response.close()
                    except Exception as e:
                        dl_elapsed_ms = (time.perf_counter() - dl_start) * 1000
                        logger.error(
                            f"Failed to download audio: {dl_elapsed_ms:.2f}ms, "
                            f"error={type(e).__name__}: {e}, url={audio}"
                        )
                        raise
                else:
                    file = audio
            try:
                samples = AudioDecoder(file).get_all_samples()
            except RuntimeError as e:
                audio_source = audio if isinstance(audio, str) and (audio.startswith("http://") or audio.startswith("https://")) else "<bytes or base64>"
                logger.error(
                    f"Failed to decode audio: {e}, source={audio_source}"
                )
                raise ValueError(
                    f"Invalid audio format: source={audio_source}, detail={e}"
                ) from e
            waveform = samples.data  # torch.Tensor
            original_sr = samples.sample_rate  # int

        if original_sr != self.audio_sampling_rate:
            if original_sr in self._resamplers:
                self._resamplers.move_to_end(original_sr)
            else:
                if len(self._resamplers) >= self._resamplers_max:
                    self._resamplers.popitem(last=False)
                self._resamplers[original_sr] = torchaudio.transforms.Resample(
                    orig_freq=original_sr, new_freq=self.audio_sampling_rate
                )
            waveform = self._resamplers[original_sr](waveform)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)                     # (wav_len,)
        spec = self.mel_spectrogram(waveform[None, :])          # (1, n_mels, seq_len)
        spec = torch.log(torch.clip(spec, min=1e-7)).squeeze()  # (n_mels, seq_len)
        spec = spec.transpose(0, 1)                             # (seq_len, n_mels)

        audio_token_len = spec.shape[0] + 3 - self.audio_kernel_size
        audio_token_len = (audio_token_len + 2 - self.audio_kernel_size) // self.audio_stride_size + 1
        audio_token_len = audio_token_len // self.audio_avg_pooler + int(audio_token_len % self.audio_avg_pooler != 0)        # This is the length for input local transformer
        audio_token_len = math.ceil(audio_token_len / self.audio_group_size)        # This is the length for LM

        return spec, audio_token_len

    def process_image(self, image: ImageInput):
        """
        - Input: ImageInput
        - Output: nomrmalized image, torch.Tensor (C, H, W)
        """
        kwargs = self.prepare_image_kwargs(image)
        image = image.image
        if isinstance(image, (str, bytes)):
            image = fetch_image(image)
        image_transformed_tensor, _, _ = get_visual_transform(
            image, 
            factor=self.patch_size*self.merge_size, 
            min_pixels=kwargs["min_pixels"], 
            max_pixels=kwargs["max_pixels"],
            device=self.device,
        )
        return image_transformed_tensor


    def process_video(self, video_input: VideoInput | VideoAudioInput, temporal_padding_factor=None):

        def smart_resize_video(num_total_frames, min_pixels, max_pixels, total_max_pixels, **kwargs):
            max_pixels_per_frame = total_max_pixels * self.temporal_patch_size * self.temporal_compression_ratio // num_total_frames
            max_pixels = max(min_pixels, min(max_pixels_per_frame, max_pixels))
            return min_pixels, max_pixels

        def segment_frame_selector(all_timestamps, start_time, end_time):
            """
            Select frame indices from all_timestamps based on the start_time, end_time

            Principle 1: 左闭右开，纳入所有在区间 [start, end) 内的帧
                all_timestamps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                0-0 --> [0]
                0-1 --> [0]
                0-2 --> [0, 1]
                1-1 --> [1]

            Principle 2: 至少选择一帧，如果一帧都没有，则按照向左选取最近帧
                all_timestamps = [0, 3, 6, 9, 12, 15]
                1-1 --> [0]
                1-3 --> [0]
                1-4 --> [3]
                all_timestamps = [0, 1.1, 2.2, 3.3, 4.4, 5.5]
                1-1 --> [0]
                1-2 --> [1.1]
                1.2-2 --> [1.1]
            """
            # Convert to tensor if needed
            if not isinstance(all_timestamps, torch.Tensor):
                all_timestamps = torch.tensor(all_timestamps)

            # Find all frames in range [start_time, end_time)
            mask = (all_timestamps >= start_time) & (all_timestamps < end_time)
            candidate_indices = torch.where(mask)[0]

            # If no frames found, apply Principle 2: select the closest frame to the left
            if len(candidate_indices) == 0:
                # Find frames before or at start_time
                left_mask = all_timestamps <= start_time
                left_indices = torch.where(left_mask)[0]
                if len(left_indices) > 0:
                    # Select the rightmost frame before start_time
                    selected_frame_indices = left_indices[-1:].clone()
                else:
                    # No frames before start_time
                    raise ValueError(f"No frames before start_time {start_time} in all_timestamps {all_timestamps.tolist()}")
            else:
                # Frames found in the range
                selected_frame_indices = candidate_indices

            assert len(selected_frame_indices) > 0, f"No frames selected for segment {start_time} - {end_time} in all_timestamps {all_timestamps.tolist()}"
            return selected_frame_indices

        kwargs = self.prepare_video_kwargs(video_input)
        video = video_input.video

        if not isinstance(video, tuple):
            raise ValueError(
                f"video must be a tuple of (video_tensor, timestamps), but got {type(video)}. "
                "Video download and decoding should be done by sglang load_video before calling process_video."
            )

        video_tensor, timestamps_sampled = video
        if len(timestamps_sampled) < 2:
            logger.info("[Warning] Less than two frames are sampled, using default fps (1 fps)")
            fps_sampled = 1
        else:
            fps_sampled = 1 / (timestamps_sampled[1] - timestamps_sampled[0])
        num_frames_sampled = video_tensor.shape[0]

        start_time = video_input.start_time if video_input.start_time is not None else timestamps_sampled[0]
        end_time = video_input.end_time if video_input.end_time is not None else timestamps_sampled[-1] + (1 / fps_sampled)

        if video_input.segment_type == "individual":
            start_time_seg = start_time
            end_time_seg = end_time
            timestamps_seg = timestamps_sampled
            frames = video_tensor
            num_frames_seg = num_frames_sampled
        else:
            # Use segment_frame_selector to select frames
            selected_indices = segment_frame_selector(timestamps_sampled, start_time, end_time)

            timestamps_seg = timestamps_sampled[selected_indices]
            frames = video_tensor[selected_indices]
            num_frames_seg = len(timestamps_seg)
            start_time_seg = timestamps_seg[0].item() if isinstance(timestamps_seg[0], torch.Tensor) else timestamps_seg[0]
            end_time_seg = (timestamps_seg[-1].item() if isinstance(timestamps_seg[-1], torch.Tensor) else timestamps_seg[-1]) + (1 / fps_sampled).item()

        video_meta = {
            "fps_sampled": fps_sampled,
            "segment_start_time": start_time_seg,
            "segment_end_time": end_time_seg,
        }

        min_pixels, max_pixels = smart_resize_video(num_frames_sampled, **kwargs)

        assert num_frames_seg > 0, f"Sampled frame number must be >0. start_time {video_input.start_time}, end_time {video_input.end_time}, start_time_seg {start_time_seg}, end_time_seg {end_time_seg}. Full timestamps {timestamps_sampled.tolist()}. "

        # Align num_frames_seg to temporal_padding_factor
        temporal_padding_factor = self.temporal_patch_size * self.temporal_compression_ratio if temporal_padding_factor is None else temporal_padding_factor

        if num_frames_seg % temporal_padding_factor == 0:
            # Already aligned, no need to do anything
            aligned_frames = frames
            aligned_timestamps = timestamps_seg
        else:
            # Not aligned, replicate last frame until aligned
            aligned_num_frames = ((num_frames_seg + temporal_padding_factor - 1) // temporal_padding_factor) * temporal_padding_factor
            num_frames_needed = aligned_num_frames - num_frames_seg
            aligned_frames = torch.cat([frames, frames[-1:].repeat(num_frames_needed, *[1]*(frames.ndim-1))], dim=0)
            aligned_timestamps = torch.cat([timestamps_seg, timestamps_seg[-1:].repeat(num_frames_needed)], dim=0)

        # Batch视频帧变换（使用PyTorch的F.interpolate进行高效的批量resize）
        video_transformed_tensor, _, _ = get_visual_transform_batch(
            aligned_frames,
            factor=self.patch_size*self.merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            device=self.device,
        )          # (t, c, h, w)

        visual_patches, thw_grid = self._flatten_visual_inputs(video_transformed_tensor, "video")
        return visual_patches, thw_grid, aligned_timestamps, video_meta


    def process_audio(self, audio: AudioInput):
        """
        - Input: AudioInput
        - Output:
            - Inference: audio is str/bytes/np.ndarray, return mel_spectrogram and number of tokens
            - Training: audio is torch.Tensor, return padded_audio
        """
        audio = audio.audio
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            audio = (waveform, self.audio_sampling_rate)
        if isinstance(audio, (str, bytes, tuple)):
            audio_spec, audio_token_len = self.preprocess_audio(audio)
            return audio_spec, audio_token_len

        assert audio.shape[1] >= self.audio_channels, f"audio must have at least {self.audio_channels} channels, but got {audio.shape[1]}"
        T = audio.shape[0]
        audio = audio[:,:self.audio_channels].to(torch.long)
        padded_T = (T + self.audio_group_size - 1) // self.audio_group_size * self.audio_group_size
        padded_audio = torch.cat([audio, torch.zeros(padded_T - T, self.audio_channels, dtype=torch.long) + audio[-1, :]], dim=0)   # pad using the last embedding
        padded_audio = padded_audio.reshape(padded_T // self.audio_group_size, self.audio_group_size, self.audio_channels)
        return padded_audio

    def convert_conversation_to_contents(
        self,
        conversation: Conversation,
        apply_chat_template=True,
        continue_final_message=False,
        end_of_think_mask=False,
        last_turn_only=False,
    ):
        contents = []

        if apply_chat_template:
            # get default system prompt length for constructing user/assistant messages
            default_system_prompt = self.tokenizer.apply_chat_template([{}], tokenize=False)
            default_system_prompt_length = len(default_system_prompt)

            # add system prompt
            if (len(conversation) > 0 and conversation[0].role == "system"):
                # custom system prompt
                assert len(conversation[0].contents) == 1 and conversation[0].contents[0].type == "text"
                system_prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": conversation[0].contents[0].content}], tokenize=False)
            else:
                system_prompt = default_system_prompt
            if len(system_prompt) > 0:
                contents.append(Content(type="text", content=system_prompt, is_target=False))
        else:
            default_system_prompt_length = 0

        for turn_idx, turn in enumerate(conversation):
            # system prompt has been handled before
            if turn.role == "system":
                continue

            # FIXME: use is_target in contents?
            is_target = (turn.role == "assistant") and ((not last_turn_only) or turn_idx == len(conversation) - 1)
            if apply_chat_template:
                turn_text = self.tokenizer.apply_chat_template([{"role": turn.role, "content": MIMO_PLACEHOLDER}], tokenize=False)
                turn_start_text, turn_end_text = turn_text[default_system_prompt_length:].split(MIMO_PLACEHOLDER)
                contents.append(Content(type="text", content=turn_start_text, is_target=is_target))

            for content in turn.contents:
                if content.type == "text":
                    text = content.content
                    if content.is_target is not None:
                        text_is_target = content.is_target
                    else:
                        text_is_target = is_target

                    # TODO: check
                    if text_is_target and end_of_think_mask:
                        text_split = text.split("</think>\n\n")
                        for i in range(len(text_split)-1):
                            contents.append(Content(type="text", content=text_split[i], is_target=text_is_target))
                            contents.append(Content(type="text", content="</think>\n\n", is_target=False))
                        contents.append(Content(type="text", content=text_split[-1], is_target=text_is_target))
                    else:
                        contents.append(Content(type="text", content=text, is_target=text_is_target))
                elif content.type == "image":
                    contents.append(Content(type="image", content=content.content, is_target=is_target))
                elif content.type == "video":
                    contents.append(Content(type="video", content=content.content, is_target=is_target))
                elif content.type == "audio":
                    contents.append(Content(type="audio", content=content.content, is_target=is_target))
                elif content.type == "video_audio":
                    contents.append(Content(type="video_audio", content=content.content, is_target=is_target))
                else:
                    raise ValueError(f"unexpected content type {content.type} in {turn.contents}")

            if apply_chat_template and not (continue_final_message and turn_idx == len(conversation) - 1):
                contents.append(Content(type="text", content=turn_end_text, is_target=is_target))
        # print(f"=====contents: {contents}")
        return contents

    def process_conversation(
        self,
        conversation: Conversation,
        apply_chat_template=True,
        continue_final_message=False,
        end_of_think_mask=False,
        last_turn_only=False,
    ):
        contents = self.convert_conversation_to_contents(
            conversation,
            apply_chat_template=apply_chat_template,
            continue_final_message=continue_final_message,
            end_of_think_mask=end_of_think_mask,
            last_turn_only=last_turn_only,
        )
        return self.process(contents)


    def process(
            self,
            contents: list[Content],
            verbose: bool = True,
        ):
        verbose = False
        input_ids, labels = [], []
        image_pixel_values, image_thw_grids = [], []
        video_pixel_values, video_thw_grids = [], []
        audio_inputs = []
        is_audio_tokenized = []
        second_per_grid_ts = []             # for mrope
        extra = {}

        verbose_str = ""

        # Pre-process all videos in parallel for better performance
        video_contents_info = []  # List of (index, content, is_video_audio)
        for idx, content in enumerate(contents):
            if content.type == "video":
                video_contents_info.append((idx, content.content, False))
            elif content.type == "video_audio":
                video_contents_info.append((idx, content.content, True))

        # Process videos in parallel using ThreadPoolExecutor
        video_results = {}  # index -> (video_tensor, timestamps, video_meta)
        if len(video_contents_info) > 0:
            num_threads = min(self.video_process_num_threads, len(video_contents_info))
            if num_threads > 1 and len(video_contents_info) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    future_to_idx = {
                        executor.submit(self.process_video, video_input): idx
                        for idx, video_input, _ in video_contents_info
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            video_results[idx] = future.result()
                        except Exception as e:
                            raise RuntimeError(f"Error processing video at index {idx}: {e}") from e
            else:
                # Sequential processing (when only 1 video or num_threads <= 1)
                for idx, video_input, _ in video_contents_info:
                    video_results[idx] = self.process_video(video_input)

        for content_idx, content in enumerate(contents):
            _labels = None
            if content.type == "text":
                if isinstance(content.content, str):
                    _input_ids = self.tokenizer.encode(content.content)
                else:
                    _input_ids = content.content
                if content.is_target:
                    _labels = _input_ids

                if verbose:
                    if isinstance(content.content, str):
                        verbose_str += f"Text: [{content.content}]\n"
                    else:
                        verbose_str += f"Text: [{self.tokenizer.decode(content.content)}]\n"
            # Only text can be target
            elif content.type == "image":
                image_tensor = self.process_image(content.content)
                visual_patches, thw_grid = self._flatten_visual_inputs(image_tensor, "image")
                grid_t, grid_h, grid_w = thw_grid
                num_media_tokens = (grid_t * grid_h * grid_w) // (self.merge_size ** 2)
                image_pixel_values.append(visual_patches)
                image_thw_grids.append(thw_grid)
                _input_ids = [self.vision_start_token_id] + [self.image_token_id] * num_media_tokens + [self.vision_end_token_id]

                if verbose:
                    verbose_str += f"Image (shape={image_tensor.shape}, image_thw_grid={thw_grid}): [<vision_start> {num_media_tokens}*<vision> <vision_end>]\n"

            elif content.type == "video":
                # Use pre-computed video results from parallel processing
                visual_patches, thw_grid, timestamps, video_meta = video_results[content_idx]
                grid_t, grid_h, grid_w = thw_grid
                num_media_tokens = (grid_t * grid_h * grid_w) // (self.merge_size ** 2) // self.temporal_compression_ratio
                video_pixel_values.append(visual_patches)
                video_thw_grids.append(thw_grid)

                assert len(timestamps) == grid_t * self.temporal_patch_size, f"Expected {grid_t} * {self.temporal_patch_size} = {grid_t * self.temporal_patch_size} timestamps, but got {len(timestamps)}"

                if self.use_video_timestamps:
                    num_media_tokens_per_grid = grid_h * grid_w // (self.merge_size ** 2)
                    text_timestamps = [format_timestamp(ts) for ts in timestamps[::self.temporal_patch_size*self.temporal_compression_ratio]]
                    text_timestamp_ids = [self.tokenizer.encode(ts) for ts in text_timestamps]
                    _input_ids = ([self.video_start_token_id] +
                        sum([
                            ts_ids +
                            [self.vision_start_token_id] +
                            [self.video_token_id] * num_media_tokens_per_grid +
                            [self.vision_end_token_id] for ts_ids in text_timestamp_ids], []) +
                        [self.video_end_token_id]
                    )
                    if verbose:
                        verbose_str += f"Video (video_thw_grid={thw_grid}, video_meta={video_meta}): [<video_start> "
                        for i, ts in enumerate(text_timestamps):
                            verbose_str += f"{ts} <vision_start> {timestamps.tolist()[i*self.temporal_patch_size*self.temporal_compression_ratio : (i+1)*self.temporal_patch_size*self.temporal_compression_ratio]} {num_media_tokens_per_grid}*<vision> <vision_end> "
                        verbose_str += "<video_end>]\n"

                else:
                    # not implemented after adding video_start/video_end tags
                    raise NotImplementedError
                    _input_ids = [self.vision_start_token_id] + [self.video_token_id] * num_media_tokens + [self.vision_end_token_id]
                    if verbose:
                        verbose_str += f"Video (video_thw_grid={thw_grid}, video_meta={video_meta}): [<vision_start> {num_media_tokens}*<vision> <vision_end>]\n"

                second_per_grid_ts.append(self.temporal_patch_size / video_meta['fps_sampled'])
            elif content.type == "audio":
                audio = content.content
                processed_audio = self.process_audio(audio)
                if isinstance(processed_audio, tuple):
                    is_audio_tokenized.append(False)
                    audio_spec, audio_token_len = processed_audio
                    audio_inputs.append(audio_spec)
                else:
                    audio_inputs.append(processed_audio)
                    audio_token_len = processed_audio.shape[0]
                    is_audio_tokenized.append(True)
                _input_ids = [self.audio_start_token_id] + [self.audio_token_id] * audio_token_len + [self.audio_end_token_id]

                if verbose:
                    verbose_str += f"Audio (is_tokenized={is_audio_tokenized[-1]}): [<audio_start> {audio_token_len}*<audio> <audio_end>]\n"

            elif content.type == "video_audio":
                video = content.content.video
                audio = content.content.audio

                # Use pre-computed video results from parallel processing
                visual_patches, thw_grid, timestamps, video_meta = video_results[content_idx]  # timestamps is a list of seconds: e.g., [2.5, 5, 7.5, 10]
                second_per_grid_ts.append(self.temporal_patch_size / video_meta['fps_sampled'])

                processed_audio = self.process_audio(content.content)
                audio_token_per_second = self.audio_input_id_per_second / self.audio_group_size

                grid_t, grid_h, grid_w = thw_grid
                num_media_tokens = (grid_t * grid_h * grid_w) // (self.merge_size ** 2) // self.temporal_compression_ratio
                video_pixel_values.append(visual_patches)
                video_thw_grids.append(thw_grid)

                if self.use_video_timestamps:
                    if isinstance(processed_audio, tuple):
                        assert content.content.start_time is None and content.content.end_time is None, "Audio start_time and end_time must be None when audio is not tokenized"
                        is_audio_tokenized.append(False)
                        audio_spec, audio_token_len = processed_audio
                        audio_inputs.append(audio_spec)
                    else:
                        is_audio_tokenized.append(True)
                        audio_token_len = processed_audio.shape[0]

                    video_audio_units = []        # (timestamp, timestamp_text, timestamp_ids, video_token_len, audio_token_len, video, audio)

                    num_media_tokens_per_grid = grid_h * grid_w // (self.merge_size ** 2)
                    grid_t_timestamps = timestamps[::self.temporal_patch_size*self.temporal_compression_ratio]
                    text_timestamps = [format_timestamp(ts) for ts in grid_t_timestamps]
                    text_timestamp_ids = [self.tokenizer.encode(ts) for ts in text_timestamps]

                    # bind each grid_t video, corresponding timestamps and audio
                    for i in range(len(grid_t_timestamps)):
                        timestamp = grid_t_timestamps[i]
                        timestamp_text = text_timestamps[i]
                        timestamp_ids = text_timestamp_ids[i]
                        video_token_len = num_media_tokens_per_grid
                        segment_video = None

                        audio_start_token_idx = int(grid_t_timestamps[i] * audio_token_per_second)
                        audio_end_token_idx = int(grid_t_timestamps[i+1] * audio_token_per_second) if i < len(grid_t_timestamps) - 1 else int(video_meta["segment_end_time"] * audio_token_per_second)

                        segment_audio_token_len = min(audio_end_token_idx, audio_token_len) - audio_start_token_idx
                        assert segment_audio_token_len > 0
                        segment_audio = processed_audio[audio_start_token_idx:audio_start_token_idx+segment_audio_token_len] if is_audio_tokenized[-1] else None
                        video_audio_units.append((timestamp, timestamp_text, timestamp_ids, video_token_len, segment_audio_token_len, segment_video, segment_audio))

                    # group based on video_audio_interleave_length
                    if self.video_audio_interleave_length == -1:
                        groups = [[(i, u) for i, u in enumerate(video_audio_units)]]
                    elif self.video_audio_interleave_length == 0:
                        groups = [[(i, u)] for i, u in enumerate(video_audio_units)]
                    else:
                        assert self.video_audio_interleave_length > 0
                        groups = []
                        unit_idx = 0
                        current_group = []
                        time_ptr = 0
                        while unit_idx < len(video_audio_units):
                            while unit_idx < len(video_audio_units) and video_audio_units[unit_idx][0] >= time_ptr and video_audio_units[unit_idx][0] < time_ptr + self.video_audio_interleave_length:
                                current_group.append((unit_idx, video_audio_units[unit_idx]))
                                unit_idx += 1
                            if len(current_group) > 0:
                                groups.append(current_group)
                                current_group = []
                            time_ptr += self.video_audio_interleave_length

                    # each group follows the format <timestamp> <frames> <audio_start> <audio> <audio_end>

                    _input_ids = [self.video_start_token_id]
                    if verbose: verbose_str += f"VideoAudio (video_thw_grid={thw_grid}, video_meta={video_meta}, is_audio_tokenized={is_audio_tokenized[-1]}, audio_token_len={audio_token_len}): [<video_start> "
                    for group in groups:
                        if not self.use_per_grid_t_timestamps:
                            _input_ids += group[0][1][2]
                            if verbose: verbose_str += f"{group[0][1][1]} "
                        _video_tokens, _audio_tokens = [], []
                        video_verbose_str, audio_verbose_str = "", ""
                        for unit_idx, unit in group:
                            timestamp, timestamp_text, timestamp_ids, video_token_len, segment_audio_token_len, segment_video, segment_audio = unit
                            if self.use_per_grid_t_timestamps:
                                _video_tokens += timestamp_ids
                                _audio_tokens += timestamp_ids
                                video_verbose_str += timestamp_text + " "
                                audio_verbose_str += timestamp_text + " "
                            _video_tokens += [self.vision_start_token_id] + [self.video_token_id] * video_token_len + [self.vision_end_token_id]
                            video_verbose_str += f"[{','.join([f'{ts:.2f}' for ts in timestamps.tolist()[unit_idx*self.temporal_patch_size*self.temporal_compression_ratio : (unit_idx+1)*self.temporal_patch_size*self.temporal_compression_ratio]])}] <vision_start> {video_token_len}*<video> <vision_end> "
                            _audio_tokens += [self.audio_token_id] * segment_audio_token_len
                            audio_verbose_str += f"{segment_audio_token_len}*<audio> "
                            assert segment_video is None
                            if segment_audio is not None:
                                audio_inputs.append(segment_audio)

                        _input_ids += _video_tokens + [self.audio_start_token_id] + _audio_tokens + [self.audio_end_token_id]
                        if verbose: verbose_str += f"{video_verbose_str}<audio_start> {audio_verbose_str}<audio_end> "
                    _input_ids += [self.video_end_token_id]
                    if verbose: verbose_str += "<video_end>]\n"

                else:
                    # not implemented after adding video_start/video_end tags
                    raise NotImplementedError
                    _input_ids = [self.vision_start_token_id] + [self.video_token_id] * num_media_tokens + [self.vision_end_token_id]
                    if verbose: verbose_str += f"Video (shape={video_tensor.shape}, video_thw_grid={thw_grid}, video_meta={video_meta}): [<vision_start> {num_media_tokens}*<vision> <vision_end>]\n"

                    if not is_audio_tokenized[-1]:
                        _input_ids += [self.audio_start_token_id] + [self.audio_token_id] * audio_token_len + [self.audio_end_token_id]
                        if verbose: verbose_str += f"Audio (is_tokenized={is_audio_tokenized[-1]}, shape={processed_audio[0].shape}): [<audio_start> {audio_token_len}*<audio> <audio_end>]\n"
                    else:
                        start_time, end_time = video_meta["segment_start_time"], video_meta["segment_end_time"]
                        start_token_idx = int(start_time * audio_token_per_second)
                        end_token_idx = int(end_time * audio_token_per_second)
                        audio_inputs.append(processed_audio[start_token_idx:end_token_idx])
                        audio_token_len = processed_audio[start_token_idx:end_token_idx].shape[0]
                        _input_ids += [self.audio_start_token_id] + [self.audio_token_id] * audio_token_len + [self.audio_end_token_id]

                        if verbose: verbose_str += f"Audio (is_tokenized={is_audio_tokenized[-1]}, shape={processed_audio.shape}): [<audio_start> {audio_token_len}*<audio> <audio_end>]\n"

            input_ids.extend(_input_ids)
            labels.extend(_labels or [self.pad_token_id] * len(_input_ids))

        input_ids = torch.tensor(input_ids)
        labels = np.roll(labels, shift=-1)
        labels[-1] = self.pad_token_id
        labels = torch.tensor(labels)

        if len(is_audio_tokenized) > 0:
            assert all(is_audio_tokenized) or not any(is_audio_tokenized), "All audio inputs must be tokenized or not tokenized"
            extra["is_audio_tokenized"] = is_audio_tokenized[0]

        if self.rope_type == "rope":
            position_ids = torch.arange(input_ids.shape[0]).expand(3, -1)
            rope_deltas = torch.zeros((1, 1), dtype=torch.int32)
        elif self.rope_type == "mrope":
            from .rope_utils import get_rope_index
            position_ids, rope_deltas = get_rope_index(
                spatial_merge_size=self.merge_size,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                model_type="qwen2_5_vl",
                tokens_per_second=self.video_tokens_per_second,
                image_grid_thw=image_thw_grids if len(image_thw_grids) > 0 else None,
                video_grid_thw=video_thw_grids if len(video_thw_grids) > 0 else None,
                second_per_grid_ts=second_per_grid_ts,
                input_ids=input_ids[None, :],
            )
            position_ids = position_ids.squeeze(1)
            # print(position_ids.shape, rope_deltas.shape)

        if verbose:
            print(verbose_str.strip())

        return MiMoVLInputSample(
            input_ids=input_ids,
            labels=labels,
            pixel_values=image_pixel_values,
            pixel_values_videos=video_pixel_values,
            image_thw_grids=image_thw_grids,
            video_thw_grids=video_thw_grids,
            audio_inputs=audio_inputs,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            extra=extra
        )


    def _flatten_visual_inputs(self, visual: torch.Tensor, visual_type: str):
        if visual_type == 'image':
            resized_height, resized_width = visual.shape[-2:]
            patches = visual.unsqueeze(0).repeat(self.temporal_patch_size, 1, 1, 1)
        elif visual_type == 'video' or visual_type == 'video_audio':
            assert len(visual) % (self.temporal_compression_ratio * self.temporal_patch_size) == 0
            patches = visual
            resized_height, resized_width = patches.shape[-2:]
        else:
            raise ValueError(f"Unknown visual_type: {visual_type}")

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.contiguous().view(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        # (grid_t, grid_h/merge, grid_w/merge, merge_h, merge_w, channel, temporal, patch_h, patch_w)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        
        # (num_patches, patch_dim)
        flatten_patches = patches.view(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        thw_grids = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.int32)

        return flatten_patches, thw_grids


# ---------------------------------------------------------------------------
# MiMoV2OmniProcessor (sglang integration)
# ---------------------------------------------------------------------------

use_image_processor_gpu = (
    int(os.getenv("SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU", "0")) == 1
)


class _AsNumpyArray:
    def __init__(self, array):
        self._array = array

    def asnumpy(self):
        return self._array


def _as_numpy_array(batch):
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu().numpy()
    if isinstance(batch, np.ndarray):
        return batch
    if hasattr(batch, "numpy"):
        return batch.numpy()
    return None


class _VideoReaderAsNumpyAdapter:
    def __init__(self, reader):
        self._reader = reader

    def __len__(self):
        return len(self._reader)

    def get_avg_fps(self):
        return self._reader.get_avg_fps()

    def get_batch(self, idx):
        batch = self._reader.get_batch(idx)
        if hasattr(batch, "asnumpy"):
            return batch
        array = _as_numpy_array(batch)
        if array is None:
            return batch
        return _AsNumpyArray(array)


def _wrap_video_reader(video):
    if all(hasattr(video, attr) for attr in ("get_batch", "__len__", "get_avg_fps")):
        return _VideoReaderAsNumpyAdapter(video)
    return video


def _to_mimo_video(video_result):
    if (
        isinstance(video_result, tuple)
        and len(video_result) == 2
        and isinstance(video_result[1], dict)
    ):
        video_tensor, metadata = video_result
        frames_indices = metadata.get("frames_indices")
        if frames_indices is None:
            frames_indices = np.arange(video_tensor.shape[0])
        fps = metadata.get("fps")
        if fps:
            timestamps = torch.as_tensor(frames_indices, dtype=torch.float32) / float(
                fps
            )
        else:
            timestamps = torch.as_tensor(frames_indices, dtype=torch.float32)
        return (video_tensor, timestamps)
    return video_result


def has_audio_track(path_or_data: str) -> bool:
    try:
        is_base64 = path_or_data.startswith("data:") and ";base64," in path_or_data
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "a",
            "pipe:0" if is_base64 else path_or_data,
        ]
        inp = base64.b64decode(path_or_data.split(";base64,")[1]) if is_base64 else None
        r = subprocess.run(cmd, input=inp, capture_output=True, timeout=30)
        return bool(r.returncode == 0 and json.loads(r.stdout).get("streams"))
    except Exception:
        return False


class MiMoV2OmniProcessor(BaseMultimodalProcessor):
    models = [MiMoV2OmniForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.vision_config = Qwen2_5_VLVisionConfig.from_dict(hf_config.vision_config)

        patch_size = self.vision_config.patch_size
        spatial_merge_size = getattr(self.vision_config, "spatial_merge_size", 2)
        unit_size = patch_size * spatial_merge_size
        self.image_factor = unit_size

        rope_type = "rope"
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling:
            if (
                rope_scaling.get("type", None) == "default"
                and rope_scaling.get("mrope_section", None) is not None
            ):
                rope_type = "mrope"

        processor_config = getattr(hf_config, "processor_config", {})
        audio_config = getattr(hf_config, "audio_config", None)
        audio_sample_rate = None
        if isinstance(audio_config, dict):
            audio_sample_rate = audio_config.get("sampling_rate") or audio_config.get(
                "sample_rate"
            )
        elif audio_config is not None:
            audio_sample_rate = getattr(audio_config, "sampling_rate", None) or getattr(
                audio_config, "sample_rate", None
            )
        self.audio_sample_rate = (
            processor_config.get("audio_sampling_rate", None)
            or audio_sample_rate
            or 16000
        )

        self.IM_START_TOKEN_ID = processor_config.get("vision_start_token_id", None)
        self.IM_END_TOKEN_ID = processor_config.get("vision_end_token_id", None)
        self.IM_TOKEN_ID = processor_config.get("image_token_id", None)
        self.VIDEO_TOKEN_ID = processor_config.get("video_token_id", None)
        self.vision_start_token_id = processor_config.get("vision_start_token_id", None)
        self.vision_end_token_id = processor_config.get("vision_end_token_id", None)

        self.AUDIO_TOKEN_ID = processor_config.get("audio_token_id", None)
        self.AUDIO_START_TOKEN_ID = processor_config.get("audio_start_token_id", None)
        self.AUDIO_END_TOKEN_ID = processor_config.get("audio_end_token_id", None)

        self.video_start_token_id = processor_config.get("video_start_token_id", None)
        self.video_end_token_id = processor_config.get("video_end_token_id", None)

        deivce = server_args.device if use_image_processor_gpu else None

        self.mimo_processor = MiMoVLProcessor(
            tokenizer=self._processor.tokenizer,
            patch_size=patch_size,
            image_min_pixels=processor_config.get("image_min_pixels", None)
            or 4 * unit_size * unit_size,
            image_max_pixels=processor_config.get("image_max_pixels", None)
            or 4096 * unit_size * unit_size,
            video_min_pixels=processor_config.get("video_min_pixels", None)
            or 4 * unit_size * unit_size,
            video_max_pixels=processor_config.get("video_max_pixels", None)
            or 4096 * unit_size * unit_size,
            video_total_max_pixels=processor_config.get("video_total_max_pixels", None)
            or 16384 * unit_size * unit_size,
            fps=processor_config.get("fps", None) or 2,
            num_frames=processor_config.get("num_frames", None),
            max_frames=processor_config.get("max_frames", None) or 256,
            min_frames=processor_config.get("min_frames", None) or 8,
            video_audio_interleave_length=processor_config.get(
                "video_audio_interleave_length", 0
            ),
            use_per_grid_t_timestamps=processor_config.get(
                "use_per_grid_t_timestamps", False
            ),
            image_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            audio_token_id=self.AUDIO_TOKEN_ID,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            audio_start_token_id=self.AUDIO_START_TOKEN_ID,
            audio_end_token_id=self.AUDIO_END_TOKEN_ID,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
            pad_token_id=self._processor.tokenizer.pad_token_id,
            rope_type=rope_type,
            use_video_timestamps=processor_config.get("use_video_timestamps", False),
            device=deivce,
        )
        self._processor = self.mimo_processor

        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|mimo_audio_start\|>(?:<\|audio_pad\|>)+<\|mimo_audio_end\|>"
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token="<|vision_start|><|video_pad|><|vision_end|>",
            video_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=self.VIDEO_TOKEN_ID,
            audio_token="<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>",
            audio_token_id=self.AUDIO_TOKEN_ID,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
        ).build(_processor)

    @property
    def spatial_merge_size(self):
        return self.vision_config.spatial_merge_size

    def build_input_ids(self, prompt, grid_thw, mm_token_id, modality_name):
        if not isinstance(prompt, list):
            prompt = self.mimo_processor.tokenizer.encode(prompt)

        spatial_merge_size = self.spatial_merge_size
        input_ids = []
        offsets = []

        cur_idx = 0
        mm_start_indices = list(
            filter(lambda i: prompt[i + 1] == mm_token_id, range(len(prompt) - 1))
        )

        for cur_mm_idx, mm_start_idx in enumerate(mm_start_indices):
            assert cur_idx <= mm_start_idx
            # include img_start_id
            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
            mm_offset_start = len(input_ids)
            mm_token_num = grid_thw[cur_mm_idx].prod() // (spatial_merge_size**2)
            input_ids.extend([mm_token_id] * mm_token_num)
            # jump to img_end_id
            cur_idx = mm_start_idx + 2
            offsets.append((mm_offset_start, len(input_ids) - 1))
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets

    def build_audio_input_ids(self, prompt, audio_lens, audio_token_id):
        if not isinstance(prompt, list):
            prompt = self.mimo_processor.tokenizer.encode(prompt)

        if audio_lens is None:
            audio_lens = []
        if isinstance(audio_lens, torch.Tensor):
            audio_lens = audio_lens.flatten().tolist()

        audio_token_ids = {audio_token_id}
        try:
            audio_pad_id = self.mimo_processor.tokenizer.encode("<|audio_pad|>")[0]
            audio_token_ids.add(audio_pad_id)
        except Exception:
            audio_pad_id = None

        input_ids = []
        offsets = []
        cur_idx = 0

        audio_start_indices = []
        for i in range(len(prompt) - 1):
            if (
                prompt[i] == self.AUDIO_START_TOKEN_ID
                and prompt[i + 1] in audio_token_ids
            ):
                audio_start_indices.append(i)
            elif prompt[i] == self.AUDIO_START_TOKEN_ID:
                logger.warning(
                    "[EPD] Audio start token found without audio token. "
                    f"audio_start_id={self.AUDIO_START_TOKEN_ID}, "
                    f"audio_token_id={audio_token_id}, "
                    f"audio_pad_id={audio_pad_id}, "
                    f"prompt_len={len(prompt)}."
                )

        if len(audio_start_indices) == 0 and audio_lens:
            logger.warning(
                "[EPD] No audio tokens found in prompt. "
                f"audio_start_id={self.AUDIO_START_TOKEN_ID}, "
                f"audio_token_id={audio_token_id}, "
                f"prompt_len={len(prompt)}, audio_lens={audio_lens}. "
                f"First 50 tokens: {prompt}. "
            )
            audio_start_indices = [0]

        for cur_audio_idx, audio_start_idx in enumerate(audio_start_indices):
            assert cur_idx <= audio_start_idx
            input_ids.extend(prompt[cur_idx : audio_start_idx + 1])
            audio_offset_start = len(input_ids)
            audio_token_num = (
                int(audio_lens[cur_audio_idx]) if cur_audio_idx < len(audio_lens) else 0
            )
            input_ids.extend([audio_token_id] * audio_token_num)
            cur_idx = audio_start_idx + 2
            offsets.append((audio_offset_start, len(input_ids) - 1))
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets

    def get_mm_data(
        self,
        prompt,
        embeddings,
        grid_thw,
        *,
        modality=Modality.IMAGE,
        second_per_grid_ts=None,
    ):
        """
        Build mm_inputs from precomputed embeddings for EPD disaggregation mode.
        """
        if isinstance(modality, str):
            modality = Modality.from_str(modality)

        # Ensure prompt contains the appropriate multimodal token placeholder
        # This is needed for EPD disaggregation mode where prompt may not have placeholders
        prompt_text = prompt if isinstance(prompt, str) else ""
        if modality == Modality.AUDIO:
            if prompt_text and not self.AUDIO_TOKEN_REGEX.search(prompt_text):
                prompt = f"{self.mm_tokens.audio_token}{prompt_text}"
        elif modality == Modality.VIDEO:
            video_token_regex = self.mm_tokens.video_token_regex
            if (
                prompt_text
                and video_token_regex
                and not video_token_regex.search(prompt_text)
            ):
                # Insert video placeholder at the beginning if not present
                prompt = f"{self.mm_tokens.video_token}{prompt_text}"
        elif modality == Modality.IMAGE:
            image_token_regex = self.mm_tokens.image_token_regex
            if (
                prompt_text
                and image_token_regex
                and not image_token_regex.search(prompt_text)
            ):
                # Insert image placeholder at the beginning if not present
                prompt = f"{self.mm_tokens.image_token}{prompt_text}"

        if modality == Modality.VIDEO:
            mm_token_id = self.mm_tokens.video_token_id
            modality_name = "video"
            input_ids, offsets = self.build_input_ids(
                prompt, grid_thw, mm_token_id, modality_name
            )
        elif modality == Modality.AUDIO:
            mm_token_id = self.mm_tokens.audio_token_id
            modality_name = "audio"
            input_ids, offsets = self.build_audio_input_ids(
                prompt, grid_thw, mm_token_id
            )
        else:
            mm_token_id = self.mm_tokens.image_token_id
            modality_name = "image"
            input_ids, offsets = self.build_input_ids(
                prompt, grid_thw, mm_token_id, modality_name
            )
        if (
            second_per_grid_ts is None
            and modality == Modality.VIDEO
            and grid_thw is not None
        ):
            second_per_grid_ts = torch.ones((grid_thw.shape[0],), dtype=torch.float32)

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.vision_config.spatial_merge_size,
            image_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            vision_start_token_id=self.vision_start_token_id,
            model_type="qwen2_5_vl",
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            image_grid_thw=grid_thw if modality == Modality.IMAGE else None,
            video_grid_thw=grid_thw if modality == Modality.VIDEO else None,
            tokens_per_second=getattr(self.vision_config, "tokens_per_second", None),
            second_per_grid_ts=second_per_grid_ts,
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_item = MultimodalDataItem(
            modality=modality,
            offsets=offsets,
            precomputed_embeddings=embeddings,
        )
        if modality == Modality.IMAGE:
            mm_items = mm_item.split_by_offset()
        else:
            mm_items = [mm_item]

        return {
            "input_ids": input_ids,
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.mimo_processor.image_token_id,
            "video_token_id": self.mimo_processor.video_token_id,
            "audio_token_id": self.mimo_processor.audio_token_id,
            "audio_start_id": self.AUDIO_START_TOKEN_ID,
            "audio_end_id": self.AUDIO_END_TOKEN_ID,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

    def get_mm_data_video(
        self,
        prompt,
        video_embeddings,
        video_grid_thw,
        audio_embeddings=None,
        audio_lens=None,
        *,
        second_per_grid_ts=None,
    ):
        """
        Build mm_inputs from precomputed embeddings for video (and optionally audio) in EPD mode.

        When audio_embeddings and audio_lens are provided, this handles video+audio data.
        When they are None, this handles video-only data.
        """
        has_audio = audio_embeddings is not None and audio_lens is not None

        # Ensure prompt contains video token placeholder
        if isinstance(prompt, str):
            video_token_regex = self.mm_tokens.video_token_regex
            if video_token_regex and not video_token_regex.search(prompt):
                prompt = f"{self.mm_tokens.video_token}{prompt}"

            # If we have audio, also add audio token placeholder
            if has_audio and not self.AUDIO_TOKEN_REGEX.search(prompt):
                # Find the video end token and insert audio after it
                vision_end_pattern = r"<\|vision_end\|>"
                import re as _re

                match = _re.search(vision_end_pattern, prompt)
                if match:
                    # Insert audio token after vision_end
                    insert_pos = match.end()
                    prompt = (
                        prompt[:insert_pos]
                        + self.mm_tokens.audio_token
                        + prompt[insert_pos:]
                    )
                else:
                    # Fallback: append at the end of content (before assistant marker if present)
                    assistant_marker = "<|im_start|>assistant"
                    if assistant_marker in prompt:
                        idx = prompt.rfind(assistant_marker)
                        prompt = (
                            prompt[:idx] + self.mm_tokens.audio_token + prompt[idx:]
                        )
                    else:
                        prompt = prompt + self.mm_tokens.audio_token

        input_ids, video_offsets = self.build_input_ids(
            prompt, video_grid_thw, self.mm_tokens.video_token_id, "video"
        )

        audio_offsets = []
        if has_audio:
            audio_lens_list = (
                audio_lens.flatten().tolist()
                if isinstance(audio_lens, torch.Tensor)
                else audio_lens
            )
            input_ids, audio_offsets = self.build_audio_input_ids(
                input_ids,
                audio_lens_list,
                self.mm_tokens.audio_token_id,
            )

        rope_model_type = "qwen2_5_vl"
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.vision_config.spatial_merge_size,
            image_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            vision_start_token_id=self.vision_start_token_id,
            model_type=rope_model_type,
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts
            if second_per_grid_ts is not None
            else torch.ones((video_grid_thw.shape[0],), dtype=torch.float32),
            tokens_per_second=getattr(self.vision_config, "tokens_per_second", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_items = [
            MultimodalDataItem(
                modality=Modality.VIDEO,
                offsets=video_offsets,
                precomputed_embeddings=video_embeddings,
            ),
        ]

        if has_audio:
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    offsets=audio_offsets,
                    precomputed_embeddings=audio_embeddings,
                )
            )

        result = {
            "input_ids": input_ids,
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.mimo_processor.image_token_id,
            "video_token_id": self.mimo_processor.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

        if has_audio:
            result["audio_token_id"] = self.mimo_processor.audio_token_id
            result["audio_start_id"] = self.AUDIO_START_TOKEN_ID
            result["audio_end_id"] = self.AUDIO_END_TOKEN_ID

        return result

    def _preprocess_video_sync(self, vdw, preprocess_kwargs=None):
        """Sample frames from a pre-loaded VideoDecoderWrapper.

        Returns (video_tensor, timestamps) tuple suitable for VideoInput.
        video_tensor: TCHW float Tensor
        timestamps: 1D float Tensor of per-frame timestamps in seconds
        """
        ele = preprocess_kwargs or {}
        total_frames, video_fps = len(vdw), vdw.avg_fps
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        idx = list(
            np.unique(np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64))
        )
        try:
            video_tensor = vdw.get_frames_as_tensor(idx)  # NHWC uint8 Tensor
        except Exception as e:
            logger.error(f"Video decode failed in _preprocess_video_sync: {e}")
            raise HTTPException(
                status_code=432, detail="Video file is corrupted or cannot be decoded"
            )
        video_tensor = video_tensor.permute(0, 3, 1, 2).float()  # NHWC → TCHW float
        timestamps = torch.as_tensor(idx, dtype=torch.float32) / video_fps
        return (video_tensor, timestamps)

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        if audios and not self.AUDIO_TOKEN_REGEX.search(input_text or ""):
            input_text = f"{self.mm_tokens.audio_token}{input_text or ''}"

        # Preprocess all multimodal items first
        processed_images = []
        processed_videos = []  # List of (raw_video_source, use_audio, audio_source, preprocess_kwargs)
        processed_audios = []

        if images:
            processed_images = list(images)

        if videos:
            for video in videos:
                preprocess_kwargs = {}
                audio_source = None
                raw_video_source = video
                if isinstance(video, VideoData):
                    preprocess_kwargs = getattr(video, "preprocess_kwargs", {}) or {}
                    raw_video_source = video.url
                    audio_source = video.url
                    video = video.url
                elif isinstance(video, dict):
                    preprocess_kwargs = video.get("preprocess_kwargs", {}) or {}
                    audio_source = video.get("audio") or video.get("url")
                    video = video.get("url", video)
                    raw_video_source = video
                elif isinstance(video, str):
                    raw_video_source = video
                    audio_source = None

                # Determine use_audio: check preprocess_kwargs first, then auto-detect
                if "use_audio" in preprocess_kwargs:
                    use_audio = preprocess_kwargs["use_audio"]
                elif isinstance(raw_video_source, str):
                    # Auto-detect audio track if not specified
                    use_audio = has_audio_track(raw_video_source)
                elif isinstance(video, VideoData):
                    use_audio = has_audio_track(raw_video_source.url)
                else:
                    use_audio = False

                if (
                    use_audio
                    and audio_source is None
                    and isinstance(raw_video_source, (str, bytes, torch.Tensor))
                ):
                    audio_source = raw_video_source

                processed_videos.append(
                    (raw_video_source, use_audio, audio_source, preprocess_kwargs)
                )

        if audios:
            for audio in audios:
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio).float()
                elif isinstance(audio, torch.Tensor):
                    audio_tensor = audio.float()
                else:
                    processed_audios.append(audio)
                    continue
                if audio_tensor.ndim == 1:
                    processed_audios.append(
                        (audio_tensor.cpu().contiguous(), self.audio_sample_rate)
                    )
                else:
                    processed_audios.append(audio_tensor.cpu().contiguous())

        # Build contents with text parts interleaved with multimodal items
        contents = []

        # If we have input_text, parse it to interleave text and multimodal content
        if input_text and (processed_images or processed_videos or processed_audios):
            multimodal_tokens_pattern = self.mm_tokens.get_combined_regex()
            text_parts = re.split(multimodal_tokens_pattern, input_text)

            image_iter = iter(processed_images)
            video_iter = iter(processed_videos)
            audio_iter = iter(processed_audios)

            for text_part in text_parts:
                if multimodal_tokens_pattern.match(text_part):
                    modality = self.mm_tokens.get_modality_of_token(text_part)
                    if modality == Modality.IMAGE:
                        try:
                            img = next(image_iter)
                            contents.append(
                                Content(type="image", content=ImageInput(image=img))
                            )
                        except StopIteration:
                            pass
                    elif modality == Modality.VIDEO:
                        try:
                            (
                                processed_video,
                                use_audio,
                                audio_source,
                                preprocess_kwargs,
                            ) = next(video_iter)
                            if use_audio:
                                contents.append(
                                    Content(
                                        type="video_audio",
                                        content=VideoAudioInput(
                                            video=processed_video,
                                            audio=audio_source,
                                            min_pixels=preprocess_kwargs.get(
                                                "min_pixels", None
                                            ),
                                            max_pixels=preprocess_kwargs.get(
                                                "max_pixels", None
                                            ),
                                            total_max_pixels=preprocess_kwargs.get(
                                                "total_max_pixels", None
                                            ),
                                            fps=preprocess_kwargs.get("fps", None),
                                            num_frames=preprocess_kwargs.get(
                                                "num_frames", None
                                            ),
                                            max_frames=preprocess_kwargs.get(
                                                "max_frames", None
                                            ),
                                            min_frames=preprocess_kwargs.get(
                                                "min_frames", None
                                            ),
                                        ),
                                    )
                                )
                            else:
                                contents.append(
                                    Content(
                                        type="video",
                                        content=VideoInput(
                                            video=processed_video,
                                            min_pixels=preprocess_kwargs.get(
                                                "min_pixels", None
                                            ),
                                            max_pixels=preprocess_kwargs.get(
                                                "max_pixels", None
                                            ),
                                            total_max_pixels=preprocess_kwargs.get(
                                                "total_max_pixels", None
                                            ),
                                            fps=preprocess_kwargs.get("fps", None),
                                            num_frames=preprocess_kwargs.get(
                                                "num_frames", None
                                            ),
                                            max_frames=preprocess_kwargs.get(
                                                "max_frames", None
                                            ),
                                            min_frames=preprocess_kwargs.get(
                                                "min_frames", None
                                            ),
                                        ),
                                    )
                                )
                        except StopIteration:
                            pass
                    elif modality == Modality.AUDIO:
                        try:
                            audio = next(audio_iter)
                            contents.append(
                                Content(type="audio", content=AudioInput(audio=audio))
                            )
                        except StopIteration:
                            pass
                else:
                    # Add text content
                    if text_part:
                        contents.append(Content(type="text", content=text_part))
        else:
            # Fallback: just add multimodal contents without text interleaving
            contents.extend(
                Content(type="image", content=ImageInput(image=image))
                for image in processed_images
            )
            for (
                processed_video,
                use_audio,
                audio_source,
                preprocess_kwargs,
            ) in processed_videos:
                if use_audio:
                    contents.append(
                        Content(
                            type="video_audio",
                            content=VideoAudioInput(
                                video=processed_video,
                                audio=audio_source,
                                min_pixels=preprocess_kwargs.get("min_pixels", None),
                                max_pixels=preprocess_kwargs.get("max_pixels", None),
                                total_max_pixels=preprocess_kwargs.get(
                                    "total_max_pixels", None
                                ),
                                fps=preprocess_kwargs.get("fps", None),
                                num_frames=preprocess_kwargs.get("num_frames", None),
                                max_frames=preprocess_kwargs.get("max_frames", None),
                                min_frames=preprocess_kwargs.get("min_frames", None),
                            ),
                        )
                    )
                else:
                    contents.append(
                        Content(
                            type="video",
                            content=VideoInput(
                                video=processed_video,
                                min_pixels=preprocess_kwargs.get("min_pixels", None),
                                max_pixels=preprocess_kwargs.get("max_pixels", None),
                                total_max_pixels=preprocess_kwargs.get(
                                    "total_max_pixels", None
                                ),
                                fps=preprocess_kwargs.get("fps", None),
                                num_frames=preprocess_kwargs.get("num_frames", None),
                                max_frames=preprocess_kwargs.get("max_frames", None),
                                min_frames=preprocess_kwargs.get("min_frames", None),
                            ),
                        )
                    )
            contents.extend(
                Content(type="audio", content=AudioInput(audio=audio))
                for audio in processed_audios
            )

        if not contents:
            input_ids = self.mimo_processor.tokenizer(
                input_text or "",
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids
            return {"input_ids": input_ids}

        input_sample = self.mimo_processor.process(contents, verbose=False)

        ret = {
            "input_ids": input_sample.input_ids,
            "mrope_positions": getattr(input_sample, "position_ids", None),
            "mrope_position_delta": getattr(input_sample, "rope_deltas", None),
        }
        if getattr(input_sample, "pixel_values", None):
            pixel_values = torch.cat(input_sample.pixel_values, dim=0)
            image_grids = torch.stack(input_sample.image_thw_grids)
            ret.update(
                {
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grids,
                }
            )
        if getattr(input_sample, "pixel_values_videos", None):
            pixel_values_videos = torch.cat(input_sample.pixel_values_videos, dim=0)
            video_grids = torch.stack(input_sample.video_thw_grids)
            ret.update(
                {
                    "pixel_values_videos": pixel_values_videos,
                    "video_grid_thw": video_grids,
                }
            )
            # Add second_per_grid_ts for video temporal information (EPD alignment)
            second_per_grid_ts = getattr(input_sample, "second_per_grid_ts", None)
            if second_per_grid_ts is None:
                second_per_grid_ts = getattr(
                    input_sample, "video_second_per_grid", None
                )
            if second_per_grid_ts is not None:
                ret["second_per_grid_ts"] = second_per_grid_ts
            # Add video_start_token_id and video_end_token_id for EPD alignment
            ret["video_start_token_id"] = getattr(
                self.mimo_processor, "video_start_token_id", None
            )
            ret["video_end_token_id"] = getattr(
                self.mimo_processor, "video_end_token_id", None
            )
        audio_inputs = getattr(input_sample, "audio_inputs", None)
        if audio_inputs is not None and len(audio_inputs) > 0:
            ret["audio_features"] = audio_inputs
            audio_attention_mask = getattr(
                input_sample, "audio_attention_mask", None
            ) or getattr(input_sample, "feature_attention_mask", None)
            if audio_attention_mask is not None:
                ret["audio_attention_mask"] = audio_attention_mask
            audio_feature_lens = getattr(input_sample, "audio_feature_lens", None)
            if audio_feature_lens is None:
                audio_feature_lens = audio_attention_mask
                if audio_feature_lens is not None:
                    audio_feature_lens = audio_feature_lens.sum(dim=-1)
            if audio_feature_lens is not None:
                ret["audio_feature_lens"] = audio_feature_lens

        device = kwargs.get("device")
        if device:
            for key in (
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
                "audio_features",
                "audio_feature_lens",
            ):
                if key in ret and isinstance(ret[key], torch.Tensor):
                    ret[key] = ret[key].to(device)

        return ret

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if audio_data is None:
            audio_data = getattr(request_obj, "audio_data", [])
        if audio_data and not self.AUDIO_TOKEN_REGEX.search(input_text):
            input_text = f"{self.mm_tokens.audio_token}{input_text}"

        video_data = getattr(request_obj, "video_data", [])
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.audio_sample_rate,
        )
        multimodal_tokens_pattern = self.mm_tokens.get_combined_regex()

        # Reconstruct contents from base_output and request_obj metadata
        raw_image_data = image_data or []
        raw_video_data = getattr(request_obj, "video_data", None) or []
        raw_audio_data = audio_data or []

        loaded_image_iter = iter(base_output.images)
        loaded_video_iter = iter(base_output.videos)
        loaded_audio_iter = iter(base_output.audios)

        raw_image_iter = iter(raw_image_data)
        raw_video_iter = iter(raw_video_data)
        raw_audio_iter = iter(raw_audio_data)

        # base_output.input_text may contain replaced tokens, but we split by regex anyway
        text_parts = re.split(multimodal_tokens_pattern, base_output.input_text)
        contents = []

        for text_part in text_parts:
            if multimodal_tokens_pattern.match(text_part):
                modality = self.mm_tokens.get_modality_of_token(text_part)
                assert modality is not None

                if modality == Modality.IMAGE:
                    loaded_img = next(loaded_image_iter)
                    raw_img_item = next(raw_image_iter)

                    preprocess_kwargs = {}
                    if isinstance(raw_img_item, ImageData):
                        preprocess_kwargs = (
                            getattr(raw_img_item, "preprocess_kwargs", {}) or {}
                        )

                    contents.append(
                        Content(
                            type="image",
                            content=ImageInput(
                                image=loaded_img,
                                min_pixels=preprocess_kwargs.get("min_pixels", None),
                                max_pixels=preprocess_kwargs.get("max_pixels", None),
                            ),
                        )
                    )
                elif modality == Modality.VIDEO:
                    loaded_video = next(loaded_video_iter)
                    raw_video_item = next(raw_video_iter)

                    preprocess_kwargs = {}
                    raw_video_item_audio = None
                    use_audio = False
                    if isinstance(raw_video_item, VideoData):
                        preprocess_kwargs = (
                            getattr(raw_video_item, "preprocess_kwargs", {}) or {}
                        )
                        use_audio = has_audio_track(raw_video_item.url)
                        raw_video_item_audio = raw_video_item.url
                    elif isinstance(raw_video_item, dict):
                        use_audio = has_audio_track(
                            raw_video_item.get("url", raw_video_item)
                        )
                        raw_video_item_audio = raw_video_item
                    elif isinstance(raw_video_item, str):
                        use_audio = has_audio_track(raw_video_item)
                        raw_video_item_audio = raw_video_item

                    # Pre-process video: VideoDecoderWrapper → (TCHW tensor, timestamps) tuple
                    video_tuple = self._preprocess_video_sync(
                        loaded_video, preprocess_kwargs
                    )

                    if use_audio:
                        contents.append(
                            Content(
                                type="video_audio",
                                content=VideoAudioInput(
                                    video=video_tuple,
                                    audio=raw_video_item_audio,
                                    min_pixels=preprocess_kwargs.get(
                                        "min_pixels", None
                                    ),
                                    max_pixels=preprocess_kwargs.get(
                                        "max_pixels", None
                                    ),
                                    total_max_pixels=preprocess_kwargs.get(
                                        "total_max_pixels", None
                                    ),
                                    fps=preprocess_kwargs.get("fps", None),
                                    num_frames=preprocess_kwargs.get(
                                        "num_frames", None
                                    ),
                                    max_frames=preprocess_kwargs.get(
                                        "max_frames", None
                                    ),
                                    min_frames=preprocess_kwargs.get(
                                        "min_frames", None
                                    ),
                                ),
                            )
                        )
                    else:
                        contents.append(
                            Content(
                                type="video",
                                content=VideoInput(
                                    video=video_tuple,
                                    min_pixels=preprocess_kwargs.get(
                                        "min_pixels", None
                                    ),
                                    max_pixels=preprocess_kwargs.get(
                                        "max_pixels", None
                                    ),
                                    total_max_pixels=preprocess_kwargs.get(
                                        "total_max_pixels", None
                                    ),
                                    fps=preprocess_kwargs.get("fps", None),
                                    num_frames=preprocess_kwargs.get(
                                        "num_frames", None
                                    ),
                                    max_frames=preprocess_kwargs.get(
                                        "max_frames", None
                                    ),
                                    min_frames=preprocess_kwargs.get(
                                        "min_frames", None
                                    ),
                                ),
                            )
                        )
                elif modality == Modality.AUDIO:
                    loaded_audio = next(loaded_audio_iter)
                    raw_audio_item = next(raw_audio_iter)

                    if isinstance(loaded_audio, np.ndarray):
                        audio_source = loaded_audio
                    elif isinstance(raw_audio_item, dict):
                        audio_source = raw_audio_item.get("url", loaded_audio)
                    elif isinstance(raw_audio_item, (str, bytes, torch.Tensor)):
                        audio_source = raw_audio_item

                    contents.append(
                        Content(
                            type="audio",
                            content=AudioInput(
                                audio=audio_source,
                            ),
                        )
                    )
            else:
                if text_part:
                    contents.append(Content(type="text", content=text_part))

        # Run MiMo processor
        loop = asyncio.get_running_loop()
        try:
            input_sample = await loop.run_in_executor(
                self.io_executor,
                lambda: self.mimo_processor.process(contents, verbose=False),
            )
        except RuntimeError as e:
            logger.error(f"MiMo processor failed in process_mm_data_async: {e}")
            raise ValueError(f"Multimodal data is corrupted or cannot be decoded: {e}")

        input_ids = input_sample.input_ids.flatten()
        mm_items: list[MultimodalDataItem] = []
        if len(input_sample.image_thw_grids) > 0:
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=torch.cat([v.cpu() for v in input_sample.pixel_values], dim=0),
                    model_specific_data={
                        "image_grid_thw": torch.stack(input_sample.image_thw_grids)
                    },
                    offsets=self.get_mm_items_offset(
                        input_ids=input_ids,
                        mm_token_id=self.mimo_processor.image_token_id,
                    ),
                )
            )
        if len(input_sample.video_thw_grids) > 0:
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.VIDEO,
                    feature=torch.cat([v.cpu() for v in input_sample.pixel_values_videos], dim=0),
                    model_specific_data={
                        "video_grid_thw": torch.stack(input_sample.video_thw_grids)
                    },
                    offsets=self.get_mm_items_offset(
                        input_ids=input_ids,
                        mm_token_id=self.mimo_processor.video_token_id,
                    ),
                )
            )
        audio_inputs = getattr(input_sample, "audio_inputs", None)
        if audio_inputs is not None and len(audio_inputs) > 0:
            audio_item = MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=audio_inputs,
                offsets=self.get_mm_items_offset(
                    input_ids=input_ids, mm_token_id=self.mimo_processor.audio_token_id
                ),
            )
            audio_feature_lens = getattr(input_sample, "audio_feature_lens", None)
            if audio_feature_lens is None:
                audio_attention_mask = getattr(
                    input_sample, "audio_attention_mask", None
                ) or getattr(input_sample, "feature_attention_mask", None)
                if audio_attention_mask is not None:
                    audio_feature_lens = audio_attention_mask.sum(dim=-1)
            if audio_feature_lens is not None:
                audio_item.audio_feature_lens = audio_feature_lens
            mm_items.append(audio_item)

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.mimo_processor.image_token_id,
            video_token_id=self.mimo_processor.video_token_id,
            audio_token_id=self.mimo_processor.audio_token_id,
            audio_start_id=self.AUDIO_START_TOKEN_ID,
            audio_end_id=self.AUDIO_END_TOKEN_ID,
            mrope_positions=input_sample.position_ids,
            mrope_position_delta=input_sample.rope_deltas,
        )
