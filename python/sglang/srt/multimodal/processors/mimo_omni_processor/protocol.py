from typing import Literal, Optional, Tuple, Union
from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import torch

try:
    from decord import VideoReader
except ImportError:
    VideoReader = None


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