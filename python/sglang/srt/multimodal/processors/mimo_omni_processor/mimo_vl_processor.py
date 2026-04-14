"""
audio

(T, n_vq+)
(T, n_vq+) -> (padded_T, n_vq+) -> (padded_T // group_size, group_size, n_vq)

audio_group_size, audio_channels, audio_input_id_per_second
"""

import os
import io
import json
import logging
from collections import OrderedDict

import pybase64
import requests
import shutil
import time
import math
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from .mimo_vl_utils import (
    fetch_image,
    get_visual_transform,
    get_visual_transform_batch,
    MIMO_PLACEHOLDER,
    format_timestamp,
)
from .protocol import (
    TextInput, ImageInput, AudioInput, VideoInput, VideoAudioInput,
    Content, MessageTurn, Conversation,
    MiMoVLInputSample
)

from torchcodec.decoders import AudioDecoder
try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram
except ImportError:
    print("[Warning] torchaudio is not installed, audio inference will not be supported")
    torchaudio = None
    MelSpectrogram = None

logger = logging.getLogger(__name__)


class TorchCodecVideoReader:
    """Wrapper around torchcodec VideoDecoder with a decord-like interface."""

    _PARALLEL_FRAME_THRESHOLD = int(os.environ.get("SGLANG_VIDEO_DECODE_THRESHOLD", "8"))
    _thread_pool = ThreadPoolExecutor(max_workers=min(_PARALLEL_FRAME_THRESHOLD, os.cpu_count()))

    def __init__(self, source):
        from torchcodec.decoders import VideoDecoder

        self._source = source
        self._decoder = VideoDecoder(source, seek_mode="approximate")
        self._metadata = self._decoder.metadata

    def __len__(self):
        return self._metadata.num_frames

    def get_avg_fps(self):
        return self._metadata.average_fps

    @property
    def height(self):
        return self._metadata.height

    @property
    def width(self):
        return self._metadata.width

    def get_batch(self, indices):
        from torchcodec.decoders import VideoDecoder

        indices = list(indices)
        if len(indices) == 0:
            return torch.empty(0, 3, self.height, self.width)

        pool = self._thread_pool
        chunk_size = max(1, len(indices) // pool._max_workers)
        chunks = [
            indices[i : i + chunk_size]
            for i in range(0, len(indices), chunk_size)
        ]
        source = self._source

        def _decode_chunk(chunk):
            d = VideoDecoder(source, seek_mode="approximate")
            return d.get_frames_at(chunk).data  # (N, C, H, W) uint8

        futures = [pool.submit(_decode_chunk, chunk) for chunk in chunks]
        parts = [f.result() for f in futures]
        return torch.cat(parts, dim=0).float()  # (N, C, H, W)


def _load_video_torchcodec(video_file):
    """Load video using torchcodec. Supports file path, URL, base64, and bytes."""
    if isinstance(video_file, bytes):
        source = video_file
    elif isinstance(video_file, str):
        if video_file.startswith(("http://", "https://")):
            timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
            response = requests.get(video_file, timeout=timeout)
            response.raise_for_status()
            source = response.content
        elif video_file.startswith("data:"):
            _, encoded = video_file.split(",", 1)
            source = pybase64.b64decode(encoded, validate=True)
        elif os.path.isfile(video_file):
            source = video_file
        else:
            source = pybase64.b64decode(video_file, validate=True)
    else:
        raise ValueError(f"Unsupported video input type: {type(video_file)}")

    return TorchCodecVideoReader(source)

from transformers import Qwen2TokenizerFast
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

        def smart_nframes(fps_ori, total_frames_ori, frame_factor, num_frames=None, fps=None, max_frames=None, min_frames=None, **kwargs):
            if num_frames is not None:
                nframes = num_frames
            elif fps is not None:
                nframes = math.ceil(len(vr) / fps_ori * fps)
                if max_frames is not None:
                    nframes = min(nframes, max_frames)
                if min_frames is not None:
                    nframes = max(nframes, min_frames)
            if nframes > total_frames_ori:
                nframes = total_frames_ori
            nframes = math.ceil(nframes / frame_factor) * frame_factor
            if nframes == 0:
                nframes = frame_factor
            return nframes

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

        # TODO: currently always sample a multiple of frame_factor frames, so no padding is needed
        frame_factor = self.temporal_patch_size * self.temporal_compression_ratio

        if isinstance(video, (str, bytes)):
            vr = _load_video_torchcodec(video)

            fps_ori = vr.get_avg_fps()
            total_frames_ori = len(vr)
            duration_ori = total_frames_ori / fps_ori

            start_time = video_input.start_time if video_input.start_time is not None else 0
            end_time = video_input.end_time if video_input.end_time is not None else duration_ori
            duration_seg = end_time - start_time
            total_frames_seg = duration_seg * fps_ori

            if video_input.segment_type == "individual":
                num_frames_sampled = smart_nframes(fps_ori, total_frames_seg, frame_factor=frame_factor, **kwargs)
                fps_sampled = num_frames_sampled / duration_seg
                start_frame_idx, end_frame_idx = int(start_time * fps_ori), int(end_time * fps_ori)
            else:
                num_frames_sampled = smart_nframes(fps_ori, total_frames_ori, frame_factor=frame_factor, **kwargs)
                fps_sampled = num_frames_sampled / duration_ori
                start_frame_idx, end_frame_idx = 0, len(vr)

            if video_input.do_include_last_frame:
                frame_idx_sampled = torch.linspace(start_frame_idx, end_frame_idx - 1, num_frames_sampled).round().int()
            else:
                frame_idx_sampled = torch.linspace(start_frame_idx, end_frame_idx, num_frames_sampled + 1).round().int()[:-1]
            frame_idx_sampled = frame_idx_sampled.clamp(max=len(vr)-1)
            timestamps_sampled = frame_idx_sampled / fps_ori

            if video_input.segment_type == "individual":
                frame_idx_seg = frame_idx_sampled
                timestamps_seg = timestamps_sampled
                num_frames_seg = num_frames_sampled
                start_time_seg = start_time
                end_time_seg = end_time
            else:
                # Use segment_frame_selector to select frames aligned to frame_factor
                frame_idx_seg = segment_frame_selector(timestamps_sampled, start_time, end_time)

                timestamps_seg = timestamps_sampled[frame_idx_seg]
                num_frames_seg = len(timestamps_seg)
                start_time_seg = timestamps_seg[0].item()
                end_time_seg = timestamps_seg[-1].item() + (1 / fps_sampled)

            assert num_frames_seg == len(frame_idx_seg)
            assert num_frames_seg == timestamps_seg.shape[0]

            frames = vr.get_batch(frame_idx_seg.tolist())  # (N, C, H, W) float

            video_meta = {
                "fps_sampled": fps_sampled,
                "segment_start_time": start_time_seg,
                "segment_end_time": end_time_seg,
            }

        else:
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




if __name__ == "__main__":
    pass

