import asyncio
import base64
import json
import os
import re
import subprocess
from typing import List, Union

import numpy as np
import torch
from fastapi import HTTPException
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
from sglang.srt.multimodal.processors.mimo_omni_processor import (
    AudioInput,
    Content,
    ImageInput,
    MiMoVLProcessor,
    VideoAudioInput,
    VideoInput,
)
from sglang.srt.multimodal.processors.qwen_vl import smart_nframes
from sglang.srt.utils import ImageData, VideoData
from sglang.utils import logger
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)

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
