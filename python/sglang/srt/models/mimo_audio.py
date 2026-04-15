import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.mimo_audio_tokenizer import MiMoAudioTokenizer
from sglang.srt.models.mimo_audio_utils import tokenize_audio_batch
from sglang.srt.server_args import get_global_server_args
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

logger = logging.getLogger(__name__)


@dataclass
class MimoAudioEncoderConfig:
    tokenizer_version: str = "v1"
    speech_vocab_size: str = "1025-1025-129-129-129-129-129-129"
    speech_zeroemb_idx: str = "1024-1024-128-128-128-128-128-128"
    group_size: int = 4
    audio_channels: int = 8
    input_local_layers: int = 6
    input_local_dim: int = 1024
    input_full_attention: bool = True
    input_local_attn_heads: int = 64
    input_local_head_dim: int = 16
    input_local_intermediate_size: int = 4096
    input_local_hidden_dropout: float = 0.0
    out_hidden_size: int = 4096  # mimo vl hidden dim
    rope_theta: float = 640000.0
    partial_rotary_factor: float = 0.334
    projection_layers: int = 1
    add_post_norm: bool = False
    audio_segment_size: int = 6000


class AudioProjection(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MiMoV2AudioConfig:
    def __init__(
        self,
        speech_vocab_size: str | int = "1280",
        speech_lm_head_sizes: str | int | None = None,
        speech_zeroemb_idx: str | int = "1280",
        delay_pattern: str = "0-1-2-3-4-5-6-7-7-7-7-7-7-7-7-7-7-7-7-7",
        group_size: int = 4,
        audio_channels: int = 20,
        input_local_dim: int = 1024,
        input_local_layers: int = 6,
        input_local_attn_heads: int = 16,
        input_local_intermediate_size: int = 4096,
        input_local_rope_theta: float = 640000.0,
        input_local_partial_rotary_factor: float = 1.0,
        output_local_dim: int = 1024,
        output_local_layers: int = 6,
        output_local_attn_heads: int = 16,
        output_local_intermediate_size: int = 4096,
        output_local_rope_theta: float = 640000.0,
        output_local_partial_rotary_factor: float = 1.0,
        input_projection_layers: int = 2,
        output_projection_layers: int = 2,
        add_encoder_post_norm: bool = True,
        # add_decoder_post_norm: bool = True,
        audio_config: dict = None,
        **kwargs,
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if audio_config is not None:
            self._load_from_audio_config(audio_config)
        else:
            self.speech_vocab_size = speech_vocab_size
            self.speech_lm_head_sizes = (
                speech_lm_head_sizes
                if speech_lm_head_sizes is not None
                else speech_vocab_size
            )
            self.speech_zeroemb_idx = speech_zeroemb_idx
            self.delay_pattern = delay_pattern
            self.group_size = group_size
            self.audio_channels = audio_channels
            self.input_local_dim = input_local_dim
            self.input_local_layers = input_local_layers
            self.input_local_attn_heads = input_local_attn_heads
            self.input_local_intermediate_size = input_local_intermediate_size
            self.input_local_rope_theta = input_local_rope_theta
            self.input_local_partial_rotary_factor = input_local_partial_rotary_factor
            self.output_local_dim = output_local_dim
            self.output_local_layers = output_local_layers
            self.output_local_attn_heads = output_local_attn_heads
            self.output_local_intermediate_size = output_local_intermediate_size
            self.output_local_rope_theta = output_local_rope_theta
            self.output_local_partial_rotary_factor = output_local_partial_rotary_factor
            self.input_projection_layers = input_projection_layers
            self.output_projection_layers = output_projection_layers
            self.add_encoder_post_norm = add_encoder_post_norm
            # self.add_decoder_post_norm = add_decoder_post_norm

        self._attn_implementation_internal = "sdpa"

    def _load_from_audio_config(self, audio_config: dict):
        """Load audio parameters from audio_config dict in checkpoint.

        Uses naming that matches megatron2hf conversion output to minimize manual mapping.
        """
        self.group_size = audio_config.get("group_size", 4)
        self.audio_channels = audio_config.get("audio_channels", 20)
        self.speech_vocab_size = audio_config.get("speech_vocab_size", "1280")
        self.speech_lm_head_sizes = audio_config.get(
            "speech_lm_head_sizes", self.speech_vocab_size
        )
        self.speech_zeroemb_idx = audio_config.get("speech_zeroemb_idx", "1280")
        self.delay_pattern = audio_config.get(
            "audio_output_delay_pattern", "0-1-2-3-4-5-6-7-7-7-7-7-7-7-7-7-7-7-7-7"
        )

        self.input_local_dim = audio_config.get("input_local_dim", 1024)
        self.input_local_layers = audio_config.get("input_local_layers", 6)
        self.input_local_attn_heads = audio_config.get("input_local_attn_heads", 16)
        self.input_local_intermediate_size = audio_config.get(
            "input_local_intermediate_size", 4096
        )
        self.input_local_rope_theta = audio_config.get(
            "input_local_rope_theta", 640000.0
        )
        self.input_local_partial_rotary_factor = audio_config.get(
            "input_local_partial_rotary_factor", 1.0
        )

        self.output_local_dim = audio_config.get("output_local_dim", 1024)
        self.output_local_layers = audio_config.get("output_local_layers", 6)
        self.output_local_attn_heads = audio_config.get("output_local_attn_heads", 16)
        self.output_local_intermediate_size = audio_config.get(
            "output_local_intermediate_size", 4096
        )
        self.output_local_rope_theta = audio_config.get(
            "output_local_rope_theta", 640000.0
        )
        self.output_local_partial_rotary_factor = audio_config.get(
            "output_local_partial_rotary_factor", 1.0
        )

        self.input_projection_layers = audio_config.get("input_projection_layers", 2)
        self.output_projection_layers = audio_config.get("output_projection_layers", 2)

        self.add_encoder_post_norm = audio_config.get("add_encoder_post_norm", True)
        # self.add_decoder_post_norm = audio_config.get("add_decoder_post_norm", True)

    def _parse_maybe_list(self, value: str | int, length: int) -> list[int]:
        if isinstance(value, str) and "-" in value:
            return [int(s) for s in value.split("-")]
        return [int(value)] * length

    def parsed_speech_empty_ids(self):
        return self._parse_maybe_list(self.speech_zeroemb_idx, self.audio_channels)

    def parsed_speech_vocab_sizes(self):
        return self._parse_maybe_list(self.speech_vocab_size, self.audio_channels)

    def parsed_speech_lm_head_sizes(self):
        return self._parse_maybe_list(self.speech_lm_head_sizes, self.audio_channels)

    def parsed_delay_pattern(self):
        return self._parse_maybe_list(self.delay_pattern, self.audio_channels)

    def input_local_config(self):
        """Create config for input local transformer."""
        config = Qwen2Config()
        for attr in dir(self):
            if not attr.startswith("_") and hasattr(config, attr):
                setattr(config, attr, getattr(self, attr))

        config.hidden_size = self.input_local_dim
        config.num_hidden_layers = self.input_local_layers
        config.num_attention_heads = self.input_local_attn_heads
        config.num_key_value_heads = self.input_local_attn_heads
        config.intermediate_size = self.input_local_intermediate_size
        config.rope_theta = self.input_local_rope_theta
        config.partial_rotary_factor = self.input_local_partial_rotary_factor
        config._attn_implementation_internal = "sdpa"

        return config

    def output_local_config(self):
        """Create config for output local transformer."""
        config = Qwen2Config()
        for attr in dir(self):
            if not attr.startswith("_") and hasattr(config, attr):
                setattr(config, attr, getattr(self, attr))

        config.hidden_size = self.output_local_dim
        config.num_hidden_layers = self.output_local_layers
        config.num_attention_heads = self.output_local_attn_heads
        config.num_key_value_heads = self.output_local_attn_heads
        config.intermediate_size = self.output_local_intermediate_size
        config.rope_theta = self.output_local_rope_theta
        config.partial_rotary_factor = self.output_local_partial_rotary_factor
        config._attn_implementation_internal = "sdpa"

        return config


class MimoAudioEncoder(nn.Module):
    config: MimoAudioEncoderConfig

    def __init__(self, config):
        super().__init__()
        if not isinstance(config, MiMoV2AudioConfig):
            config_dict = (
                vars(config) if hasattr(config, "__dict__") else config.__dict__
            )
            config = MiMoV2AudioConfig(**config_dict)
        self.config = config
        self.server_args = get_global_server_args()
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        self.speech_empty_ids = self.parsed_speech_empty_ids()
        self.audio_channels = config.audio_channels
        self.audio_group_size = config.group_size
        self.audio_segment_size = config.audio_segment_size
        speech_vocab_size = self._parse_maybe_list(
            self.config.speech_vocab_size, self.config.audio_channels
        )
        input_local_config = Qwen2Config(
            hidden_size=self.config.input_local_dim,
            num_hidden_layers=self.config.input_local_layers,
            num_attention_heads=self.config.input_local_attn_heads,
            num_key_value_heads=self.config.input_local_attn_heads,
            intermediate_size=self.config.input_local_intermediate_size,
            attention_dropout=self.config.input_local_hidden_dropout,
            rope_theta=self.config.rope_theta,
            partial_rotary_factor=self.config.partial_rotary_factor,
        )

        self.input_local_transformer = Qwen2Model(input_local_config)

        if not self.config.add_post_norm:
            self.input_local_transformer.norm = nn.Identity()

        self.speech_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    speech_vocab_size[i],
                    self.config.input_local_dim,
                    padding_idx=self.speech_empty_ids[i],
                )
                for i in range(self.config.audio_channels)
            ]
        )

        if self.config.projection_layers == 1:
            self.projection = nn.Linear(
                self.config.input_local_dim * self.config.group_size,
                self.config.out_hidden_size,
                bias=False,
            )
        elif self.config.projection_layers == 2:
            self.projection = AudioProjection(
                self.config.input_local_dim * self.config.group_size,
                self.config.input_local_dim * self.config.group_size * 4,
                self.config.out_hidden_size,
            )
        else:
            raise ValueError(
                f"Invalid projection layers: {self.config.projection_layers}"
            )

        audio_tokenizer_path = os.path.join(
            self.server_args.model_path, "audio_tokenizer"
        )
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.audio_tokenizer = self._load_audio_tokenizer(
            audio_tokenizer_path, dev
        )

    @staticmethod
    def _load_audio_tokenizer(
        path: str, device: torch.device
    ) -> MiMoAudioTokenizer:
        """Load MiMoAudioTokenizer manually to avoid new-transformers compat issues."""
        import json
        import os
        from safetensors.torch import load_file

        config_path = os.path.join(path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        config = MiMoAudioTokenizer.config_class(**config_dict)
        model = MiMoAudioTokenizer(config)
        # Load weights from safetensors or pytorch bin
        safetensors_path = os.path.join(path, "model.safetensors")
        bin_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No model weights found in {path} "
                "(expected model.safetensors or pytorch_model.bin)"
            )
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        model.requires_grad_(False)
        return model

    def parsed_speech_empty_ids(self):
        return self._parse_maybe_list(
            self.config.speech_zeroemb_idx, self.config.audio_channels
        )

    def _parse_maybe_list(self, value: str | int, length: int) -> List[int]:
        if isinstance(value, str) and "-" in value:
            return [int(s) for s in value.split("-")]
        return [int(value)] * length

    # adapted from mimo-audio
    def apply_input_local_transformer(self, speech_embeddings: torch.Tensor):
        output = self.input_local_transformer(
            inputs_embeds=speech_embeddings,
            return_dict=True,
            is_causal=not self.config.input_full_attention,  # for SDPA
        )
        return output.last_hidden_state  # [T//group_size,  group_size, hidden_size]

    def apply_speech_embeddings(self, audio_codes: torch.Tensor) -> torch.Tensor:
        num_segments = audio_codes.shape[0]
        _audio_embeddings = torch.zeros(
            (num_segments, self.config.group_size, self.config.input_local_dim),
            dtype=next(self.speech_embeddings[0].parameters()).dtype,
            device=audio_codes.device,
        )
        for i in range(self.config.audio_channels):
            _audio_embeddings.add_(self.speech_embeddings[i](audio_codes[:, :, i]))
        return _audio_embeddings

    def process_audio(self, audio):
        T = audio.shape[0]
        audio = audio[:, : self.audio_channels]
        padded_T = (
            (T + self.audio_group_size - 1)
            // self.audio_group_size
            * self.audio_group_size
        )
        padded_audio = torch.cat(
            [
                audio,
                torch.zeros(
                    padded_T - T,
                    self.audio_channels,
                    dtype=torch.int32,
                    device=audio.device,
                )
                + audio[-1, :],
            ],
            dim=0,
        )  # pad using the last embedding
        padded_audio = padded_audio.reshape(
            padded_T // self.audio_group_size,
            self.audio_group_size,
            self.audio_channels,
        )
        return padded_audio

    def get_audio_feature(self, items) -> torch.Tensor:
        # items: already audio-only MultimodalDataItem list from caller.
        # Each item.feature is either one mel tensor or a list of mel tensors (e.g. long audio split into chunks).
        all_mels = []
        for item in items:
            f = item.feature
            if isinstance(f, (list, tuple)):
                all_mels.extend(f)
            else:
                all_mels.append(f)
        if not all_mels:
            device = next(self.projection.parameters()).device
            dtype = next(self.projection.parameters()).dtype
            return torch.empty(
                0, self.config.out_hidden_size, device=device, dtype=dtype
            )
        # Batch tokenize: one encode_batch call for all mels
        device = next(self.audio_tokenizer.encoder.parameters()).device
        code_list = tokenize_audio_batch(
            all_mels,
            self.audio_tokenizer.encoder,
            segment_size=self.audio_segment_size,
            device=device,
        )
        codecs_to_concat = []
        for codecs in code_list:
            padded_codes = self.process_audio(
                codecs
            )  # [T//group_size, group_size, audio_channels]
            codecs_to_concat.append(padded_codes)
        audio_codes = torch.cat(
            codecs_to_concat, dim=0
        )  # [T//group_size, group_size, audio_channels]

        _audio_embeddings = self.apply_speech_embeddings(audio_codes)
        audio_embeds = self.apply_input_local_transformer(
            _audio_embeddings
        )  #  [T//group_size,  group_size, hidden_size]
        B = audio_embeds.shape[0]
        audio_embeds = self.projection(audio_embeds.reshape(B, -1))
        return audio_embeds
