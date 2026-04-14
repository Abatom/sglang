# copied from https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer.git
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.layers.audio_linear import (
    AudioRotaryEmbedding,
    ResidualVectorQuantizer,
    apply_rotary_pos_emb,
)
from sglang.srt.utils import is_cuda
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

_is_cuda = is_cuda()

if _is_cuda:
    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func
    except Exception:
        try:
            from flash_attn_3.flash_attn_interface import flash_attn_varlen_func
        except Exception:
            try:
                from flash_attn.flash_attn_interface import flash_attn_varlen_func
            except Exception:
                flash_attn_varlen_func = None

logger = logging.get_logger(__name__)


class MiMoAudioTokenizerConfig(PretrainedConfig):
    model_type = "mimo_audio_tokenizer"

    def __init__(
        self,
        max_audio_seconds: int = 1800,
        stride_size: int = 2,
        avg_pooler: int = 1,
        d_model: int = 768,
        scale_embedding: bool = True,
        kernel_size: int = 3,
        activation_function: str = "gelu",
        encoder_layers: int = 8,
        encoder_skip_layer_id: int = None,
        encoder_attention_heads: int = 12,
        encoder_ffn_dim: int = 3072,
        encoder_causal: bool = False,
        encoder_attn_window_size: list = None,
        decoder_layers: int = 8,
        decoder_attention_heads: int = 12,
        decoder_ffn_dim: int = 3072,
        decoder_kernel_size: int = 3,
        decoder_stride_size: int = 2,
        decoder_causal: bool = True,
        decoder_attn_window_size: list = None,
        nfft: int = 1024,
        vocoder_dim: int = 512,
        vocoder_intermediate_dim: int = 4096,
        vocoder_num_layers: int = 30,
        n_mels: int = 80,
        sampling_rate: int = 24000,
        hop_length: int = 240,
        window_size: int = 1024,
        vocoder_padding: str = "same",
        fmin: int = 0,
        fmax: int = None,
        num_quantizers: int = 12,
        codebook_size: list = None,
        threshold_ema_dead_code: int = 10,
        position_embedding_type: str = "rope",
        rope_theta: int = 10000,
        rope_type: str = "default",
        ln_type: str = "LayerNorm",
        vocoder_attention_heads: int = 4,
        vocoder_attn_window_size: list = None,
        use_istft_only: bool = False,
        hybrid_attention: bool = False,
        hybrid_block_size: int = 8,
        swa_per_block: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_audio_seconds = max_audio_seconds
        self.stride_size = stride_size
        self.avg_pooler = avg_pooler
        self.d_model = d_model
        self.scale_embedding = scale_embedding
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        self.encoder_layers = encoder_layers
        self.encoder_skip_layer_id = encoder_skip_layer_id
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_causal = encoder_causal
        self.encoder_attn_window_size = (
            encoder_attn_window_size
            if encoder_attn_window_size is not None
            else [-1, -1]
        )
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_stride_size = decoder_stride_size
        self.decoder_causal = decoder_causal
        self.decoder_attn_window_size = (
            decoder_attn_window_size
            if decoder_attn_window_size is not None
            else [-1, -1]
        )
        self.nfft = nfft
        self.vocoder_dim = vocoder_dim
        self.vocoder_intermediate_dim = vocoder_intermediate_dim
        self.vocoder_num_layers = vocoder_num_layers
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.window_size = window_size
        self.vocoder_padding = vocoder_padding
        self.fmin = fmin
        self.fmax = fmax
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size if codebook_size is not None else [1024]
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.ln_type = ln_type
        self.vocoder_attention_heads = vocoder_attention_heads
        self.vocoder_attn_window_size = (
            vocoder_attn_window_size
            if vocoder_attn_window_size is not None
            else [40, 10]
        )
        self.use_istft_only = use_istft_only
        self.hybrid_attention = hybrid_attention
        self.hybrid_block_size = hybrid_block_size
        self.swa_per_block = swa_per_block


def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(
        bsz, tgt_len, 1
    )
    unpacking_index = torch.cumsum(sequence_mask.to(torch.int64).view(-1), dim=0) - 1
    return sequence_mask, unpacking_index


def unpack_hidden_states(
    hidden_states, lengths, sequence_mask=None, unpacking_index=None
):
    bsz = lengths.shape[0]
    if sequence_mask is None or unpacking_index is None:
        sequence_mask, unpacking_index = get_sequence_mask(hidden_states, lengths)
    hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
        bsz, torch.max(lengths), hidden_states.shape[-1]
    )
    return torch.where(sequence_mask, hidden_states, 0)


def get_position_ids(lengths):
    total_len = lengths.sum()
    offset = torch.cat([torch.zeros(1).to(lengths), lengths[:-1].cumsum(dim=0)])
    offset = torch.repeat_interleave(offset, lengths)
    return torch.arange(0, total_len).to(offset) - offset


LAYER_NORM = {"LayerNorm": nn.LayerNorm}


class AudioEncoderAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.causal = causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_position_embeddings=None,
    ):
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(
            bsz, self.num_heads, self.head_dim
        )

        if rope_position_embeddings is not None:
            cos, sin = rope_position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=self.causal,
            window_size=self.window_size,
        )

        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class AudioEncoderTransformerLayer(nn.Module):
    def __init__(
        self,
        config: MiMoAudioTokenizerConfig,
        causal: bool,
        attn_window_size: Tuple[int, int] = (-1, -1),
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = AudioEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            window_size=attn_window_size,
            causal=causal,
        )
        self.self_attn_layer_norm = LAYER_NORM[config.ln_type](self.embed_dim)

        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LAYER_NORM[config.ln_type](self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            rope_position_embeddings=rope_position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AudioEncoder(nn.Module):
    def __init__(
        self,
        config: MiMoAudioTokenizerConfig,
    ):
        super().__init__()
        self.config = config
        self.max_source_positions = (
            config.max_audio_seconds * config.sampling_rate // config.hop_length
        ) // config.stride_size
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.skip_layer_idx = config.encoder_skip_layer_id

        self.conv1 = nn.Conv1d(
            config.n_mels,
            config.d_model,
            kernel_size=config.kernel_size,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.kernel_size,
            stride=config.stride_size,
            padding=1,
        )

        self.position_embedding = AudioRotaryEmbedding(
            config.rope_theta,
            config.d_model // config.encoder_attention_heads,
            self.max_source_positions,
            config.rope_type,
        )

        attn_window_sizes = []
        if config.hybrid_attention:
            for i in range(config.encoder_layers):
                if i % config.swa_per_block < config.swa_per_block - 1:
                    attn_window_sizes.append(tuple(config.encoder_attn_window_size))
                else:
                    attn_window_sizes.append((-1, -1))
        else:
            attn_window_sizes = [
                tuple(config.encoder_attn_window_size)
            ] * config.encoder_layers

        self.layers = nn.ModuleList(
            [
                AudioEncoderTransformerLayer(
                    config=config,
                    causal=config.encoder_causal,
                    attn_window_size=attn_window_sizes[i],
                )
                for i in range(config.encoder_layers)
            ]
        )

        self.layer_norm = LAYER_NORM[config.ln_type](config.d_model)

        if config.avg_pooler != 1:
            self.down_sample_layer = nn.Sequential(
                nn.Conv1d(
                    config.d_model,
                    config.d_model,
                    config.avg_pooler,
                    config.avg_pooler,
                    bias=False,
                ),
                nn.GELU(),
            )
            self.down_sample_norm = LAYER_NORM[config.ln_type](config.d_model)
        else:
            self.down_sample_layer = None

        if config.num_quantizers != 0:
            self.quantizer = ResidualVectorQuantizer(
                dimension=config.d_model,
                n_q=config.num_quantizers,
                bins=config.codebook_size,
                threshold_ema_dead_code=config.threshold_ema_dead_code,
            )
        else:
            self.quantizer = None

    def get_features(self, input_features, output_length):
        input_features = input_features.to(self.conv1.weight)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        bsz, tgt_len, _ = inputs_embeds.size()
        hidden_states = inputs_embeds

        position_ids = get_position_ids(output_length).long().to(input_features.device)
        rope_position_embeddings = self.position_embedding(input_features, position_ids)

        attention_mask, unpacking_index = get_sequence_mask(
            hidden_states, output_length
        )
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(
            torch.sum(output_length), self.config.d_model
        )

        cu_seqlens = F.pad(
            torch.cumsum(output_length, dim=0), (1, 0), "constant", 0
        ).to(device=hidden_states.device, dtype=torch.int32)
        max_seqlen = torch.max(output_length).to(torch.int32).item()

        skip_connect_hidden_states = 0.0
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                rope_position_embeddings=rope_position_embeddings,
            )
            if (self.skip_layer_idx is not None) and idx == self.skip_layer_idx - 1:
                skip_connect_hidden_states = hidden_states.clone()

        hidden_states += skip_connect_hidden_states
        hidden_states = self.layer_norm(hidden_states)

        if self.down_sample_layer is not None:
            hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
                bsz, tgt_len, self.config.d_model
            )
            if hidden_states.size(1) % self.config.avg_pooler:
                pad_len = (
                    self.config.avg_pooler
                    - hidden_states.size(1) % self.config.avg_pooler
                )
                hidden_states = torch.nn.functional.pad(
                    hidden_states, (0, 0, 0, pad_len), mode="constant", value=0.0
                )
                tgt_len += pad_len
            tgt_len = tgt_len // self.config.avg_pooler
            hidden_states = self.down_sample_layer(hidden_states.transpose(1, 2))
            output_length = (
                output_length // self.config.avg_pooler
                + (output_length % self.config.avg_pooler != 0).int()
            )
            hidden_states = hidden_states.transpose(1, 2)
            attention_mask, unpacking_index = get_sequence_mask(
                hidden_states, output_length
            )
            hidden_states = torch.masked_select(hidden_states, attention_mask).view(
                torch.sum(output_length), self.config.d_model
            )
            hidden_states = self.down_sample_norm(hidden_states)

        return (
            hidden_states,
            output_length,
            attention_mask,
            unpacking_index,
            tgt_len,
            bsz,
        )

    def get_output_length(self, mel_len):
        tgt_len = mel_len + 3 - self.config.kernel_size
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1

    @torch.no_grad()
    def encode(
        self,
        input_features,
        input_lens=None,
        output_length=None,
        return_codes_only=False,
        n_q=None,
        use_quantizer=True,
    ):
        if output_length is None:
            output_length = self.get_output_length(input_lens)
        input_features = unpack_hidden_states(input_features, input_lens)
        hidden_states, output_length, attention_mask, unpacking_index, tgt_len, bsz = (
            self.get_features(
                input_features=input_features.transpose(1, 2),
                output_length=output_length,
            )
        )

        dtype = hidden_states.dtype
        if use_quantizer and self.quantizer is not None:
            self.quantizer.float()
            codes = self.quantizer.encode(hidden_states.float(), n_q=n_q)
            if return_codes_only:
                return codes, output_length
            hidden_states = self.quantizer.decode(codes)
            hidden_states = hidden_states.to(dtype)
        else:
            codes = None

        hidden_states_packed = hidden_states.clone()
        hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
            bsz, tgt_len, self.config.d_model
        )
        hidden_states = torch.where(attention_mask, hidden_states, 0)
        return hidden_states, hidden_states_packed, output_length, codes

    @torch.no_grad()
    def decode_vq(self, codes):
        self.quantizer.float()
        return self.quantizer.decode(codes)


class MiMoAudioTokenizer(PreTrainedModel):
    config_class = MiMoAudioTokenizerConfig

    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__(config)
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.encoder = AudioEncoder(config=config)
        self.downsample_rate = int(config.hop_length * 2 * config.avg_pooler)

    def get_output_length(self, mel_len):
        tgt_len = mel_len + 3 - self.config.kernel_size
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1

    @torch.no_grad()
    def encode(self, mels, input_lens, use_quantizer=True):
        input_features = mels
        encoder_output_length = self.get_output_length(input_lens)
        hidden_states, hidden_states_packed, encoder_output_length, codes = (
            self.encoder.encode(
                input_features, input_lens=input_lens, use_quantizer=use_quantizer
            )
        )
        return hidden_states, hidden_states_packed, encoder_output_length, codes
