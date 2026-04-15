"""Inference-only MiMo vision model: attention + ViT."""

from __future__ import annotations

import functools
import logging
import math
from functools import lru_cache, partial
from typing import Any, Callable, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from sglang.srt.distributed import utils as dist_utils
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionPatchMerger, Qwen2_5_VLMLP
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    get_device_capability,
    is_blackwell,
    is_cuda,
    print_info_once,
)

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel.flash_attn import flash_attn_varlen_func

logger = logging.getLogger(__name__)


# TODO: requires real seqlens from images
@functools.lru_cache(maxsize=128)
def _get_cu_seqlens_for_shape(batch_size: int, seqlen: int, device) -> torch.Tensor:
    """
    Generates cumulative sequence lengths (cu_seqlens) for a given batch_size, seqlen, and device.
    Caches the result based on these parameters.
    """
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=device,
    )
    return cu_seqlens


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product

    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        flatten_batch: bool = False,
        softmax_in_single_precision: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = flatten_batch
        self.softmax_in_single_precision = softmax_in_single_precision
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_size)

    @staticmethod
    @lru_cache(maxsize=128)
    def _generate_mask_cache(
        s: int, flatten_batch: bool, cu_seqlens: tuple
    ) -> torch.BoolTensor:
        """
        Generate a boolean attention mask with caching mechanism.
        Args:
            s: sequence length
            flatten_batch: whether to flatten batch dimension
            cu_seqlens: tuple of cumulative sequence lengths
        Returns:
            attention mask tensor of shape [b, 1, s, s] or [1, s, s]
        """
        if flatten_batch:
            mask = torch.zeros([1, s, s], dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                start = cu_seqlens[i - 1]
                end = cu_seqlens[i]
                mask[..., start:end, start:end] = True
        else:
            # [1, 1, 1, s]
            row_indices = torch.arange(s).view(1, 1, 1, s)
            # [1, 1, s, 1]
            col_indices = torch.arange(s).view(1, 1, s, 1)
            # [b, 1, 1, 1]
            seq_lens = torch.tensor(
                [end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])],
            ).view(-1, 1, 1, 1)

            mask = (row_indices < seq_lens) & (col_indices < seq_lens)

        return mask

    def generate_patch_attention_mask(
        self,
        s: int,
        cu_seqlens: Optional[torch.Tensor],
        flatten_batch: bool = False,
    ) -> Optional[torch.Tensor]:
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        Args:
            s: sequence length
            cu_seqlens: cumulative sequence lengths tensor. If not, returns an empty mask
            flatten_batch: whether to flatten batch dimension
        Returns:
            attention mask tensor or None
        """
        if cu_seqlens is None:
            return None

        cu_seqlens_tuple = tuple(cu_seqlens.cpu().tolist())

        return self._generate_mask_cache(s, flatten_batch, cu_seqlens_tuple)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if self.flatten_batch:
            assert bsz == 1, "flatten_batch is True, bsz must be 1"

        assert q.dim() == 3, q.shape

        s = q.shape[0] // bsz

        # [b, 1, s, s]
        if attention_mask is None:
            attention_mask = self.generate_patch_attention_mask(
                s, cu_seqlens, flatten_batch=self.flatten_batch
            )

        if attention_mask is None:
            if self.softmax_in_single_precision:
                raise RuntimeError("Empty attention mask")
        else:
            attention_mask = attention_mask.to(device=q.device)

        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]

        if self.softmax_in_single_precision:
            k = rearrange(k, "b h s d -> b h d s")
            attn_weights = torch.matmul(q, k) * self.scale
            del k
            # masking
            attention_mask = (~attention_mask) * torch.finfo(q.dtype).min
            attn_weights = attn_weights + attention_mask
            del attention_mask
            # full-precision
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=False
            )
            output = torch.matmul(attn_weights, v)
            del attn_weights, v
        else:
            # SDPA
            # [b, h, s, head_size]
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout,
                is_causal=False,
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]
        output = rearrange(output, "b h s d -> (b s) h d")

        return output


class VisionTritonAttention(nn.Module):
    """
    Triton-implemented attention without a causal mask
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        bsz: int,
        seq_len: int,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if cu_seqlens is None:
            cu_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)

        # [b * s, head, head_size]
        output = torch.empty_like(q)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()
        context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens.cuda(),
            seq_lens.cuda(),
            max_seqlen,
            is_causal=False,
        )

        return output


class VisionFlash3Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlash3Attention is only available for cuda")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: int,
        bsz: int,
        seq_len: int,
        window_size: Tuple[int, int],
        s_aux,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        fa_kwargs = dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            window_size=window_size,
        )
        if s_aux is not None:
            fa_kwargs["sinks"] = s_aux
        output = flash_attn_varlen_func(q, k, v, **fa_kwargs)

        return output


QKV_BACKEND_IMPL = {
    "triton_attn": VisionTritonAttention,
    "sdpa": VisionSdpaAttention,
    "fa3": VisionFlash3Attention,
}


class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for multimodal transformers.


    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        softmax_in_single_precision (bool, default to False):
            if ``True``, the softmax will be performed in single-precision
            Otherwise, it will be performed in half-precision

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        num_kv_heads: Optional[int] = None,
        kv_channels: Optional[int] = None,
        qk_channels: Optional[int] = None,
        qkv_backend: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        dropout: float = 0.0,
        softmax_in_single_precision: bool = False,
        flatten_batch: bool = False,
        prefix: str = "",
        proj_bias: bool = True,
        num_dummy_heads: int = 0,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        layer_norm_eps: float = 1e-06,
        customized_position_embedding_applier: Callable[
            [torch.Tensor, torch.Tensor, Any, Any], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        use_data_parallel: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        num_kv_heads = num_kv_heads or num_heads
        self.dropout = dropout
        self.head_size = embed_dim // num_heads
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_heads, self.tp_size
        )
        self.num_attention_kv_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_kv_heads, self.tp_size
        )

        # Use qk_channels/kv_channels if provided (for GQA with custom head dims), otherwise use head_size
        self.qk_channels = qk_channels if qk_channels is not None else self.head_size
        self.kv_channels = kv_channels if kv_channels is not None else self.head_size
        self.q_size = self.num_attention_heads_per_partition * self.qk_channels
        self.kv_size = self.num_attention_kv_heads_per_partition * self.kv_channels
        self.qk_normalization = qk_normalization

        # Additional dummy heads are used to enable TP for common GPU counts.
        # Use qk_channels for dummy_dim if provided, otherwise use head_size
        effective_head_dim = (
            self.qk_channels if qk_channels is not None else self.head_size
        )
        self.dummy_dim = (num_dummy_heads + num_heads) * effective_head_dim

        if self.qk_normalization:
            self.q_norm = RMSNorm(
                self.dummy_dim, eps=layer_norm_eps, var_hidden_size=embed_dim
            )
            self.k_norm = RMSNorm(
                self.dummy_dim, eps=layer_norm_eps, var_hidden_size=embed_dim
            )

        # Select attention backend via a unified method
        _passed_backend = qkv_backend
        qkv_backend = self._determine_attention_backend(_passed_backend)
        if (
            get_global_server_args().mm_attention_backend is None
            and _passed_backend is None
        ):
            print_info_once(f"Multimodal attention backend not set. Use {qkv_backend}.")
        print_info_once(f"Using {qkv_backend} as multimodal attention backend.")

        ##########################
        # Sink Attention
        ##########################
        use_sink = kwargs.get("use_sink", False)
        if use_sink:
            assert qkv_backend == "fa3"
            # Init and load full sharded sink parameters
            # But use only the local partition in attention forward
            self.sinks = nn.Parameter(
                torch.empty(
                    self.num_attention_heads_per_partition * self.tp_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
        else:
            self.sinks = None
        self.window_size = kwargs.get("window_size", (-1, -1))

        self.customized_position_embedding_applier = (
            customized_position_embedding_applier
        )
        self.qkv_backend = QKV_BACKEND_IMPL[qkv_backend](
            head_dim=self.head_size,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_kv_heads_per_partition,
            dropout=dropout,
            flatten_batch=flatten_batch,
            softmax_in_single_precision=softmax_in_single_precision,
        )

        self.use_qkv_parallel = use_qkv_parallel
        if use_qkv_parallel:
            # Use qk_channels as head_size if provided (for GQA with custom head dims)
            effective_head_size = (
                self.qk_channels if qk_channels is not None else self.head_size
            )
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=effective_head_size,
                total_num_heads=num_dummy_heads + num_heads,
                total_num_kv_heads=num_dummy_heads + num_kv_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        else:
            self.qkv_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=3 * self.dummy_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        self.proj = RowParallelLinear(
            input_size=self.dummy_dim,
            output_size=embed_dim,
            bias=proj_bias,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("proj", prefix),
        )

    def _determine_attention_backend(self, passed_backend: Optional[str]) -> str:
        """Decide the multimodal attention backend string.

        Priority: server args override > constructor arg > platform default.

        Platform defaults:
        - CUDA: "triton_attn"
        - Non-CUDA: "sdpa"
        """
        override_backend = get_global_server_args().mm_attention_backend
        if override_backend is not None:
            backend = override_backend
        elif passed_backend is not None:
            backend = passed_backend
        elif is_cuda():
            major, minor = get_device_capability()
            if major == 9:
                backend = "fa3"
            else:
                backend = "triton_attn"
        else:
            backend = "sdpa"
        if backend == "fa3" and is_blackwell():
            raise ValueError("The 'fa3' backend is not supported on Blackwell GPUs")

        return backend

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        """apply qk norm for internvl vit attn"""
        q = q.flatten(1, 2)
        k = k.flatten(1, 2)

        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        q = q.unflatten(-1, (-1, self.head_size))
        k = k.unflatten(-1, (-1, self.head_size))
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: int = -1,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_attn: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, head * head_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        x_shape = x.shape
        bsz, s, _ = x_shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition
        if self.use_qkv_parallel:
            # [b, s, embed_dim] --> [b, s, embed_dim]
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [b, s, embed_dim] --> [b * s, head, head_size]
            q = q.reshape(bsz * s, head, -1).contiguous()
            k = k.reshape(bsz * s, kv_head, -1).contiguous()
            v = v.reshape(bsz * s, kv_head, -1).contiguous()
        else:
            # [b, s, embed_dim] --> [s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]
            qkv, _ = self.qkv_proj(x)

            # [s, b, head, head_dim_sum]
            new_x_shape = qkv.size()[:-1] + (
                head,
                self.q_size + 2 * self.kv_size,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
            ]

        if position_embeddings is not None:
            original_q_shape = q.shape
            original_k_shape = k.shape

            if self.customized_position_embedding_applier is not None:
                q, k = self.customized_position_embedding_applier(
                    q, k, position_embeddings, x_shape
                )
                q = q.view(original_q_shape)
                k = k.view(original_k_shape)
            else:
                cos, sin = position_embeddings

                # [total_tokens, head, head_size] for q, [total_tokens, kv_head, head_size] for k
                q = q.view(-1, head, self.qk_channels)
                k = k.view(-1, kv_head, self.kv_channels)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                q = q.view(original_q_shape)
                k = k.view(original_k_shape)

        if q.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q = rearrange(q, "b s ... -> (b s) ...")
        if k.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            k = rearrange(k, "b s ... -> (b s) ...")
        if v.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            v = rearrange(v, "b s ... -> (b s) ...")

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        # internvl
        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        if full_attn:
            window_size = (-1, -1)
            s_aux = None
        else:
            window_size = self.window_size
            q_head_start = self.tp_rank * self.num_attention_heads_per_partition
            q_head_end = (self.tp_rank + 1) * self.num_attention_heads_per_partition
            s_aux = (
                self.sinks[q_head_start:q_head_end] if self.sinks is not None else None
            )

        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            bsz=bsz,
            seq_len=s,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
            window_size=window_size,
            s_aux=s_aux,
        )

        assert output.dim() == 3, output.shape

        if self.use_qkv_parallel:
            # [b * s, h, head_size] --> [b, s, h * head_size]
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

            # [b, s, h * head_size] --> [b, s, h * head_size]
            output, _ = self.proj(output)
        else:
            # [b * s, h, head_size] --> [s, b, h * head_size]
            context_layer = rearrange(
                output, "(b s) h d -> s b (h d)", b=bsz, s=s
            ).contiguous()

            # [s, b, h * head_size] --> [s, b, h * head_size]
            output, _ = self.proj(context_layer)

            # [s, b, h * head_size] --> [b, s, h * head_size]
            output = output.view(bsz, s, -1)

        return output


# ---------------------------------------------------------------------------
# Vision config / ViT
# ---------------------------------------------------------------------------


class Mimo_VLVisionConfig(PretrainedConfig):
    model_type = "mimovl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=28,
        hidden_size=1280,
        hidden_act="silu",
        intermediate_size=4608,
        num_heads=32,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=2,
        window_size=128,
        out_hidden_size=2048,
        fullatt_block_indexes=[7, 15, 23, 31],
        initializer_range=0.02,
        kv_channels=64,  # HACK
        qk_channels=64,
        num_query_groups=4,
        num_key_value_heads=8,
        vit_window_attn_types=None,
        visual_token_window_size=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        # Support GQA: if num_key_value_heads is not provided, default to num_heads (MHA)
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range
        self.kv_channels = kv_channels
        self.qk_channels = qk_channels
        self.num_query_groups = num_query_groups
        self.vit_window_attn_types = vit_window_attn_types or [-1] * depth
        self.visual_token_window_size = visual_token_window_size


class Mimo_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1536,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )
        self.proj_weight_linear_format = None

    @torch.no_grad()
    def sync_proj_weight_linear_format(self):
        self.proj_weight_linear_format = self.proj.weight.view(self.embed_dim, -1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = F.linear(
            hidden_states.to(dtype=target_dtype), self.proj_weight_linear_format
        )
        return hidden_states


class Mimo_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        attn_implementation: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-6,
        use_sink: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        num_kv_heads: Optional[int] = None,
        qk_channels: Optional[int] = None,
        kv_channels: Optional[int] = None,
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)
        self.use_data_parallel = use_data_parallel

        if attn_implementation is None:
            softmax_in_single_precision = False
            qkv_backend = None
            flatten_batch = True
        elif attn_implementation == "sdpa":
            softmax_in_single_precision = False
            qkv_backend = "sdpa"
            flatten_batch = True
        elif attn_implementation == "flash_attention_2":
            softmax_in_single_precision = False
            qkv_backend = "triton_attn"
            flatten_batch = True
        elif attn_implementation == "eager":
            softmax_in_single_precision = True
            qkv_backend = "sdpa"
            flatten_batch = True
        elif attn_implementation == "flash_attention_3":
            softmax_in_single_precision = False
            qkv_backend = "fa3"
            flatten_batch = True

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            rotary_embed="normal",
            proj_bias=True,
            qkv_bias=True,
            qkv_backend=qkv_backend,
            softmax_in_single_precision=softmax_in_single_precision,
            flatten_batch=flatten_batch,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
            use_sink=use_sink,
            window_size=window_size,
            num_kv_heads=num_kv_heads,
            qk_channels=qk_channels,
            kv_channels=kv_channels,
            use_data_parallel=use_data_parallel,
        )
        self.mlp = Qwen2_5_VLMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: torch.Tensor,
        full_attn: bool = True,
    ) -> torch.Tensor:
        S, B, H = x.shape
        # norm1: flatten to 2D -> [S*B, H], then reshape back
        x2d = x.reshape(-1, H)
        hidden_states = self.norm1(x2d).reshape(S, B, H)

        # Attention expects [B, S, H]
        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            full_attn=full_attn,
        )
        attn = rearrange(attn, "b s h -> s b h")

        # norm2 with fused residual-add: also 2D
        attn2d = attn.reshape(-1, H)
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
        x_norm = x_norm_2d.reshape(S, B, H)
        x_after_add = x_after_add_2d.reshape(S, B, H)

        # MLP and final residual
        mlp_out = self.mlp(x_norm)
        x = x_after_add + mlp_out
        return x


class Mimo_VisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: Mimo_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.server_args = get_global_server_args()
        self.vit_window_attn_types = vision_config.vit_window_attn_types
        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size
        in_channels: int = vision_config.in_channels
        hidden_size: int = vision_config.hidden_size
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        # Support GQA: get num_kv_heads from config if available, otherwise default to num_heads
        num_kv_heads = getattr(vision_config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_heads  # Default to MHA if not specified
        self.num_kv_heads = num_kv_heads
        self.qk_channels = getattr(vision_config, "qk_channels", None)
        self.kv_channels = getattr(vision_config, "kv_channels", None)
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.use_data_parallel = self.server_args.mm_enable_dp_encoder
        mlp_hidden_size: int = vision_config.intermediate_size
        self.patch_embed = Mimo_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        self.use_sink = getattr(vision_config, "use_sink", False)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        # head_dim = hidden_size // num_heads  # HACK
        head_dim = (
            self.qk_channels
            if self.qk_channels is not None
            else hidden_size // num_heads
        )
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.visual_token_window_size = getattr(
            vision_config, "visual_token_window_size", -1
        )
        self.blocks = nn.ModuleList(
            [
                Mimo_VisionBlock(
                    dim=hidden_size,
                    intermediate_dim=mlp_hidden_size,
                    num_heads=num_heads,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    attn_implementation="flash_attention_3",
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                    use_sink=(
                        self.use_sink if i not in self.fullatt_block_indexes else False
                    ),
                    window_size=(
                        self.visual_token_window_size,
                        self.visual_token_window_size,
                    ),
                    num_kv_heads=num_kv_heads,
                    qk_channels=self.qk_channels,
                    kv_channels=self.kv_channels,
                    use_data_parallel=self.use_data_parallel,
                )
                for i in range(depth)
            ]
        )

        self.vision_config = vision_config
        self.merger = Qwen2_5_VisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=self.use_data_parallel,
        )
        self._post_init()

    def apply_index(self, tensor: torch.Tensor, index: torch.Tensor):
        tensor = tensor.unflatten(0, (-1, self.spatial_merge_unit))
        tensor = tensor[index]
        tensor = tensor.flatten(0, 1)
        return tensor

    def _post_init(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                param.data.zero_()

    def get_window_index_1d(self, grid_thw, col=True):
        window_index: list = []
        window_index_id = 0
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            if col:
                index_new = index.transpose(1, 2).reshape(-1)
            else:
                index_new = index.reshape(-1)
            window_index.append(index_new + window_index_id)
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(
            window_index,
            dim=0,
        )
        return window_index

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.gate_up_proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for i in range(grid_thw.size(0)):
            t, h, w = grid_thw[i].tolist()
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)

            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def _prepare_forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ):
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)
        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index_1d_col = self.get_window_index_1d(grid_thw, col=True).to(
            device=x.device
        )
        # Move window_index to the same device as x before using it to index x
        reverse_window_index_1d_col = torch.argsort(window_index_1d_col).to(
            device=x.device
        )

        # Ensure rotary_pos_emb is on the same device/dtype as x
        rotary_pos_emb = rotary_pos_emb.to(device=x.device)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

        def get_position_embeddings(emb, x):
            position_embeddings = (emb.cos(), emb.sin())
            # After building position_embeddings, make sure both cos and sin are on the same device/dtype as the attention input
            position_embeddings = (
                position_embeddings[0].to(x.device),
                position_embeddings[1].to(x.device),
            )
            return position_embeddings

        # compute cu_seqlens - move cu_seqlens to GPU and make it int32
        seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                seqlens.cumsum(dim=0).to(device=x.device, dtype=torch.int32),
            ]
        )
        max_seqlen = seqlens.max().item()

        row_based_embeddings = get_position_embeddings(emb, x)
        col_based_embeddings = get_position_embeddings(
            self.apply_index(emb, window_index_1d_col), x
        )

        # transformers
        x = x.unsqueeze(1)  # [S, 1, H]

        return (
            x,
            row_based_embeddings,
            col_based_embeddings,
            window_index_1d_col,
            reverse_window_index_1d_col,
            cu_seqlens,
            max_seqlen,
        )

    def run_blocks(
        self,
        x: torch.Tensor,
        row_based_embeddings: Tuple[torch.Tensor, torch.Tensor],
        col_based_embeddings: Tuple[torch.Tensor, torch.Tensor],
        window_index_1d_col: torch.Tensor,
        reverse_window_index_1d_col: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        for layer_num, blk in enumerate(self.blocks):
            window_attn_type = self.vit_window_attn_types[layer_num]

            # window_attn_type = 1: col-based SWA
            if window_attn_type == 1 and (
                layer_num == 0 or self.vit_window_attn_types[layer_num - 1] != 1
            ):
                x = self.apply_index(x, window_index_1d_col)

            if (
                layer_num > 0
                and window_attn_type != 1
                and self.vit_window_attn_types[layer_num - 1] == 1
            ):
                x = self.apply_index(x, reverse_window_index_1d_col)

            position_embeddings = (
                col_based_embeddings if window_attn_type == 1 else row_based_embeddings
            )
            full_attn = layer_num in self.fullatt_block_indexes

            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                full_attn=full_attn,
            )
        x = self.merger(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        (
            x,
            row_based_embeddings,
            col_based_embeddings,
            window_index_1d_col,
            reverse_window_index_1d_col,
            cu_seqlens,
            max_seqlen,
        ) = self._prepare_forward(x, grid_thw)

        return self.run_blocks(
            x,
            row_based_embeddings,
            col_based_embeddings,
            window_index_1d_col,
            reverse_window_index_1d_col,
            cu_seqlens,
            max_seqlen,
        )
