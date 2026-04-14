# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""

import logging
from typing import Iterable, List, Optional, Tuple

import torch
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
)
from sglang.srt.models.mimo_audio import MimoAudioEncoder, MimoAudioEncoderConfig
from sglang.srt.models.mimo_v2_flash import (
    MiMoV2FlashForCausalLM,
)
from sglang.srt.models.mimo_vit import Mimo_VisionTransformer, Mimo_VLVisionConfig
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

MiMoV2OmniConfig = None


class MiMoV2OmniForCausalLM(MiMoV2FlashForCausalLM):
    def __init__(
        self,
        config: MiMoV2OmniConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        vision_config = Mimo_VLVisionConfig.from_dict(config.vision_config)
        # Omni ViT/Audio Encoder BF16
        self.visual = Mimo_VisionTransformer(
            vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=None,
            prefix=add_prefix("visual", prefix),
        )
        self.audio_config = MimoAudioEncoderConfig(**config.audio_config)
        self.audio_encoder = MimoAudioEncoder(self.audio_config)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        features = self.audio_encoder.get_audio_feature(items)
        return features

    def get_input_embeddings(self):
        if getattr(self.config, "encoder_only", False):
            return None
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # encoder_only mode: skip language model forward
        if getattr(self.config, "encoder_only", False):
            raise NotImplementedError(
                "forward() is not supported in encoder_only mode. "
                "Use get_image_feature(), get_video_feature(), or get_audio_feature() instead."
            )

        hidden_states_ = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        hidden_states, hidden_states_before_norm = hidden_states_[0], hidden_states_[1]

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    hidden_states_before_norm=hidden_states_before_norm,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        stacked_params_mapping_vit = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # try:
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = DeepEPMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        # except:
        #     expert_params_mapping = []

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # audio

            if "audio" in name:
                # audio_projection

                # logger.info(f"loading audio weights: {name}")

                if "projection" in name:
                    if (
                        "audio_encoder.audio_projection" in name
                        and "audio_encoder.projection" not in name
                    ):
                        name = name.replace(
                            "audio_encoder.audio_projection", "audio_encoder.projection"
                        )
                    elif (
                        "audio_projection" in name
                        and "audio_encoder.projection" not in name
                    ):
                        name = name.replace(
                            "audio_projection", "audio_encoder.projection"
                        )
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )

                    weight_loader(param, loaded_weight)
                    continue

                if "input_local_transformer" in name:
                    if (
                        "audio_input_local_transformer" in name
                        and "audio_encoder.input_local_transformer" not in name
                    ):
                        name = name.replace(
                            "audio_input_local_transformer",
                            "audio_encoder.input_local_transformer",
                        )
                    if name not in params_dict:
                        logger.warning(
                            f"Parameter {name} not found in params_dict, skipping"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    continue

            if "speech_embeddings" in name:
                # logger.info(f"loading speech weights: {name}")

                if (
                    "speech_embeddings" in name
                    and "audio_encoder.speech_embeddings" not in name
                ):
                    name = name.replace(
                        "speech_embeddings", "audio_encoder.speech_embeddings"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight[: param.shape[0], :])
                continue

            if "visual" in name:
                name = name.replace("vision_model.", "")
                name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                match_stacked_vit = False
                for param_name, weight_name, shard_id in stacked_params_mapping_vit:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        match_stacked_vit = True
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    match_stacked_vit = True
                    break
                if match_stacked_vit:
                    continue
                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                # if "merger" in name and name.endswith(".bias"):
                #     loaded_weight = param.data * 0.0
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

                if name.endswith("patch_embed.proj.weight"):
                    patch_embed = self.get_submodule(name.rsplit(".", 2)[0])
                    if hasattr(patch_embed, "sync_proj_weight_linear_format"):
                        patch_embed.sync_proj_weight_linear_format()
                continue

            layer_id = get_layer_id(name)
            if (
                not getattr(self.config, "encoder_only", False)
                and not getattr(self.config, "language_only", False)
                and layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue

            if "mtp" in name:
                continue

            # Checkpoint stores fused qkv_proj; manually chunk for TP
            if "qkv_proj" in name:
                if name in params_dict:
                    tp_size = get_attention_tp_size()
                    tp_rank = get_attention_tp_rank()
                    param = params_dict[name]
                    loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    default_weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if (
                    "compression_attention" in name
                    or "hybrid_softmax_attention" in name
                    or "compressed_softmax_attn" in name
                ):
                    continue
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading visual/language model weights
                if (
                    getattr(self.config, "encoder_only", False)
                    or getattr(self.config, "language_only", False)
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading visual/language model weights
                    if (
                        getattr(self.config, "encoder_only", False)
                        or getattr(self.config, "language_only", False)
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Skip loading visual/language model weights
                    if (
                        getattr(self.config, "encoder_only", False)
                        or getattr(self.config, "language_only", False)
                    ) and name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                        if "attention_sink_bias" in name:
                            start = get_attention_tp_rank() * param.numel()
                            param.data.copy_(
                                loaded_weight[start : start + param.numel()]
                            )
                        else:
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = MiMoV2OmniForCausalLM
