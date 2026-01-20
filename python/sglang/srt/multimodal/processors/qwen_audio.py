import re
from typing import List, Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen2_audio import Qwen2AudioForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class Qwen2AudioMultimodalProcessor(BaseMultimodalProcessor):
    models = [Qwen2AudioForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"
        )
        # Collect special token ids
        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update({"feature_attention_mask": Modality.AUDIO})

    def build_audio_input_ids(
        self, prompt: Union[str, List[int]], audio_feature_lens: torch.Tensor
    ):
        """
        Build input_ids from prompt and audio_feature_lens for audio EPD.

        Args:
            prompt: The input text or tokenized input_ids
            audio_feature_lens: Tensor containing the length of each audio embedding

        Returns:
            input_ids: List of token ids with audio tokens expanded
            offsets: List of (start, end) tuples for each audio embedding position
        """
        if not isinstance(prompt, list):
            prompt = self._processor.tokenizer.encode(prompt)

        input_ids = []
        offsets = []

        # Find all audio_start_id positions (audio_bos token)
        audio_start_indices = [
            i for i in range(len(prompt) - 1) if prompt[i] == self.audio_start_id
        ]

        cur_idx = 0
        for cur_audio_idx, audio_start_idx in enumerate(audio_start_indices):
            assert cur_idx <= audio_start_idx
            # Include tokens up to and including audio_start_id (audio_bos)
            input_ids.extend(prompt[cur_idx : audio_start_idx + 1])
            audio_offset_start = len(input_ids)

            # Get the number of audio tokens for this audio
            audio_token_num = int(audio_feature_lens[cur_audio_idx].item())
            input_ids.extend([self.audio_token_id] * audio_token_num)

            # Find the audio_end position (should be audio_start_idx + 2 in original prompt)
            # because original format is: audio_bos, AUDIO, audio_eos
            cur_idx = audio_start_idx + 2  # skip audio_bos and single AUDIO token

            offsets.append((audio_offset_start, len(input_ids) - 1))

        # Add remaining tokens
        input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets

    def get_mm_data(
        self,
        prompt: Union[str, List[int]],
        embeddings: torch.Tensor,
        audio_feature_lens: torch.Tensor,
    ):
        """
        Get multimodal data for EPD (Encode-Prefill-Decode) disaggregation.

        This method is called by the receiver to reconstruct the multimodal inputs
        from precomputed embeddings sent by the encoder server.

        Args:
            prompt: The input text or tokenized input_ids
            embeddings: Precomputed audio embeddings from the encoder
            audio_feature_lens: Tensor containing the length of each audio embedding

        Returns:
            Dictionary containing input_ids and mm_items for the scheduler
        """
        input_ids, offsets = self.build_audio_input_ids(prompt, audio_feature_lens)

        mm_items = [
            MultimodalDataItem(
                modality=Modality.AUDIO,
                offsets=offsets,
                precomputed_embeddings=embeddings,
                model_specific_data={"audio_feature_lens": audio_feature_lens},
            )
        ]

        return {
            "input_ids": input_ids,
            "mm_items": mm_items,
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        assert (
            "feature_attention_mask" in ret
        ), "feature_attention_mask not found in processor output"
        input_lengths = ret["feature_attention_mask"].sum(dim=-1)
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1

        mm_items[0].audio_feature_lens = output_lengths

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
