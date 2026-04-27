import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import EagleVerifyInput


def _make_verify_input(**kwargs):
    defaults = dict(
        draft_token=torch.arange(4),
        custom_mask=torch.ones(4, dtype=torch.bool),
        positions=torch.arange(4),
        retrieve_index=torch.zeros((1, 4), dtype=torch.long),
        retrieve_next_token=torch.full((4,), -1, dtype=torch.long),
        retrieve_next_sibling=torch.full((4,), -1, dtype=torch.long),
        retrieve_cum_len=None,
        spec_steps=3,
        topk=1,
        draft_token_num=4,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=0,
        seq_lens_cpu=torch.tensor([0], dtype=torch.int32),
    )
    defaults.update(kwargs)
    return EagleVerifyInput(**defaults)


def test_eagle_verify_input_defaults_num_tokens_per_req_to_draft_tokens():
    verify_input = _make_verify_input()

    assert verify_input.num_tokens_per_req == verify_input.draft_token_num


def test_eagle_verify_input_preserves_explicit_num_tokens_per_req():
    verify_input = _make_verify_input(num_tokens_per_req=2)

    assert verify_input.num_tokens_per_req == 2
