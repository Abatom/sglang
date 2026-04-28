import torch

from sglang.srt.models.mimo_v2_nextn import MiMoV2MTP


class _Config:
    tie_word_embeddings = False


class _FakeMiMoV2MTP:
    config = _Config()
    map_model_name_to_mtp_param_name = MiMoV2MTP.map_model_name_to_mtp_param_name

    def __init__(self):
        self.qkv_weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)

    def named_parameters(self):
        return [("model.mtp_block.self_attn.qkv_proj.weight", self.qkv_weight)]


def test_mimo_v2_mtp_loads_fused_qkv_weight_without_split_rewrite():
    model = _FakeMiMoV2MTP()
    loaded_weight = torch.ones_like(model.qkv_weight)

    MiMoV2MTP.load_weights(
        model,
        [("model.mtp.layers.0.self_attn.qkv_proj.weight", loaded_weight)],
    )

    assert torch.equal(model.qkv_weight, loaded_weight)
