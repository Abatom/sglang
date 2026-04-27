from types import SimpleNamespace

import torch

from sglang.srt.layers.linear import QKVParallelLinear


class _FakeBlockScaleParam:
    output_dim = 0

    def __init__(self):
        self.loaded = []

    def load_qkv_weight(
        self,
        loaded_weight,
        num_heads,
        shard_id,
        shard_offset,
        shard_size,
        tp_rank,
        use_presharded_weights,
    ):
        self.loaded.append((shard_id, tuple(loaded_weight.shape)))


def test_qkv_block_scale_uses_v_head_size_for_uneven_kv_heads():
    layer = QKVParallelLinear.__new__(QKVParallelLinear)
    layer.total_num_heads = 64
    layer.total_num_kv_heads = 8
    layer.num_heads = 16
    layer.num_kv_heads = 2
    layer.num_kv_head_replicas = 1
    layer.head_size = 192
    layer.v_head_size = 128
    layer.tp_rank = 0
    layer.use_presharded_weights = False
    layer.quant_method = SimpleNamespace(
        quant_config=SimpleNamespace(weight_block_size=(128, 128))
    )
    param = _FakeBlockScaleParam()
    loaded_weight = torch.empty(116, 32)

    QKVParallelLinear._load_qkv_block_scale(layer, param, loaded_weight)

    assert param.loaded == [
        ("q", (96, 32)),
        ("k", (12, 32)),
        ("v", (8, 32)),
    ]
