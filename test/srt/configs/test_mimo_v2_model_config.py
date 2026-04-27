from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    is_hybrid_swa_model,
    is_multimodal_model,
)


def test_mimo_v2_public_arch_maps_to_mtp_for_draft_model():
    config = SimpleNamespace(
        is_draft_model=True,
        hf_config=SimpleNamespace(architectures=["MiMoV2ForCausalLM"]),
    )

    ModelConfig._config_draft_model(config)

    assert config.hf_config.architectures == ["MiMoV2MTP"]


def test_mimo_v2_public_arch_is_multimodal_and_hybrid_swa():
    assert is_multimodal_model(["MiMoV2ForCausalLM"])
    assert is_hybrid_swa_model(["MiMoV2ForCausalLM"])


def test_mimo_v2_public_arch_uses_hybrid_layer_pattern():
    hf_text_config = SimpleNamespace(
        num_hidden_layers=4,
        hybrid_layer_pattern=[0, 1, 1, 0],
    )

    swa_layers, full_layers = get_hybrid_layer_ids(
        ["MiMoV2ForCausalLM"], hf_text_config
    )

    assert swa_layers == [1, 2]
    assert full_layers == [0, 3]


def test_mimo_v2_public_arch_detects_attention_sinks():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["MiMoV2ForCausalLM"]),
        hf_text_config=SimpleNamespace(
            add_swa_attention_sink_bias=True,
            add_full_attention_sink_bias=False,
        ),
    )

    assert ModelConfig._detect_attention_sinks(config)
