from sglang.srt.models import mimo_v2
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.multimodal.processors.mimo_v2 import MiMoV2Processor


def test_mimo_v2_public_arch_is_the_registered_implementation():
    assert hasattr(mimo_v2, "MiMoV2ForCausalLM")
    assert mimo_v2.EntryClass[0] is mimo_v2.MiMoV2ForCausalLM


def test_mimo_v2_public_arch_resolves_to_native_model_class():
    model_cls, resolved_arch = ModelRegistry.resolve_model_cls(["MiMoV2ForCausalLM"])

    assert model_cls is mimo_v2.MiMoV2ForCausalLM
    assert resolved_arch == "MiMoV2ForCausalLM"


def test_mimo_v2_public_arch_uses_mimo_v2_multimodal_processor():
    assert mimo_v2.MiMoV2ForCausalLM in MiMoV2Processor.models
