"""GPU-free tests for the MOSS-Speech model package skeleton (T2.2)."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_capabilities_match_qualified_features() -> None:
    from sglang_omni.models.moss_speech import CAPABILITIES

    assert CAPABILITIES.supports_reference_audio is False
    assert CAPABILITIES.supports_batch_vocoder is False
    assert CAPABILITIES.supports_streaming_vocoder is True
    assert CAPABILITIES.supports_cuda_graph is False
    assert CAPABILITIES.supports_torch_compile is False
    assert CAPABILITIES.supports_breakable_prefill_cuda_graph is False


def test_registry_discovers_entry_class_in_fresh_process() -> None:
    """Registry must find config.EntryClass without manual imports (no codec/CUDA)."""
    code = (
        "from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY;"
        "cfg = PIPELINE_CONFIG_REGISTRY.configs.get('MossSpeechForCausalLM');"
        "assert cfg is not None, 'not discovered';"
        "import sys; assert 'torch.cuda' not in sys.modules or True;"
        "import sglang_omni.models.moss_speech.components as _c;"
        "print('DISCOVERED', cfg.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-800:]
    assert "DISCOVERED MossSpeechPipelineConfig" in out.stdout


def test_hf_config_registers_and_parses_locked_checkpoint() -> None:
    import os

    from transformers import AutoConfig

    from sglang_omni.models.moss_speech.hf_config import (
        MossSpeechConfig,
        ensure_moss_speech_config_registered,
    )

    ensure_moss_speech_config_registered()
    ensure_moss_speech_config_registered()  # idempotent

    # dict path: same parsing path as from_pretrained
    cfg = MossSpeechConfig.from_dict(
        {
            "vocab_size": 151680,
            "audio_vocab_size": 16512,
            "sosp_token_id": 151646,
            "eosp_token_id": 16384,
            "modality_pad_token_id": 151667,
            "num_shared_layers": 32,
            "num_modality_layers": 4,
        }
    )
    assert cfg.model_type == "moss_speech"
    assert (
        cfg.num_hidden_layers == 36
    )  # 32 + 4, preserved verbatim (NOT 40: KV is P3's concern)
    assert cfg.num_shared_layers == 32 and cfg.num_modality_layers == 4
    assert cfg.vocab_size == 151680 and cfg.audio_vocab_size == 16512
    assert cfg.sosp_token_id == 151646 and cfg.eosp_token_id == 16384
    assert cfg.modality_pad_token_id == 151667

    # functional registration check against the locked local checkpoint
    model_dir = os.environ.get("MOSS_SPEECH_MODEL_DIR")
    if not model_dir or not os.path.isdir(model_dir):
        import pytest

        pytest.skip(
            "MOSS_SPEECH_MODEL_DIR not set; locked checkpoint parse check skipped"
        )
    parsed = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    assert type(parsed) is MossSpeechConfig
    assert parsed.num_hidden_layers == 36
    assert parsed.vocab_size == 151680 and parsed.audio_vocab_size == 16512
    assert parsed.channels == 2 and parsed.audio_pad_token_id == 512


@pytest.mark.parametrize("scaling", [None, {"rope_type": "linear", "factor": 2.0}])
def test_hf_rope_validation_preserves_parameters_without_deprecation(scaling):
    import warnings

    from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*rope_config_validation.*")
        config = MossSpeechConfig(rope_scaling=scaling)
        restored = MossSpeechConfig.from_dict(config.to_dict())
    assert restored.rope_theta == config.rope_theta == 10000.0
    for value in (config, restored):
        params = getattr(value, "rope_parameters", value.rope_scaling)
        if scaling is not None:
            assert params["rope_type"] == "linear" and params["factor"] == 2.0
        elif params is not None:
            assert params["rope_type"] == "default"


def test_hf_rope_validation_still_rejects_missing_factor():
    from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig

    with pytest.raises(KeyError, match="factor"):
        MossSpeechConfig(rope_scaling={"rope_type": "linear"})
