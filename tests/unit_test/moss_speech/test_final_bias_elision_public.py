"""Public elision selection preserves explicit defaults and factory ownership."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech import stages, streaming
from sglang_omni.models.moss_speech.components import streaming_codec
from sglang_omni.models.moss_speech.components.cosyvoice_flow.decoder import (
    CausalConditionalDecoder,
)
from sglang_omni.models.moss_speech.config import (
    MossSpeechPipelineConfig,
    MossSpeechStreamingPipelineConfig,
)


def test_default_and_explicit_false_merge():
    for cls in (MossSpeechPipelineConfig, MossSpeechStreamingPipelineConfig):
        assert cls(model_path="local").codec_final_bias_elision is None
    with pytest.raises(ValueError):
        MossSpeechPipelineConfig(model_path="local", codec_final_bias_elision=True)
    config = MossSpeechStreamingPipelineConfig(
        model_path="local",
        codec_final_bias_elision=True,
        codec_final_bias_alignment=True,
        codec_optimizations=True,
    )
    merged = ConfigManager(config).merge_config({"codec_final_bias_elision": False})
    stage = next(s for s in merged.stages if s.name == "audio_vocoder")
    assert stage.factory_args["codec_final_bias_elision"] is False
    assert stage.factory_args["codec_final_bias_alignment"] is True
    assert config.codec_final_bias_elision is True


@pytest.mark.parametrize("enabled", [False, True])
def test_real_owner_construction_and_close(enabled):
    decoder = object.__new__(CausalConditionalDecoder)
    torch.nn.Module.__init__(decoder)
    codec = streaming_codec.MossSpeechStreamingCodec(
        None,
        device="cpu",
        chunk_size=25,
        codec_optimizations=True,
        codec_final_bias_alignment=True,
        codec_final_bias_elision=enabled,
        flow=SimpleNamespace(decoder=SimpleNamespace(estimator=decoder)),
        hift=object(),
    )
    owner = codec._final_bias_alignment
    assert owner is decoder._final_bias_alignment
    assert owner.status()["elide_zero_bias"] is enabled
    codec.close()
    codec.close()
    assert owner.status()["closed"] is True


@pytest.mark.parametrize("value", [True, 1, "true", None])
def test_invalid_elision_rejected_before_loading(value):
    with pytest.raises(ValueError):
        streaming_codec.MossSpeechStreamingCodec(
            None,
            device="cpu",
            chunk_size=25,
            codec_optimizations=True,
            codec_final_bias_elision=value,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_stage_factory_passes_only_enabled_option(monkeypatch, enabled):
    created = []

    def create(*args, **kwargs):
        created.append(kwargs)
        return SimpleNamespace(hift=object(), close=Mock())

    monkeypatch.setattr(streaming_codec, "MossSpeechStreamingCodec", create)
    monkeypatch.setattr(stages, "_resolve_codec_dir", lambda *a: ".")
    monkeypatch.setattr(
        stages,
        "create_audio_vocoder_executor",
        lambda *a, **k: SimpleNamespace(_fn=object()),
    )
    monkeypatch.setattr(
        streaming, "MossSpeechStreamingVocoder", lambda codec, *a, **k: codec
    )
    streaming.create_streaming_audio_executor(
        "local",
        chunk_size=25,
        codec_optimizations=True,
        codec_final_bias_alignment=True,
        codec_final_bias_elision=enabled,
    )
    assert len(created) == 1
    assert created[0].get("codec_final_bias_elision", False) is enabled
    if not enabled:
        assert "codec_final_bias_elision" not in created[0]


def test_stage_rejects_elision_without_alignment(monkeypatch):
    monkeypatch.setattr(
        stages,
        "_resolve_codec_dir",
        Mock(side_effect=AssertionError("loaded before validation")),
    )
    with pytest.raises(ValueError):
        streaming.create_streaming_audio_executor(
            "local",
            chunk_size=25,
            codec_optimizations=True,
            codec_final_bias_elision=True,
        )
