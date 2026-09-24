"""Complete CFM graphs are opt-in, source-local startup work, never defaults."""

from __future__ import annotations

from copy import deepcopy

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech.config import (
    MossSpeechPipelineConfig,
    MossSpeechStreamingPipelineConfig,
)

# ruff: noqa: F811 -- imported pytest fixture names are injected as arguments


def vocoder(config):
    return next(stage for stage in config.stages if stage.name == "audio_vocoder")


def test_unset_and_legacy_serialized_config_preserve_default():
    config = MossSpeechStreamingPipelineConfig(model_path="local")
    assert config.codec_qkv_multistream is None
    assert "codec_qkv_multistream" not in vocoder(config).factory_args
    legacy = config.model_dump()
    legacy.pop("codec_qkv_multistream")
    restored = MossSpeechStreamingPipelineConfig(**legacy)
    assert restored.codec_qkv_multistream is None
    assert restored.model_dump() == config.model_dump()


def test_inherited_stage_value_and_explicit_false_with_real_merge():
    config = MossSpeechStreamingPipelineConfig(model_path="local")
    vocoder(config).factory_args["codec_qkv_multistream"] = True
    inherited = ConfigManager(config).merge_config({})
    assert inherited.codec_qkv_multistream is None
    assert vocoder(inherited).factory_args["codec_qkv_multistream"] is True
    disabled = ConfigManager(inherited).merge_config({"codec_qkv_multistream": False})
    assert disabled.codec_qkv_multistream is False
    assert vocoder(disabled).factory_args["codec_qkv_multistream"] is False
    assert vocoder(inherited).factory_args["codec_qkv_multistream"] is True


def test_enabled_only_reaches_streaming_factory_without_memory_override():
    base = MossSpeechStreamingPipelineConfig(model_path="local")
    for stage in base.stages:
        stage.factory_args["user_preserved_marker"] = stage.name
        if stage.name == "ar_engine":
            stage.factory_args["mem_fraction_static"] = 0.37
    vocoder(base).factory_args.update(chunk_size=25, gpu_id=1)
    before = deepcopy(base.model_dump())
    enabled = ConfigManager(base).merge_config({"codec_qkv_multistream": True})
    assert enabled.codec_qkv_multistream is True
    assert vocoder(enabled).factory_args["codec_qkv_multistream"] is True
    after = enabled.model_dump()
    after["codec_qkv_multistream"] = None
    next(s for s in after["stages"] if s["name"] == "audio_vocoder")[
        "factory_args"
    ].pop("codec_qkv_multistream")
    assert after == before
    assert base.model_dump() == before


@pytest.mark.parametrize(
    "options", ({}, {"codec_qkv_multistream": None}, {"codec_qkv_multistream": False})
)
def test_offline_unset_or_false_does_not_add_unknown_factory_argument(options):
    config = MossSpeechPipelineConfig(model_path="local", **options)
    assert "codec_qkv_multistream" not in vocoder(config).factory_args


def test_offline_enabled_is_rejected():
    with pytest.raises(ValueError, match="codec_qkv_multistream.*streaming"):
        MossSpeechPipelineConfig(model_path="local", codec_qkv_multistream=True)


def test_custom_nonstreaming_vocoder_cannot_receive_graph_option():
    base = MossSpeechStreamingPipelineConfig(model_path="local")
    vocoder(base).factory = "custom.codec_factory"
    with pytest.raises(ValueError, match="codec_qkv_multistream.*streaming"):
        ConfigManager(base).merge_config({"codec_qkv_multistream": True})


from .test_codec_full_graph_config import mocked_factory  # noqa: F401,E402


@pytest.mark.parametrize(
    "changes",
    (
        {"codec_full_graphs": False},
        {"chunk_size": 5},
        {"codec_optimizations": False},
        {"codec_prefix_cache_manifest": None},
    ),
)
def test_invalid_selection_fails_before_load(mocked_factory, changes):
    options = dict(
        chunk_size=25,
        codec_optimizations=True,
        codec_prefix_cache_manifest="prepared.json",
        codec_full_graphs=True,
        codec_qkv_multistream=True,
    )
    options.update(changes)
    with pytest.raises(ValueError):
        mocked_factory.module.create_streaming_audio_executor("local", **options)
    mocked_factory.resolve.assert_not_called()
    mocked_factory.build_codec.assert_not_called()


@pytest.mark.parametrize("enabled", (False, True))
def test_only_explicit_flag_reaches_codec_and_ready_precedes_scheduler(
    mocked_factory, enabled
):
    mocked_factory.module.create_streaming_audio_executor(
        "local",
        chunk_size=25,
        codec_optimizations=True,
        codec_prefix_cache_manifest="prepared.json",
        codec_full_graphs=True,
        codec_qkv_multistream=enabled,
    )
    arguments = mocked_factory.build_codec.call_args.kwargs
    if enabled:
        assert arguments["codec_qkv_multistream"] is True
    else:
        assert "codec_qkv_multistream" not in arguments
    assert mocked_factory.events.index(("graphs",)) < mocked_factory.events.index(
        ("scheduler",)
    )
