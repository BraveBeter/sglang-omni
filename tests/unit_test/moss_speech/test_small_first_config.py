"""Explicit experimental codec settings must reach the streaming factory only."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech.config import (
    MossSpeechPipelineConfig,
    MossSpeechStreamingPipelineConfig,
)


def vocoder(config):
    return next(s for s in config.stages if s.name == "audio_vocoder")


def test_default_inheritance_and_explicit_disable():
    base = MossSpeechStreamingPipelineConfig(
        model_path="local", codec_optimizations=True
    )
    assert base.small_first_chunk in (False, None)
    assert base.codec_prefix_cache_manifest is None
    assert "small_first_chunk" not in vocoder(base).factory_args
    vocoder(base).factory_args["small_first_chunk"] = True
    inherited = ConfigManager(base).merge_config({})
    assert vocoder(inherited).factory_args["small_first_chunk"] is True
    disabled = ConfigManager(inherited).merge_config({"small_first_chunk": False})
    assert vocoder(disabled).factory_args["small_first_chunk"] is False
    assert vocoder(inherited).factory_args["small_first_chunk"] is True


def test_explicit_settings_reach_only_streaming_vocoder():
    config = MossSpeechStreamingPipelineConfig(
        model_path="local",
        small_first_chunk=True,
        codec_prefix_cache_manifest="prepared.json",
    )
    assert vocoder(config).factory_args["small_first_chunk"] is True
    assert (
        vocoder(config).factory_args["codec_prefix_cache_manifest"] == "prepared.json"
    )
    assert all(
        "small_first_chunk" not in s.factory_args
        and "codec_prefix_cache_manifest" not in s.factory_args
        for s in config.stages
        if s.name != "audio_vocoder"
    )
    for options in (
        {"small_first_chunk": True},
        {"codec_prefix_cache_manifest": "prepared.json"},
    ):
        with pytest.raises(ValueError, match="streaming"):
            MossSpeechPipelineConfig(model_path="local", **options)


def test_factory_prepares_explicit_cache_before_return(monkeypatch, tmp_path):
    stages = import_module("sglang_omni.models.moss_speech.stages")
    module = import_module("sglang_omni.models.moss_speech.streaming")
    codec_module = import_module(
        "sglang_omni.models.moss_speech.components.streaming_codec"
    )
    codec = Mock(hift=object())
    codec_factory = Mock(return_value=codec)
    monkeypatch.setattr(codec_module, "MossSpeechStreamingCodec", codec_factory)
    monkeypatch.setattr(
        stages,
        "create_audio_vocoder_executor",
        Mock(return_value=SimpleNamespace(_fn=None)),
    )
    result = module.create_streaming_audio_executor(
        "local",
        codec_path=str(tmp_path),
        chunk_size=25,
        codec_optimizations=True,
        small_first_chunk=True,
        codec_prefix_cache_manifest="prepared.json",
    )
    assert result.codec is codec
    assert codec_factory.call_args.kwargs["small_first_chunk"] is True
    codec.prepare_prefix_cache.assert_called_once_with("prepared.json")
    codec.prepare_prefix_cache.reset_mock()
    module.create_streaming_audio_executor(
        "local", codec_path=str(tmp_path), codec_optimizations=True
    )
    codec.prepare_prefix_cache.assert_not_called()
