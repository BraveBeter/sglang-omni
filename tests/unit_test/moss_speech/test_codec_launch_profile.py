"""Qualify codec preparation in shipped launch profiles without loading weights."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from typer.testing import CliRunner

from sglang_omni.cli import app
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech.config import (
    MossSpeechPipelineConfig,
    MossSpeechStreamingPipelineConfig,
)

REPO = Path(__file__).resolve().parents[3]
PROFILES = ("moss_speech", "moss_speech_streaming")


def load_profile(name: str) -> ConfigManager:
    return ConfigManager.from_file(str(REPO / f"examples/configs/{name}.yaml"))


def assert_other_optimizations_disabled(config: MossSpeechPipelineConfig) -> None:
    assert config.stream_decode_fairness in (None, False)
    assert config.codec_memory_optimizations in (None, False)
    assert config.sampling_optimizations in (None, False)
    for stage in config.stages:
        assert not stage.factory_args.get("stream_decode_fairness", False)
        assert not stage.factory_args.get("codec_memory_optimizations", False)
        assert not stage.factory_args.get("sampling_optimizations", False)


@pytest.mark.parametrize("name", PROFILES)
def test_shipped_profiles_enable_only_codec_preparation(name: str) -> None:
    config = load_profile(name).config
    assert config.codec_optimizations is True
    for stage in config.stages:
        assert bool(stage.factory_args.get("codec_optimizations")) == (
            stage.name == "audio_vocoder"
        )
    assert_other_optimizations_disabled(config)


@pytest.mark.parametrize("name", PROFILES)
@pytest.mark.parametrize("disable", [False, True])
def test_profile_codec_flag_reaches_actual_vocoder_factory(
    name: str, disable: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_profile(name)
    config = manager.merge_config(
        manager.parse_extra_args(["--codec-optimizations", "false"]) if disable else {}
    )
    stage = next(stage for stage in config.stages if stage.name == "audio_vocoder")
    stages = importlib.import_module("sglang_omni.models.moss_speech.stages")
    streaming_codec = importlib.import_module(
        "sglang_omni.models.moss_speech.components.streaming_codec"
    )
    adapter_factory = Mock(return_value=SimpleNamespace(_device=torch.device("cpu")))
    codec = SimpleNamespace(hift=object())
    codec_factory = Mock(return_value=codec)
    monkeypatch.setattr(stages, "MossSpeechCodecAdapter", adapter_factory)
    monkeypatch.setattr(streaming_codec, "MossSpeechStreamingCodec", codec_factory)
    module_name, factory_name = stage.factory.rsplit(".", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    factory("unused-local-model", codec_path=str(tmp_path), **stage.factory_args)
    adapter_factory.assert_called_once()
    options = adapter_factory.call_args.kwargs
    assert options["codec_optimizations"] is (not disable)
    assert options["codec_memory_optimizations"] is False
    if name.endswith("streaming"):
        codec_factory.assert_called_once()
        assert codec_factory.call_args.kwargs["codec_optimizations"] is (not disable)
        assert codec_factory.call_args.kwargs["codec_memory_optimizations"] is False
        assert codec_factory.call_args.kwargs["chunk_size"] == 5
        assert options["shared_hift"] is (None if disable else codec.hift)
    else:
        codec_factory.assert_not_called()


@pytest.mark.parametrize("name", PROFILES)
def test_explicit_disable_survives_profile_serialization(name: str) -> None:
    manager = load_profile(name)
    enabled = manager.merge_config({"codec_optimizations": True})
    disabled = ConfigManager(enabled).merge_config(
        manager.parse_extra_args(["--codec-optimizations", "false"])
    )
    restored = type(disabled).model_validate_json(disabled.model_dump_json())
    assert restored.codec_optimizations is False
    assert (
        next(
            stage for stage in restored.stages if stage.name == "audio_vocoder"
        ).factory_args["codec_optimizations"]
        is False
    )
    assert (
        next(
            stage for stage in enabled.stages if stage.name == "audio_vocoder"
        ).factory_args["codec_optimizations"]
        is True
    )
    assert_other_optimizations_disabled(restored)


@pytest.mark.parametrize("name", PROFILES)
@pytest.mark.parametrize("enabled", [False, True])
def test_public_serve_boolean_override_reaches_profile(
    name: str, enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = Mock()
    monkeypatch.setattr(
        importlib.import_module("sglang_omni.cli.serve"), "launch_server", launch
    )
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--config",
            str(REPO / f"examples/configs/{name}.yaml"),
            "--codec-optimizations",
            str(enabled).lower(),
            "--host",
            "127.0.0.1",
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    launch.assert_called_once()
    config = launch.call_args.args[0]
    assert config.codec_optimizations is enabled
    assert (
        next(
            stage for stage in config.stages if stage.name == "audio_vocoder"
        ).factory_args["codec_optimizations"]
        is enabled
    )
    assert_other_optimizations_disabled(config)


@pytest.mark.parametrize(
    "cls", [MossSpeechPipelineConfig, MossSpeechStreamingPipelineConfig]
)
def test_bare_model_defaults_remain_opt_in(cls: type) -> None:
    config = cls(model_path="unused-local-model")
    assert config.codec_optimizations is None
    assert all(
        "codec_optimizations" not in stage.factory_args for stage in config.stages
    )
    assert_other_optimizations_disabled(config)
