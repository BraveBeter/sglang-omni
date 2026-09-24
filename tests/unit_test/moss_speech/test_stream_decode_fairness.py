"""Keep streaming fairness independent from exact codec preparation defaults."""

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
from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingVocoder

REPO = Path(__file__).resolve().parents[3]
CONFIGS = (MossSpeechPipelineConfig, MossSpeechStreamingPipelineConfig)


@pytest.mark.parametrize("codec_optimizations", [False, True])
def test_codec_preparation_does_not_implicitly_enable_fairness(
    codec_optimizations: bool,
) -> None:
    codec = SimpleNamespace(codec_optimizations=codec_optimizations)
    assert MossSpeechStreamingVocoder(codec, None)._defer_stream_decode is False
    assert (
        MossSpeechStreamingVocoder(
            codec, None, defer_stream_decode=None
        )._defer_stream_decode
        is False
    )
    assert (
        MossSpeechStreamingVocoder(
            codec, None, defer_stream_decode=True
        )._defer_stream_decode
        is True
    )


@pytest.mark.parametrize("cls", CONFIGS)
def test_fairness_default_and_serialized_stage_override(cls: type) -> None:
    config = cls(model_path="local", codec_optimizations=True)
    assert config.stream_decode_fairness in (None, False)
    stage = next(s for s in config.stages if s.name == "audio_vocoder")
    assert not stage.factory_args.get("stream_decode_fairness", False)
    stage.factory_args["stream_decode_fairness"] = True
    inherited = ConfigManager(config).merge_config({})
    assert (
        next(s for s in inherited.stages if s.name == "audio_vocoder").factory_args[
            "stream_decode_fairness"
        ]
        is True
    )
    disabled = ConfigManager(inherited).merge_config({"stream_decode_fairness": False})
    restored = cls.model_validate_json(disabled.model_dump_json())
    assert restored.stream_decode_fairness is False
    assert (
        next(s for s in restored.stages if s.name == "audio_vocoder").factory_args[
            "stream_decode_fairness"
        ]
        is False
    )
    assert (
        next(s for s in inherited.stages if s.name == "audio_vocoder").factory_args[
            "stream_decode_fairness"
        ]
        is True
    )


@pytest.mark.parametrize("profile", ["moss_speech", "moss_speech_streaming"])
@pytest.mark.parametrize("enabled", [False, True])
def test_public_serve_fairness_boolean_reaches_only_vocoder_stage(
    profile: str, enabled: bool, monkeypatch: pytest.MonkeyPatch
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
            str(REPO / f"examples/configs/{profile}.yaml"),
            "--stream-decode-fairness",
            str(enabled).lower(),
            "--codec-optimizations",
            "true",
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    config = launch.call_args.args[0]
    assert config.stream_decode_fairness is enabled
    assert config.codec_optimizations is True
    for stage in config.stages:
        if stage.name == "audio_vocoder":
            assert stage.factory_args["stream_decode_fairness"] is enabled
        else:
            assert "stream_decode_fairness" not in stage.factory_args


@pytest.mark.parametrize("cls", CONFIGS)
@pytest.mark.parametrize("codec_optimizations", [False, True])
@pytest.mark.parametrize("fairness", [False, True])
def test_actual_factory_accepts_independent_fairness_config(
    cls: type,
    codec_optimizations: bool,
    fairness: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cls(
        model_path="local",
        codec_optimizations=codec_optimizations,
        stream_decode_fairness=fairness,
    )
    stage = next(s for s in config.stages if s.name == "audio_vocoder")
    stages = importlib.import_module("sglang_omni.models.moss_speech.stages")
    codec_module = importlib.import_module(
        "sglang_omni.models.moss_speech.components.streaming_codec"
    )
    adapter_factory = Mock(return_value=SimpleNamespace(_device=torch.device("cpu")))
    codec_factory = Mock(
        return_value=SimpleNamespace(
            hift=object(),
            codec_optimizations=codec_optimizations,
        )
    )
    monkeypatch.setattr(stages, "MossSpeechCodecAdapter", adapter_factory)
    monkeypatch.setattr(codec_module, "MossSpeechStreamingCodec", codec_factory)
    module_name, factory_name = stage.factory.rsplit(".", 1)
    scheduler = getattr(importlib.import_module(module_name), factory_name)(
        "local", codec_path=str(tmp_path), **stage.factory_args
    )
    adapter_factory.assert_called_once()
    assert "stream_decode_fairness" not in adapter_factory.call_args.kwargs
    assert (
        adapter_factory.call_args.kwargs["codec_optimizations"] is codec_optimizations
    )
    if cls is MossSpeechStreamingPipelineConfig:
        assert scheduler._defer_stream_decode is fairness
        assert "stream_decode_fairness" not in codec_factory.call_args.kwargs
    else:
        codec_factory.assert_not_called()
