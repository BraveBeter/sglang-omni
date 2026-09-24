"""Independent prefill graphs preserve every startup-policy setting."""

import importlib
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from sglang_omni.cli import app
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech.config import (
    MossSpeechPipelineConfig,
    MossSpeechStreamingPipelineConfig,
)

REPO = Path(__file__).resolve().parents[3]


def stage(config, name):
    return next(s for s in config.stages if s.name == name)


@pytest.mark.parametrize(
    "cls", [MossSpeechPipelineConfig, MossSpeechStreamingPipelineConfig]
)
def test_prefill_default_omitted_without_changing_decode_or_sampler(cls):
    config = cls(model_path="local", decode_full_graphs=True)
    assert config.prefill_full_graphs is None
    assert "prefill_full_graphs" not in stage(config, "ar_engine").factory_args
    assert stage(config, "ar_engine").factory_args["decode_full_graphs"] is True
    assert not stage(config, "ar_engine").factory_args.get(
        "sampling_optimizations", False
    )


@pytest.mark.parametrize("policy", [None, "6_8_align25"])
@pytest.mark.parametrize("enabled", [False, True])
def test_config_merge_preserves_policy_and_explicit_prefill_disable(policy, enabled):
    config = MossSpeechStreamingPipelineConfig(
        model_path="local",
        stream_startup_schedule=policy,
        prefill_full_graphs=enabled,
        decode_full_graphs=True,
    )
    assert stage(config, "ar_engine").factory_args["prefill_full_graphs"] is enabled
    assert (
        stage(config, "audio_vocoder").factory_args.get("stream_startup_schedule")
        == policy
    )
    before = config.model_dump()
    changed = ConfigManager(config).merge_config({"prefill_full_graphs": not enabled})
    assert config.model_dump() == before
    roundtrip = MossSpeechStreamingPipelineConfig.model_validate_json(
        changed.model_dump_json()
    )
    assert (
        stage(roundtrip, "ar_engine").factory_args["prefill_full_graphs"] is not enabled
    )
    assert stage(roundtrip, "ar_engine").factory_args["decode_full_graphs"] is True
    assert (
        stage(roundtrip, "audio_vocoder").factory_args
        == stage(config, "audio_vocoder").factory_args
    )
    assert all(
        "prefill_full_graphs" not in s.factory_args
        for s in changed.stages
        if s.name != "ar_engine"
    )


def test_original_small_first_policy_is_preserved():
    config = MossSpeechStreamingPipelineConfig(
        model_path="local", small_first_chunk=True, prefill_full_graphs=True
    )
    assert stage(config, "audio_vocoder").factory_args["small_first_chunk"] is True
    assert config.stream_startup_schedule is None
    assert stage(config, "ar_engine").factory_args["prefill_full_graphs"] is True


def test_public_dynamic_cli_keeps_both_independent_fields(monkeypatch):
    launch = Mock()
    monkeypatch.setattr(
        importlib.import_module("sglang_omni.cli.serve"), "launch_server", launch
    )
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--config",
            str(REPO / "examples/configs/moss_speech_streaming.yaml"),
            "--prefill-full-graphs",
            "true",
            "--stream-startup-schedule",
            "6_8_align25",
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    config = launch.call_args.args[0]
    assert stage(config, "ar_engine").factory_args["prefill_full_graphs"] is True
    assert (
        stage(config, "audio_vocoder").factory_args["stream_startup_schedule"]
        == "6_8_align25"
    )
    assert not stage(config, "audio_vocoder").factory_args.get(
        "small_first_chunk", False
    )
