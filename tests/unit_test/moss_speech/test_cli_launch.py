"""Exercise public CLI parsing and model asset wiring without starting workers."""

from __future__ import annotations

import importlib
import json
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


@pytest.mark.parametrize("streaming", [False, True])
def test_serve_cli_accepts_assets_without_generated_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": ["MossSpeechForCausalLM"]})
    )
    launch = Mock()
    monkeypatch.setattr(
        importlib.import_module("sglang_omni.cli.serve"), "launch_server", launch
    )
    args = [
        "serve",
        "--model-path",
        str(model),
        "--codec-path",
        str(tmp_path / "codec"),
        "--voice-wav",
        str(tmp_path / "voice.wav"),
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
    ]
    if streaming:
        args += ["--config", str(REPO / "examples/configs/moss_speech_streaming.yaml")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    config = launch.call_args.args[0]
    assert type(config) is (
        MossSpeechStreamingPipelineConfig if streaming else MossSpeechPipelineConfig
    )
    assert config.model_path == str(model)
    for stage in config.stages:
        if stage.name != "text_decode":
            assert stage.factory_args["codec_path"] == str(tmp_path / "codec")
        else:
            assert "codec_path" not in stage.factory_args
        if stage.name == "preprocessing":
            assert stage.factory_args["voice_wav"] == str(tmp_path / "voice.wav")
    ar = next(s for s in config.stages if s.name == "ar_engine")
    assert bool(ar.stream_to) == streaming
    assert ar.factory_args["server_args_overrides"]["max_total_tokens"] == 4096


@pytest.mark.parametrize(
    "cls", [MossSpeechPipelineConfig, MossSpeechStreamingPipelineConfig]
)
def test_cli_asset_overrides_survive_config_roundtrip_without_cross_instance_leak(
    cls: type,
) -> None:
    initial = cls(model_path="model", codec_path="first-codec", voice_wav="first.wav")
    manager = ConfigManager(initial)
    updated = manager.merge_config(
        manager.parse_extra_args(
            [
                "--codec-path",
                "second-codec",
                "--voice-wav",
                "second.wav",
            ]
        )
    )
    restored = cls.model_validate_json(updated.model_dump_json())
    for stage in restored.stages:
        if stage.name != "text_decode":
            assert stage.factory_args["codec_path"] == "second-codec"
    assert restored.stages[0].factory_args["voice_wav"] == "second.wav"
    assert initial.stages[0].factory_args["codec_path"] == "first-codec"
    fresh = cls(model_path="model")
    assert "codec_path" not in fresh.stages[0].factory_args
    # Existing fully rendered configs still own their explicit stage arguments.
    fresh.stages[0].factory_args.update(codec_path="legacy", voice_wav="legacy.wav")
    legacy = ConfigManager(fresh).merge_config({})
    assert legacy.stages[0].factory_args["codec_path"] == "legacy"
    assert legacy.stages[0].factory_args["voice_wav"] == "legacy.wav"


def test_both_console_names_use_the_same_app() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    project = tomllib.loads((REPO / "pyproject.toml").read_text())
    scripts = project["project"]["scripts"]
    assert scripts["sglang-omni"] == scripts["sgl-omni"] == "sglang_omni.cli:app"


@pytest.mark.parametrize("name", ["moss_speech", "moss_speech_streaming"])
def test_shipped_configs_have_no_operator_specific_paths(name: str) -> None:
    config = ConfigManager.from_file(str(REPO / f"examples/configs/{name}.yaml")).config
    assert not Path(config.model_path).is_absolute()
    assert not config.env_defaults
    assert "/remote-home" not in config.endpoints.base_path
