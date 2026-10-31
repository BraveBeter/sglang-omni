# SPDX-License-Identifier: Apache-2.0
"""Configuration and startup contracts for the eager MOSS-Speech pipeline."""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import pytest
import torch
from pydantic import ValidationError
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from typer.testing import CliRunner

from sglang_omni.cli import app
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_factory_signature_args,
    resolve_stage_factory_arg_defaults,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig
from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig
from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.models.moss_speech.request_data import build_sglang_moss_request
from sglang_omni.models.moss_speech.stages import (
    create_ar_engine_executor,
    create_preprocessing_executor,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DISCOVERY_PROGRAM = """
import builtins
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import pkgutil
import sys

import torch
from torch.utils import cpp_extension

def forbidden(*parameters, **keywords):
    raise AssertionError("Unexpected CUDA initialization, native build, or interactive input")

assert not torch.cuda.is_initialized()
torch.cuda._lazy_init = forbidden
cpp_extension.load = forbidden
cpp_extension.load_inline = forbidden
input_requests = []

def forbidden_input(*parameters, **keywords):
    input_requests.append(True)
    raise AssertionError("Interactive remote-code confirmation is forbidden")

builtins.input = forbidden_input

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig

checkpoint_path = sys.argv[1]
assert resolve_config_cls_for_model_path(checkpoint_path) is MossSpeechPipelineConfig
assert not input_requests, "Metadata discovery attempted interactive remote-code confirmation"
assert PIPELINE_CONFIG_REGISTRY.get_config("MossSpeechForCausalLM") is MossSpeechPipelineConfig
package = importlib.import_module("sglang_omni.models.moss_speech")
for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
    importlib.import_module(module.name)

from sglang.srt.models.registry import ModelRegistry
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel
from sglang_omni.models.moss_speech.hf_config import ensure_moss_speech_config_registered
from transformers import AutoConfig

SGLModelRunner.register_omni_model(None)
assert ModelRegistry.models["MossSpeechForCausalLM"] is MossSpeechSGLangModel
ensure_moss_speech_config_registered()
ensure_moss_speech_config_registered()
configuration = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=False)
assert configuration.num_hidden_layers == 36
assert configuration.audio_vocab_size == 16512
assert not torch.cuda.is_initialized()
print("MOSS_DISCOVERY_WITHOUT_DEVICE_SIDE_EFFECTS")
"""


@pytest.mark.parametrize("filename", ["moss_speech.yaml", "moss_speech_dual_gpu.yaml"])
def test_examples_bind_real_factories_and_survive_roundtrip(filename: str) -> None:
    config = ConfigManager.from_file(
        str(PROJECT_ROOT / "examples/configs" / filename)
    ).config
    restored = MossSpeechPipelineConfig.model_validate(config.model_dump())
    for stage in restored.stages:
        module_name, function_name = stage.factory_path.rsplit(".", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        keywords = resolve_factory_signature_args(
            factory,
            resolve_stage_typed_kwargs(stage),
            defaults=resolve_stage_factory_arg_defaults(stage, restored),
        )
        inspect.signature(factory).bind(**keywords)
        for asset_name in ("codec_path", "voice_wav"):
            asset_path = keywords.get(asset_name)
            if isinstance(asset_path, str):
                assert not Path(asset_path).is_absolute()
            else:
                pass
    assert not Path(restored.model_path).is_absolute()
    codec_gpu = 1 if filename.endswith("dual_gpu.yaml") else 0
    assert restored.stage_named("preprocessing").gpu == codec_gpu
    assert restored.stage_named("audio_vocoder").gpu == codec_gpu
    assert restored.stage_named("ar_engine").gpu == 0


def test_cli_overrides_survive_serialization_without_cross_instance_leak() -> None:
    baseline = MossSpeechPipelineConfig(model_path="model")
    manager = ConfigManager(baseline)
    updated = manager.merge_config(
        manager.parse_extra_args(
            [
                "--preprocessing.factory.voice_wav",
                "custom.wav",
                "--preprocessing.factory.codec_path",
                "codec",
                "--audio_vocoder.factory.chunk_size",
                "5",
                "--ar_engine.engine.max_running_requests",
                "2",
            ]
        )
    )
    restored = MossSpeechPipelineConfig.model_validate(updated.model_dump())
    assert (
        resolve_stage_typed_kwargs(restored.stage_named("preprocessing"))["voice_wav"]
        == "custom.wav"
    )
    assert (
        resolve_stage_typed_kwargs(restored.stage_named("preprocessing"))["codec_path"]
        == "codec"
    )
    assert (
        resolve_stage_typed_kwargs(restored.stage_named("audio_vocoder"))["chunk_size"]
        == 5
    )
    assert restored.stage_named("ar_engine").engine.max_running_requests == 2
    assert baseline == MossSpeechPipelineConfig(model_path="model")


def test_public_serve_cli_routes_model_assets_to_typed_stages(tmp_path: Path) -> None:
    MossSpeechConfig(
        audio_vocab_size=16512, architectures=["MossSpeechForCausalLM"]
    ).to_json_file(tmp_path / "config.json")
    voice_path = str(tmp_path / "voice.wav")
    codec_path = str(tmp_path / "codec")
    with patch("sglang_omni.cli.serve.launch_server") as launch:
        result = CliRunner().invoke(
            app,
            [
                "serve",
                "--model-path",
                str(tmp_path),
                "--preprocessing.factory.voice_wav",
                voice_path,
                "--preprocessing.factory.codec_path",
                codec_path,
                "--audio_vocoder.factory.codec_path",
                codec_path,
                "--ar_engine.engine.max_running_requests",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output + repr(result.exception)
        config = launch.call_args.args[0]
        assert isinstance(config, MossSpeechPipelineConfig)
        assert config.model_path == str(tmp_path)
        assert (
            resolve_stage_typed_kwargs(config.stage_named("preprocessing"))["voice_wav"]
            == voice_path
        )
        assert (
            resolve_stage_typed_kwargs(config.stage_named("audio_vocoder"))[
                "codec_path"
            ]
            == codec_path
        )
        assert config.stage_named("ar_engine").engine.max_running_requests == 2


def test_fresh_process_discovers_both_registries_without_device_side_effects(
    tmp_path: Path,
) -> None:
    """Discovery previously hid an interactive custom-config prompt behind prior registration."""
    configuration = MossSpeechConfig(
        audio_vocab_size=16512,
        architectures=["MossSpeechForCausalLM"],
        auto_map={"AutoConfig": "configuration_moss_speech.MossSpeechConfig"},
    )
    configuration.to_json_file(tmp_path / "config.json")
    environment = dict(os.environ)
    environment.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=""
    )
    process = subprocess.run(
        [sys.executable, "-c", DISCOVERY_PROGRAM, str(tmp_path)],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert process.returncode == 0, process.stdout[-3000:] + process.stderr[-3000:]
    assert "MOSS_DISCOVERY_WITHOUT_DEVICE_SIDE_EFFECTS" in process.stdout


@pytest.mark.parametrize("scaling", [None, {"rope_type": "linear", "factor": 2.0}])
def test_hf_rope_parameters_roundtrip(scaling: dict[str, str | float] | None) -> None:
    configuration = MossSpeechConfig(rope_scaling=scaling)
    restored = MossSpeechConfig.from_dict(configuration.to_dict())
    assert restored.rope_theta == configuration.rope_theta
    assert restored.rope_parameters == configuration.rope_parameters
    assert (
        restored.num_shared_layers + restored.num_modality_layers
        == restored.num_hidden_layers
    )


def test_hf_rope_rejects_missing_scaling_factor() -> None:
    with pytest.raises(KeyError, match="factor"):
        MossSpeechConfig(rope_scaling={"rope_type": "linear"})


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("attention_backend", "triton", "attention_backend"),
        ("trust_remote_code", True, "trust_remote_code"),
        ("tp_size", 2, "tp_size"),
        ("disable_cuda_graph", False, "disable_cuda_graph"),
        ("enable_torch_compile", True, "enable_torch_compile"),
        ("disable_radix_cache", False, "disable_radix_cache"),
        ("chunked_prefill_size", 128, "chunked_prefill_size"),
        ("dtype", "float16", "BF16 model"),
        ("kv_cache_dtype", "fp8_e4m3", "BF16 KV"),
        ("quantization", "fp8", "quantization"),
    ],
)
def test_engine_rejects_unsupported_execution_policy(
    name: str, value: str | int | bool, message: str
) -> None:
    config = MossSpeechPipelineConfig(model_path="unused")
    builder = MossSpeechEngineBuilder(streaming=True)
    builder.server_args_overrides = config.stage_named("ar_engine").engine.overrides()
    overrides = dict(builder.generation_defaults(dtype="bfloat16"))
    overrides[name] = value
    with pytest.raises(ValueError, match=message):
        builder.adjust_overrides(overrides)


@pytest.mark.parametrize(
    "stage_name,name,value",
    [
        ("ar_engine", "decode_full_graphs", True),
        ("audio_vocoder", "codec_optimizations", True),
        ("audio_vocoder", "chunk_size", 7),
        ("preprocessing", "context_limit", 40961),
    ],
)
def test_typed_config_rejects_obsolete_or_untrained_knobs(
    stage_name: str, name: str, value: int | bool
) -> None:
    config = MossSpeechPipelineConfig(model_path="unused")
    document = config.model_dump()
    for stage in document["stages"]:
        if stage["name"] == stage_name:
            stage["factory"][name] = value
        else:
            pass
    with pytest.raises(ValidationError):
        MossSpeechPipelineConfig.model_validate(document)


def test_ar_factory_preflights_and_forwards_placement_budget(tmp_path: Path) -> None:
    MossSpeechConfig(audio_vocab_size=16512).to_json_file(tmp_path / "config.json")
    with patch.object(MossSpeechEngineBuilder, "build", autospec=True) as build:
        build.return_value = sentinel.scheduler
        scheduler = create_ar_engine_executor(
            str(tmp_path),
            dtype="bfloat16",
            context_length=512,
            gpu_id=1,
            total_gpu_memory_fraction=0.6,
            process_total_gpu_memory_fraction=0.7,
            server_args_overrides={"max_running_requests": 2},
        )
        assert scheduler is sentinel.scheduler
        builder = build.call_args.args[0]
        assert builder.context_length == 512
        assert build.call_args.kwargs["gpu_id"] == 1
        assert build.call_args.kwargs["total_gpu_memory_fraction"] == 0.7
        assert build.call_args.kwargs["server_args_overrides"] == {
            "max_running_requests": 2,
            "mem_fraction_static": 0.6,
            "trust_remote_code": False,
        }
        with pytest.raises(ValueError, match="BF16"):
            create_ar_engine_executor(
                str(tmp_path), dtype="float16", context_length=512
            )
        assert build.call_count == 1


def test_missing_voice_fails_before_loading_codec(tmp_path: Path) -> None:
    with patch("sglang_omni.models.moss_speech.stages.MossSpeechCodecAdapter") as codec:
        with pytest.raises(FileNotFoundError, match="voice_wav"):
            create_preprocessing_executor(
                str(tmp_path),
                encode_batch_size=4,
                context_limit=512,
                voice_wav=str(tmp_path / "missing.wav"),
            )
        codec.assert_not_called()


def test_context_window_defaults_and_explicit_overrides() -> None:
    config = MossSpeechPipelineConfig(model_path="unused")
    assert config.stage_named("preprocessing").factory.context_limit == 40960
    assert config.stage_named("ar_engine").factory.context_length == 40960
    assert config.stage_named("ar_engine").engine.max_total_tokens == 40962
    assert MossSpeechConfig().max_position_embeddings == 40960
    document = config.model_dump()
    for stage in document["stages"]:
        if stage["name"] == "ar_engine":
            stage["factory"]["context_length"] = 40961
        else:
            pass
    with pytest.raises(ValidationError):
        MossSpeechPipelineConfig.model_validate(document)


@pytest.mark.parametrize("prompt_tokens,budget", [(40959, 1), (39960, 1000)])
def test_context_window_survives_upstream_generation_and_kv_guards(
    prompt_tokens: int, budget: int
) -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler.model_config = SimpleNamespace(context_len=40960)
    scheduler.max_total_num_tokens = 40962
    scheduler.model_runner = Mock()
    scheduler.kv_shard_widening = scheduler.page_size = 1
    scheduler.max_new_tokens_limit = scheduler.sliding_window_size = (
        scheduler.draft_worker
    ) = None
    scheduler.chunked_prefill_size = -1
    scheduler.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        size=40962, dtype=torch.bfloat16, device="cpu", kvcache=None, need_sort=False
    )
    MossSpeechEngineBuilder().post_scheduler_setup(scheduler, scheduler.model_runner)
    state = MossSpeechState(
        input_grid=[[12, 512]] * prompt_tokens, max_new_tokens=budget
    )
    request = build_sglang_moss_request(
        state, payload=StagePayload("full-window", OmniRequest(inputs=None), {})
    )
    assert scheduler.prepare_request_limits(request) is None
    assert (
        request.req.sampling_params.max_new_tokens == request.max_new_tokens == budget
    )
    assert scheduler.request_kv_capacity_error(request.req) is None
