"""Explicit native startup policy: boundaries, draining and request ownership."""

import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech.components.streaming_codec import (
    CodecStreamSession,
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.config import (
    MossSpeechPipelineConfig,
    MossSpeechStreamingPipelineConfig,
)

POLICY = "6_8_align25"


class Flow:
    pre_lookahead_len = 3
    token_mel_ratio = 4

    def inference(self, *, token, finalize, **kwargs):
        count = token.shape[1] - (0 if finalize else 3)
        mel = torch.ones(1, 80, count * 4)
        return mel, mel


class Hift:
    def inference(self, *, speech_feat, cache_source):
        wave = torch.rand(1, speech_feat.shape[-1] * 480)
        return wave, wave.unsqueeze(1)


def voice(prompt=342):
    return dict(
        prompt_token=torch.zeros(1, prompt, dtype=torch.int32),
        prompt_feat=torch.zeros(1, prompt * 4, 80),
        embedding=torch.zeros(1, 192),
    )


def codec(**kwargs):
    return MossSpeechStreamingCodec(
        None, device="cpu", flow=Flow(), hift=Hift(), chunk_size=25, **kwargs
    )


def vocoder(config):
    return next(s for s in config.stages if s.name == "audio_vocoder")


def test_default_legacy_and_explicit_hops_all_prompt_residues():
    new = codec(stream_startup_schedule=POLICY)
    old = codec()
    small = codec(small_first_chunk=True)
    for prompt in range(325, 350):
        state = CodecStreamSession(voice(prompt), 7)
        aligned = 25 + (-prompt) % 25
        assert old.next_hop(state) == aligned
        expected_small = (-prompt) % 25 or 25
        if expected_small <= 2:
            expected_small += 25
        assert small.next_hop(state) == expected_small
        for cursor, expected in ((0, 6), (6, 8), (14, aligned - 14), (aligned, 25)):
            state.cursor = cursor
            assert new.next_hop(state) == expected
        for cursor in (6, 14, aligned):
            state.cursor = cursor
            assert old.next_hop(state) == small.next_hop(state) == 25


def test_schedule_length_and_residue_arithmetic_matrix():
    c = codec(stream_startup_schedule=POLICY)
    for prompt in range(325, 350):
        for length in range(1, 513):
            state = CodecStreamSession(voice(prompt), 7)
            sizes = []
            while length - state.cursor >= c.next_hop(state) + 3:
                hop = c.next_hop(state)
                sizes.append(hop * 1920 - (3840 if not sizes else 0))
                state.cursor += hop
            sizes.append((length - state.cursor + (2 if sizes else 0)) * 1920)
            assert min(sizes) > 0
            assert sum(sizes) == length * 1920
            assert state.cursor <= length - 3 or state.cursor == 0


@pytest.mark.parametrize("length", [1, 8, 9, 16, 17, 31, 35, 36, 61, 512])
@pytest.mark.parametrize("limit", [1, 2, None])
def test_native_burst_and_single_code_delivery_exact_with_bounded_final(length, limit):
    c = codec(stream_startup_schedule=POLICY)
    before = torch.get_rng_state().clone()
    c.begin("burst", voice(), seed=7)
    burst_state = c.sessions["burst"]
    expected = c.push("burst", [2] * length, final=True)
    c.begin("bounded", voice(), seed=7)
    actual_state = c.sessions["bounded"]
    actual = []
    for _ in range(length):
        waves = c.push("bounded", [2], max_hops=limit)
        assert limit is None or len(waves) <= limit
        actual.extend(waves)
    actual.extend(c.push("bounded", [], final=True, max_hops=limit))
    while "bounded" in c.sessions:
        assert c.sessions["bounded"].final_requested
        waves = c.push("bounded", [], max_hops=limit)
        assert waves and (limit is None or len(waves) <= limit)
        actual.extend(waves)
    assert len(actual) == len(expected)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert sum(w.numel() for w in actual) == length * 1920
    assert actual_state.ledger == burst_state.ledger
    assert torch.equal(actual_state.cpu_rng, burst_state.cpu_rng)
    assert torch.equal(before, torch.get_rng_state())
    assert c.full_graph_status() == c.prefix_cache_status() == {"enabled": False}
    if length == 31:
        assert [w.numel() for w in actual] == [7680, 15360, 36480]


@pytest.mark.parametrize("cancel_at", [0, 9, 17, 36])
def test_interleaving_two_seeds_and_cancel_at_each_startup_boundary(cancel_at):
    c = codec(stream_startup_schedule=POLICY)
    global_rng = torch.get_rng_state().clone()
    expected = {}
    for seed in (7, 19):
        c.begin(str(seed), voice(), seed=seed)
        expected[seed] = c.push(str(seed), [2] * 61, final=True)
    c.begin("a", voice(), seed=7)
    c.begin("b", voice(), seed=19)
    c.begin("cancel", voice(), seed=7)
    c.push("cancel", [3] * cancel_at)
    c.cleanup("cancel")
    c.cleanup("cancel")
    actual = {7: [], 19: []}
    for i in range(61):
        for rid, seed in (("a", 7), ("b", 19)):
            actual[seed].extend(c.push(rid, [2], final=i == 60, max_hops=1))
    for rid, seed in (("a", 7), ("b", 19)):
        while rid in c.sessions:
            actual[seed].extend(c.push(rid, [], max_hops=1))
        assert all(torch.equal(a, b) for a, b in zip(actual[seed], expected[seed]))
        assert len(actual[seed]) == len(expected[seed])
    assert not c.sessions and torch.equal(global_rng, torch.get_rng_state())


def test_policy_validation_and_sticky_final_cleanup():
    for options in (
        dict(stream_startup_schedule="unknown"),
        dict(stream_startup_schedule=True),
        dict(stream_startup_schedule=POLICY, small_first_chunk=True),
    ):
        with pytest.raises(ValueError, match="schedule"):
            codec(**options)
    with pytest.raises(ValueError, match="chunk25"):
        MossSpeechStreamingCodec(None, chunk_size=5, stream_startup_schedule=POLICY)
    c = codec(stream_startup_schedule=POLICY)
    c.begin("r", voice(), seed=7)
    assert len(c.push("r", [2] * 36, final=True, max_hops=1)) == 1
    with pytest.raises(ValueError, match="final"):
        c.push("r", [2], max_hops=1)
    assert not c.sessions


def test_typed_config_inheritance_and_streaming_only():
    base = MossSpeechStreamingPipelineConfig(model_path="local")
    assert base.stream_startup_schedule is None
    assert "stream_startup_schedule" not in vocoder(base).factory_args
    vocoder(base).factory_args["stream_startup_schedule"] = POLICY
    inherited = ConfigManager(base).merge_config({})
    assert vocoder(inherited).factory_args["stream_startup_schedule"] == POLICY
    selected = MossSpeechStreamingPipelineConfig(
        model_path="local", stream_startup_schedule=POLICY
    )
    assert vocoder(selected).factory_args["stream_startup_schedule"] == POLICY
    assert all(
        "stream_startup_schedule" not in s.factory_args
        for s in selected.stages
        if s.name != "audio_vocoder"
    )
    with pytest.raises(ValueError, match="streaming"):
        MossSpeechPipelineConfig(model_path="local", stream_startup_schedule=POLICY)
    with pytest.raises(ValueError):
        MossSpeechStreamingPipelineConfig(
            model_path="local", stream_startup_schedule="unknown"
        )


def test_factory_validates_before_weight_load_and_preserves_cleanup(
    monkeypatch, tmp_path
):
    module = import_module("sglang_omni.models.moss_speech.streaming")
    stages = import_module("sglang_omni.models.moss_speech.stages")
    codec_module = import_module(
        "sglang_omni.models.moss_speech.components.streaming_codec"
    )
    resolve = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(stages, "_resolve_codec_dir", resolve)
    for options in (dict(chunk_size=5), dict(chunk_size=25, small_first_chunk=True)):
        with pytest.raises(ValueError, match="schedule"):
            module.create_streaming_audio_executor(
                "local", stream_startup_schedule=POLICY, **options
            )
        resolve.assert_not_called()
    native = Mock(hift=object())
    constructor = Mock(return_value=native)
    monkeypatch.setattr(codec_module, "MossSpeechStreamingCodec", constructor)
    monkeypatch.setattr(
        stages,
        "create_audio_vocoder_executor",
        Mock(return_value=SimpleNamespace(_fn=None)),
    )
    executor = module.create_streaming_audio_executor(
        "local", chunk_size=25, codec_optimizations=True, stream_startup_schedule=POLICY
    )
    assert executor.codec is native
    assert constructor.call_args.kwargs["stream_startup_schedule"] == POLICY
    native.prepare_full_graphs.assert_not_called()
    native.prepare_prefix_cache.assert_not_called()
    native.prepare_prefix_cache.side_effect = RuntimeError("prepare failure")
    with pytest.raises(RuntimeError, match="prepare failure"):
        module.create_streaming_audio_executor(
            "local",
            chunk_size=25,
            codec_optimizations=True,
            stream_startup_schedule=POLICY,
            codec_prefix_cache_manifest="own.json",
        )
    native.close.assert_called_once()


def test_new_lengths_are_not_admitted_by_existing_prefix_binding():
    from sglang_omni.models.moss_speech.components.cfm_prefix_cache import CacheBinding

    binding = CacheBinding("voice", (1500,), {"runtime": "same"}, ())
    for length in (1392, 1424):
        assert not binding.matches("voice", length, False, {"runtime": "same"}, ())
        with pytest.raises(ValueError, match="measured shapes"):
            CacheBinding("voice", (length,), {}, ())
    assert binding.matches("voice", 1500, False, {"runtime": "same"}, ())


def test_native_dynamic_cli_forwards_explicit_policy_without_starting_workers(
    tmp_path, monkeypatch
):
    from typer.testing import CliRunner

    from sglang_omni.cli import app

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": ["MossSpeechForCausalLM"]})
    )
    launch = Mock()
    monkeypatch.setattr(import_module("sglang_omni.cli.serve"), "launch_server", launch)
    config = (
        Path(__file__).resolve().parents[3]
        / "examples/configs/moss_speech_streaming.yaml"
    )
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--model-path",
            str(model),
            "--config",
            str(config),
            "--stream-startup-schedule",
            POLICY,
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    actual = launch.call_args.args[0]
    assert actual.stream_startup_schedule == POLICY
    assert vocoder(actual).factory_args["stream_startup_schedule"] == POLICY


@pytest.mark.parametrize("length", [36, 512])
@pytest.mark.parametrize("limit", [1, 2])
def test_burst_final_drains_all_startup_hops_without_new_input(length, limit):
    c = codec(stream_startup_schedule=POLICY)
    c.begin("full", voice(), seed=19)
    expected_state = c.sessions["full"]
    expected = c.push("full", [2] * length, final=True)
    c.begin("bounded", voice(), seed=19)
    actual_state = c.sessions["bounded"]
    actual = c.push("bounded", [2] * length, final=True, max_hops=limit)
    assert actual_state.final_requested and len(actual) == limit
    while "bounded" in c.sessions:
        waves = c.push("bounded", [], max_hops=limit)
        assert 0 < len(waves) <= limit
        actual.extend(waves)
    assert len(actual) == len(expected)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert actual_state.ledger == expected_state.ledger
    assert torch.equal(actual_state.cpu_rng, expected_state.cpu_rng)
