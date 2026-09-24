"""Full-CFM startup must prove real codec output before publishing graph state."""

from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)


def outcome():
    return (
        [torch.tensor([0.0, -0.0, 0.25])],
        [{"samples": 3}],
        torch.tensor([1, 2], dtype=torch.uint8),
        torch.tensor([3, 4], dtype=torch.uint8),
    )


@pytest.fixture
def prepared(monkeypatch):
    events = []
    codec = MossSpeechStreamingCodec(
        None,
        chunk_size=25,
        device="cpu",
        codec_optimizations=True,
        flow=SimpleNamespace(decoder=object()),
        hift=object(),
        small_first_chunk=True,
    )
    codec.device = torch.device("cuda:0")
    bank = SimpleNamespace(
        binding=SimpleNamespace(valid=True, shapes=(1400, 1500)),
        active_length=None,
        hits=100,
        close=lambda: events.append("prefix.close"),
    )
    codec._prefix_cache = bank
    codec._prepared_voice = object()
    graph = Mock()
    graph.prepare.side_effect = lambda callback: events.append("prepare") or {
        "capture": 2
    }
    graph.install.side_effect = lambda: events.append("install")
    graph.close.side_effect = lambda: events.append("graph.close")
    graph.status.side_effect = lambda: dict(
        enabled=True, shapes=[1400, 1500], graph_replays=counter[0], bytes=123
    )
    counter = [4]
    results = [outcome(), outcome()]

    def calibrate(value):
        assert value is codec._prepared_voice
        events.append("calibration")
        if events.count("calibration") > 1:
            counter[0] += 2
        return results.pop(0)

    monkeypatch.setattr(codec, "_calibrate_prepared_voice", calibrate, raising=False)
    monkeypatch.setattr(
        codec, "_log_prefix_cache", lambda event: events.append("prefix.log")
    )
    module_name = "sglang_omni.models.moss_speech.components.cfm_full_graph"
    module = ModuleType(module_name)
    builder = Mock(return_value=graph)
    module.FullCfmGraphBank = builder
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 1000)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 2000)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    return SimpleNamespace(
        codec=codec,
        graph=graph,
        bank=bank,
        builder=builder,
        events=events,
        results=results,
        counter=counter,
    )


def test_disabled_state_and_success_publish_only_after_exact_codec_gate(prepared):
    x = prepared
    assert x.codec.full_graph_status() == {"enabled": False}
    result = x.codec.prepare_full_graphs()
    assert result["enabled"] is True
    assert result["startup"]["exact"] is True
    assert result["startup"]["startup_graph_replays"] == 2
    assert result["startup"]["initial_allocated_bytes"] == 1000
    assert result["startup"]["peak_allocated_bytes"] == 2000
    assert result["startup"]["ready_allocated_bytes"] == 1000
    assert x.codec._full_graphs is x.graph
    x.builder.assert_called_once_with(x.codec.flow.decoder, x.bank)
    x.graph.prepare.assert_called_once()
    assert x.events.index("prepare") < x.events.index("install")
    assert x.events.count("calibration") == 2
    assert "graph.close" not in x.events
    with pytest.raises(ValueError, match="already"):
        x.codec.prepare_full_graphs()
    assert x.builder.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        "sessions",
        "no_prefix",
        "invalid_prefix",
        "active_prefix",
        "no_prepared_voice",
        "chunk5",
        "unoptimized",
        "cpu",
    ],
)
def test_ineligible_preparation_rejects_before_graph_construction(prepared, change):
    x = prepared
    if change == "sessions":
        x.codec.sessions["live"] = object()
    elif change == "no_prefix":
        x.codec._prefix_cache = None
    elif change == "invalid_prefix":
        x.bank.binding.valid = False
    elif change == "active_prefix":
        x.bank.active_length = 1400
    elif change == "no_prepared_voice":
        x.codec._prepared_voice = None
    elif change == "chunk5":
        x.codec.chunk_size = 5
    elif change == "unoptimized":
        x.codec.codec_optimizations = False
    else:
        x.codec.device = torch.device("cpu")
    with pytest.raises(ValueError):
        x.codec.prepare_full_graphs()
    x.builder.assert_not_called()
    assert "calibration" not in x.events


@pytest.mark.parametrize(
    "change", ["signed_zero", "ledger", "cpu_rng", "device_rng", "missing_wave"]
)
def test_changed_complete_codec_result_rejects_and_restores_owner(prepared, change):
    x = prepared
    value = list(copy.deepcopy(x.results[1]))
    if change == "signed_zero":
        value[0][0][1] = 0.0
    elif change == "ledger":
        value[1][0]["samples"] += 1
    elif change == "cpu_rng":
        value[2][0] += 1
    elif change == "device_rng":
        value[3][0] += 1
    else:
        value[0].append(value[0][0].clone())
    x.results[1] = tuple(value)
    with pytest.raises(RuntimeError, match="codec"):
        x.codec.prepare_full_graphs()
    assert x.codec.full_graph_status() == {"enabled": False}
    assert x.events[-1] == "graph.close"
    assert x.codec._prefix_cache is x.bank
    assert "prefix.close" not in x.events


def test_zero_actual_graph_replays_cannot_masquerade_as_success(prepared):
    x = prepared
    x.graph.status.side_effect = lambda: dict(
        enabled=True, shapes=[1400, 1500], graph_replays=4, bytes=123
    )
    with pytest.raises(RuntimeError, match="replay"):
        x.codec.prepare_full_graphs()
    assert x.codec._full_graphs is None
    x.graph.close.assert_called_once()


@pytest.mark.parametrize("failure", ["prepare", "install", "validation"])
def test_exception_rolls_back_graph_owner(prepared, failure):
    x = prepared
    if failure == "validation":
        x.codec._calibrate_prepared_voice = Mock(
            side_effect=[outcome(), RuntimeError("injected")]
        )
    else:
        getattr(x.graph, failure).side_effect = RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        x.codec.prepare_full_graphs()
    x.graph.close.assert_called_once()
    assert x.codec._full_graphs is None
    assert x.codec._prefix_cache is x.bank


def test_close_releases_graphs_before_prefix_and_drops_prepared_voice(prepared):
    x = prepared
    x.codec.prepare_full_graphs()
    x.codec.close()
    x.codec.close()
    assert x.events.index("graph.close") < x.events.index("prefix.close")
    assert x.events.count("graph.close") == x.events.count("prefix.close") == 1
    assert x.codec._prepared_voice is None
    assert x.codec.full_graph_status() == {"enabled": False}


def test_ready_and_request_logs_do_not_count_capture_replays(prepared, caplog):
    import json
    import logging

    x = prepared
    with caplog.at_level(logging.INFO):
        x.codec.prepare_full_graphs()
        x.counter[0] += 1
        x.codec._log_full_graphs("test")
    records = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MOSS complete CFM graphs")
    ]
    ready = json.loads(records[0].split(": ", 1)[1])
    request = json.loads(records[1].split(": ", 1)[1])
    assert ready["request_graph_replays"] == 0
    assert request["request_graph_replays"] == 1


def test_real_cpu_calibration_preserves_rng_and_cleans_up_on_failure():
    class Flow:
        pre_lookahead_len = 3
        token_mel_ratio = 4

        def inference(self, *, token, finalize, **kwargs):
            return torch.ones(1, 80, (token.shape[1] - 3) * 4), None

    class Hift:
        def inference(self, *, speech_feat, cache_source):
            wave = torch.rand(1, speech_feat.shape[-1] * 480)
            return wave, wave.unsqueeze(1)

    codec = MossSpeechStreamingCodec(
        None,
        chunk_size=25,
        small_first_chunk=True,
        codec_optimizations=True,
        device="cpu",
        flow=Flow(),
        hift=Hift(),
    )
    voice = dict(
        prompt_token=torch.zeros(1, 342, dtype=torch.int32),
        prompt_feat=torch.zeros(1, 1368, 80),
        embedding=torch.zeros(1, 192),
    )
    prepared = SimpleNamespace(voice=voice, warmup_audio_codes=tuple(range(36)))
    before = torch.get_rng_state().clone()
    first = codec._calibrate_prepared_voice(prepared)
    second = codec._calibrate_prepared_voice(prepared)
    assert codec._same_calibration(first, second)
    assert [wave.numel() for wave in first[0]] == [11520, 48000]
    assert first[3] is None
    assert not codec.sessions
    assert torch.equal(before, torch.get_rng_state())
    codec.hift.inference = Mock(side_effect=RuntimeError("injected"))
    with pytest.raises(RuntimeError, match="injected"):
        codec._calibrate_prepared_voice(prepared)
    assert not codec.sessions
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("failure", [False, True])
def test_prefix_startup_hits_include_real_graph_preparation_only(prepared, failure):
    x = prepared
    x.bank.hits = 100
    x.codec._prefix_cache_receipt = {"startup_attention_hits": 80}
    original = x.codec._calibrate_prepared_voice

    def calibrate(value):
        x.bank.hits += 20
        result = original(value)
        if failure and x.events.count("calibration") == 2:
            raise RuntimeError("injected")
        return result

    x.codec._calibrate_prepared_voice = calibrate
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            x.codec.prepare_full_graphs()
    else:
        x.codec.prepare_full_graphs()
    assert x.codec._prefix_cache_receipt["startup_attention_hits"] == 120
    assert x.bank.hits - x.codec._prefix_cache_receipt["startup_attention_hits"] == 20
