"""Complete CFM graphs are opt-in, source-local startup work, never defaults."""

from __future__ import annotations

from copy import deepcopy
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
    return next(stage for stage in config.stages if stage.name == "audio_vocoder")


def test_unset_and_legacy_serialized_config_preserve_default():
    config = MossSpeechStreamingPipelineConfig(model_path="local")
    assert config.codec_full_graphs is None
    assert "codec_full_graphs" not in vocoder(config).factory_args
    legacy = config.model_dump()
    legacy.pop("codec_full_graphs")
    restored = MossSpeechStreamingPipelineConfig(**legacy)
    assert restored.codec_full_graphs is None
    assert restored.model_dump() == config.model_dump()


def test_inherited_stage_value_and_explicit_false_with_real_merge():
    config = MossSpeechStreamingPipelineConfig(model_path="local")
    vocoder(config).factory_args["codec_full_graphs"] = True
    inherited = ConfigManager(config).merge_config({})
    assert inherited.codec_full_graphs is None
    assert vocoder(inherited).factory_args["codec_full_graphs"] is True
    disabled = ConfigManager(inherited).merge_config({"codec_full_graphs": False})
    assert disabled.codec_full_graphs is False
    assert vocoder(disabled).factory_args["codec_full_graphs"] is False
    assert vocoder(inherited).factory_args["codec_full_graphs"] is True


def test_enabled_only_reaches_streaming_factory_without_memory_override():
    base = MossSpeechStreamingPipelineConfig(model_path="local")
    for stage in base.stages:
        stage.factory_args["user_preserved_marker"] = stage.name
        if stage.name == "ar_engine":
            stage.factory_args["mem_fraction_static"] = 0.37
    vocoder(base).factory_args.update(chunk_size=25, gpu_id=1)
    before = deepcopy(base.model_dump())
    enabled = ConfigManager(base).merge_config({"codec_full_graphs": True})
    assert enabled.codec_full_graphs is True
    assert vocoder(enabled).factory_args["codec_full_graphs"] is True
    after = enabled.model_dump()
    after["codec_full_graphs"] = None
    next(s for s in after["stages"] if s["name"] == "audio_vocoder")[
        "factory_args"
    ].pop("codec_full_graphs")
    assert after == before
    assert base.model_dump() == before


@pytest.mark.parametrize(
    "options", ({}, {"codec_full_graphs": None}, {"codec_full_graphs": False})
)
def test_offline_unset_or_false_does_not_add_unknown_factory_argument(options):
    config = MossSpeechPipelineConfig(model_path="local", **options)
    assert "codec_full_graphs" not in vocoder(config).factory_args


def test_offline_enabled_is_rejected():
    with pytest.raises(ValueError, match="codec_full_graphs.*streaming"):
        MossSpeechPipelineConfig(model_path="local", codec_full_graphs=True)


def test_custom_nonstreaming_vocoder_cannot_receive_graph_option():
    base = MossSpeechStreamingPipelineConfig(model_path="local")
    vocoder(base).factory = "custom.codec_factory"
    with pytest.raises(ValueError, match="codec_full_graphs.*streaming"):
        ConfigManager(base).merge_config({"codec_full_graphs": True})


@pytest.fixture
def mocked_factory(monkeypatch):
    module = import_module("sglang_omni.models.moss_speech.streaming")
    stages = import_module("sglang_omni.models.moss_speech.stages")
    codec_module = import_module(
        "sglang_omni.models.moss_speech.components.streaming_codec"
    )
    events = []
    codec = Mock(hift=object())
    codec.prepare_prefix_cache.side_effect = lambda manifest: events.append(
        ("prefix", manifest)
    )
    codec.prepare_full_graphs.side_effect = lambda: events.append(("graphs",))
    codec.close.side_effect = lambda: events.append(("close",))
    build_codec = Mock(side_effect=lambda *a, **kw: events.append(("codec",)) or codec)
    build_offline = Mock(
        side_effect=lambda *a, **kw: events.append(("offline",))
        or SimpleNamespace(_fn="offline_fn")
    )
    scheduler = Mock(
        side_effect=lambda *a, **kw: events.append(("scheduler",))
        or SimpleNamespace(codec=a[0], options=kw)
    )
    resolve = Mock(return_value="codec_dir")
    monkeypatch.setattr(codec_module, "MossSpeechStreamingCodec", build_codec)
    monkeypatch.setattr(stages, "create_audio_vocoder_executor", build_offline)
    monkeypatch.setattr(stages, "_resolve_codec_dir", resolve)
    monkeypatch.setattr(module, "MossSpeechStreamingVocoder", scheduler)
    return SimpleNamespace(
        module=module,
        codec=codec,
        resolve=resolve,
        build_codec=build_codec,
        build_offline=build_offline,
        scheduler=scheduler,
        events=events,
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"chunk_size": 5},
        {"codec_optimizations": False},
        {"codec_prefix_cache_manifest": None},
        {"codec_prefix_cache_manifest": ""},
    ),
)
def test_enabled_invalid_configuration_rejected_before_any_load(
    mocked_factory, changes
):
    options = dict(
        chunk_size=25,
        codec_optimizations=True,
        codec_prefix_cache_manifest="prepared.json",
        codec_full_graphs=True,
    )
    options.update(changes)
    with pytest.raises(ValueError, match="(graph|Graph|Prefix cache)"):
        mocked_factory.module.create_streaming_audio_executor("local", **options)
    mocked_factory.resolve.assert_not_called()
    mocked_factory.build_codec.assert_not_called()
    mocked_factory.build_offline.assert_not_called()
    assert mocked_factory.events == []


def test_bank_then_graphs_then_scheduler_and_independent_options(mocked_factory):
    result = mocked_factory.module.create_streaming_audio_executor(
        "local",
        chunk_size=25,
        codec_optimizations=True,
        codec_prefix_cache_manifest="prepared.json",
        codec_full_graphs=True,
        small_first_chunk=True,
        stream_decode_fairness=True,
        gpu_id=1,
    )
    assert mocked_factory.events == [
        ("codec",),
        ("prefix", "prepared.json"),
        ("graphs",),
        ("offline",),
        ("scheduler",),
    ]
    mocked_factory.codec.prepare_full_graphs.assert_called_once_with()
    mocked_factory.codec.close.assert_not_called()
    assert result.codec is mocked_factory.codec
    assert result.options["defer_stream_decode"] is True
    assert mocked_factory.build_codec.call_args.kwargs["device"] == "cuda:1"
    assert mocked_factory.build_codec.call_args.kwargs["small_first_chunk"] is True
    assert "codec_full_graphs" not in mocked_factory.build_codec.call_args.kwargs
    assert (
        mocked_factory.build_offline.call_args.kwargs["shared_hift"]
        is mocked_factory.codec.hift
    )


@pytest.mark.parametrize("flag", (False, None))
def test_disabled_never_prepares_graphs_but_can_prepare_bank(mocked_factory, flag):
    mocked_factory.module.create_streaming_audio_executor(
        "local",
        chunk_size=25,
        codec_optimizations=True,
        codec_prefix_cache_manifest="prepared.json",
        codec_full_graphs=flag,
    )
    mocked_factory.codec.prepare_prefix_cache.assert_called_once_with("prepared.json")
    mocked_factory.codec.prepare_full_graphs.assert_not_called()
    mocked_factory.codec.close.assert_not_called()


def test_legacy_factory_without_graph_argument_stays_unchanged(mocked_factory):
    mocked_factory.module.create_streaming_audio_executor("local", chunk_size=5)
    assert mocked_factory.events == [("offline",), ("codec",), ("scheduler",)]
    mocked_factory.codec.prepare_prefix_cache.assert_not_called()
    mocked_factory.codec.prepare_full_graphs.assert_not_called()


@pytest.mark.parametrize("failure_at", ("prefix", "graphs", "offline", "scheduler"))
@pytest.mark.parametrize("exception_type", (RuntimeError, KeyboardInterrupt))
def test_any_preparation_or_publishing_failure_closes_codec(
    mocked_factory, failure_at, exception_type
):
    target = dict(
        prefix=mocked_factory.codec.prepare_prefix_cache,
        graphs=mocked_factory.codec.prepare_full_graphs,
        offline=mocked_factory.build_offline,
        scheduler=mocked_factory.scheduler,
    )[failure_at]
    error = exception_type("startup failure")
    target.side_effect = error
    with pytest.raises(exception_type, match="startup failure") as captured:
        mocked_factory.module.create_streaming_audio_executor(
            "local",
            chunk_size=25,
            codec_optimizations=True,
            codec_prefix_cache_manifest="prepared.json",
            codec_full_graphs=True,
        )
    assert captured.value is error
    mocked_factory.codec.close.assert_called_once_with()
    if failure_at == "prefix":
        mocked_factory.codec.prepare_full_graphs.assert_not_called()
    if failure_at != "scheduler":
        mocked_factory.scheduler.assert_not_called()
