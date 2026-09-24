"""Incremental sampling preparation must preserve every selected grid row."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import tests.unit_test.moss_speech.sglang_cpu_env  # noqa: F401
from sglang_omni.models.moss_speech import fsm
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.request_builders import cleanup_ar_request_state


def request(seed: int, sample: bool, penalty: float) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_rows=torch.tensor([[42, 512], [43, 512], [151646, 9]]),
        output_rows=[],
        mode=fsm.MODE_TEXT,
        rng_generator=None,
        effective_seed=seed,
        sampling_history=None,
        params=fsm.MossSamplingParams(
            do_sample=sample,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            repetition_penalty=penalty,
            max_new_tokens=16,
            min_new_tokens=3,
        ),
        pending_feedback_queue=deque(),
        finished=False,
    )


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("penalty", [1.0, 1.1, 2.0])
def test_runner_sampling_is_exact_and_request_isolated(
    sample: bool, penalty: float
) -> None:
    expected = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        logits = [(torch.randn(151680), torch.randn(16512)) for _ in range(6)]
    for optimized in [False, True]:
        runner = object.__new__(MossSpeechModelRunner)
        runner.sampling_optimizations = optimized
        requests = [request(8, sample, penalty), request(17, sample, penalty)]
        for i, (text, audio) in enumerate(logits):
            for data in requests:
                data.mode = fsm.MODE_TEXT if i % 2 else fsm.MODE_AUDIO
                row = runner._select_row(data, text, audio)
                data.output_rows.append(row)
        result = [
            (d.output_rows[:], d.rng_generator.get_state() if sample else None)
            for d in requests
        ]
        if not optimized:
            expected = result
        else:
            for ref, got in zip(expected, result):
                assert ref[0] == got[0]
                if sample:
                    assert torch.equal(ref[1], got[1])
            if penalty != 1:
                assert requests[0].sampling_history is not requests[1].sampling_history
        for data in requests:
            cleanup_ar_request_state(data)
            assert data.sampling_history is None
            assert not data.pending_feedback_queue


def test_history_only_uploads_new_rows_and_rebuilds_after_rewind() -> None:
    runner = object.__new__(MossSpeechModelRunner)
    runner.sampling_optimizations = True
    data = request(8, False, 1.1)
    t, a = torch.zeros(151680), torch.zeros(16512)
    with patch.object(
        fsm, "channel_histories", side_effect=AssertionError("full history rebuilt")
    ):
        runner._select_row(data, t, a)
        cache = data.sampling_history
        ptr = cache.rows.data_ptr()
        data.output_rows.extend([(151667, 4), (151667, 5)])
        runner._select_row(data, t, a)
        assert cache.rows.data_ptr() == ptr
        assert cache.generated == 2
        assert torch.equal(cache.rows[:, 3:5], torch.tensor([[151667, 151667], [4, 5]]))
        data.output_rows = []
        runner._select_row(data, t, a)
        assert data.sampling_history is not cache
        assert data.sampling_history.generated == 0


def test_unit_penalty_never_materializes_history() -> None:
    runner = object.__new__(MossSpeechModelRunner)
    runner.sampling_optimizations = True
    data = request(8, False, 1.0)
    with patch.object(
        fsm, "channel_histories", side_effect=AssertionError("unused history")
    ):
        assert runner._select_row(data, torch.zeros(151680), torch.zeros(16512)) == (
            0,
            0,
        )
    assert data.sampling_history is None


def test_sampling_option_only_reaches_ar_stage() -> None:
    from sglang_omni.models.moss_speech.config import (
        MossSpeechPipelineConfig,
        MossSpeechStreamingPipelineConfig,
    )

    for cls in [MossSpeechPipelineConfig, MossSpeechStreamingPipelineConfig]:
        cfg = cls(model_path="local", sampling_optimizations=True)
        for stage in cfg.stages:
            assert stage.factory_args.get("sampling_optimizations", False) == (
                stage.name == "ar_engine"
            )


def test_request_reset_and_history_capacity() -> None:
    from sglang_omni.models.moss_speech.sampling import DeviceChannelHistory

    from .test_native_lifecycle_cpu import make_data

    data = make_data()
    data.sampling_history = DeviceChannelHistory(
        data.prompt_rows, 1, torch.device("cpu")
    )
    cache = data.sampling_history
    with pytest.raises(ValueError, match="budget"):
        cache.histories([(42, 3), (43, 4)])
    assert cache.generated == 0
    data.reset()
    assert data.sampling_history is None


def test_sampling_factory_and_builder_reach_runner(monkeypatch) -> None:
    from sglang_omni.models.moss_speech import model_runner, stages
    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder

    monkeypatch.setattr(stages, "validate_ar_preconditions", lambda *a, **kw: None)
    monkeypatch.setattr(MossSpeechEngineBuilder, "build", lambda self, *a, **kw: self)
    builder = stages.create_ar_engine_executor("local", sampling_optimizations=True)
    assert builder.sampling_optimizations
    monkeypatch.setattr(
        model_runner, "MossSpeechModelRunner", lambda *a: SimpleNamespace()
    )
    assert builder.make_model_runner(None, None).sampling_optimizations
