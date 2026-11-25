# SPDX-License-Identifier: Apache-2.0
"""Exact-shape streaming estimator graph replay and eager fallbacks."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.moss_speech.flow_history_graph import (
    HistoryTransaction,
    PrivateHistoryBank,
    StreamingPrivateHistoryGraphs,
)
from sglang_omni.models.moss_speech.flow_prefix import PrefixSolve, ReferencePrefix
from sglang_omni.models.moss_speech.streaming_flow_graph import StreamingEstimatorGraphs


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU"
)
@torch.inference_mode()
def test_graph_replay_refreshes_inputs_and_owns_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    estimator = Mock(training=False, static_chunk_size=100)
    runner = StreamingEstimatorGraphs(estimator, device)
    inputs = torch.arange(16000, device=device, dtype=torch.bfloat16).view(2, 80, 100)
    # The estimator probe exercises the production graph pool without requiring codec weights.
    monkeypatch.setattr(
        runner, "run_estimator", lambda x, mask, mu, t, spks, cond: x + mu
    )
    first = runner(inputs, inputs, inputs, inputs, inputs, inputs, streaming=True)
    second = runner(inputs, inputs, -inputs, inputs, inputs, inputs, streaming=True)
    assert torch.equal(first, inputs + inputs)
    assert torch.equal(second, torch.zeros_like(inputs))
    assert set(runner.graphs) == {100}
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        assert (
            runner(inputs, inputs, inputs, inputs, inputs, inputs, streaming=True)
            is estimator.return_value
        )
    estimator.assert_called_once()
    failure = Mock(side_effect=torch.OutOfMemoryError("capture allocation failed"))
    monkeypatch.setattr(runner, "capture", failure)
    longer = torch.cat((inputs, inputs), dim=-1)
    for _ in range(2):
        assert (
            runner(longer, longer, longer, longer, longer, longer, streaming=True)
            is estimator.return_value
        )
    failure.assert_called_once()
    assert not runner.graphs


@pytest.mark.parametrize("steps", [3, 4])
@torch.inference_mode()
def test_private_eager_fallback_reads_the_correct_euler_step(steps: int) -> None:
    shared = ReferencePrefix(
        reference_key=b"voice",
        mel_frames=3,
        key_values=[torch.full((1, 3, 1), step) for step in range(steps)],
        convolution_tails=[torch.full((1, 1, 1), 10 + step) for step in range(steps)],
    )
    storage = torch.arange(20, 20 + steps).view(steps, 1)
    views = tuple(
        ReferencePrefix(
            reference_key=b"", mel_frames=1, key_values=[row], convolution_tails=[row]
        )
        for row in storage.unbind()
    )
    history = PrivateHistoryBank(storage, views)
    transaction = HistoryTransaction(history, False, read_history=True)
    call = PrefixSolve(prefix=shared, prefill=False)
    stages = []

    @contextmanager
    def record_stage(reference, private, *, write):
        assert not write
        stages.append(
            (
                reference.mel_frames,
                int(reference.key_values[0][0, 0, 0]),
                int(reference.convolution_tails[0].item()),
                int(private.key_values[0].item()),
            )
        )
        yield

    prefill = SimpleNamespace(
        cached=shared, current=Mock(), transaction=Mock(), stage=record_stage
    )
    prefill.current.get.return_value = call
    prefill.transaction.get.return_value = transaction
    native = Mock(owner_thread=-1, steps=steps)
    inputs = torch.ones(1)
    native.estimator.return_value = inputs
    dispatcher = StreamingPrivateHistoryGraphs(native, prefill)
    for _ in range(steps):
        assert dispatcher(*[inputs] * 6, streaming=True) is inputs
    assert stages == [(4, s, 10 + s, 20 + s) for s in range(steps)]
    assert call.projection_index == call.convolution_index == steps
    assert transaction.history is history and transaction.candidate is None
