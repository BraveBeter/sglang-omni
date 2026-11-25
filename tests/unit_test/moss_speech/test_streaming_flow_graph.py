# SPDX-License-Identifier: Apache-2.0
"""Exact-shape streaming estimator graph replay and eager fallbacks."""

from unittest.mock import Mock

import pytest
import torch

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
