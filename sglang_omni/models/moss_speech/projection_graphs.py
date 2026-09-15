# SPDX-License-Identifier: Apache-2.0
"""Model-local partial graphs; attention and request state remain eager."""
from __future__ import annotations

from typing import Any, Callable

import torch


class ProjectionGraph:
    """Replay a pure tensor function using owned static inputs and outputs.

    Returned tensors are valid until the next replay of this instance. Each
    layer/section has its own graph pool so residuals cannot alias a later
    section's allocations.
    """

    def __init__(self, function: Callable[..., Any], inputs: tuple[Any, ...]) -> None:
        tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
        if not tensors or any(not value.is_cuda for value in tensors):
            raise ValueError("Projection graphs require CUDA tensors")
        if any(
            value is not None and not isinstance(value, torch.Tensor)
            for value in inputs
        ):
            raise ValueError("Projection graph inputs must be tensors or None")
        self.inputs = tuple(
            value.clone() if value is not None else None for value in inputs
        )
        self.replay_count = 0
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            warmup = torch.cuda.Stream(device=tensors[0].device)
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(2):
                    function(*self.inputs)
            torch.cuda.current_stream().wait_stream(warmup)
            with torch.cuda.graph(self.graph):
                self.outputs = function(*self.inputs)

    def __call__(self, *inputs: Any) -> Any:
        if len(inputs) != len(self.inputs):
            raise ValueError("Projection graph input count changed")
        for actual, static in zip(inputs, self.inputs):
            if actual is None or static is None:
                if actual is not static:
                    raise ValueError("Projection graph optional input changed")
                continue
            if (
                actual.shape != static.shape
                or actual.dtype != static.dtype
                or actual.device != static.device
            ):
                raise ValueError("Projection graph input shape/dtype/device changed")
            static.copy_(actual)
        self.graph.replay()
        self.replay_count += 1
        return self.outputs


def run_projection(
    layer: Any,
    name: str,
    function: Callable[..., Any],
    inputs: tuple[Any, ...],
    forward_batch: Any,
) -> Any:
    lengths = forward_batch._moss_request_lengths
    if not (
        getattr(layer, "enable_projection_graphs", False)
        and forward_batch.forward_mode.is_decode()
        and 1 <= len(lengths) <= 4
        and all(length == 1 for length in lengths)
        and inputs[0].is_cuda
    ):
        return function(*inputs)
    graphs = getattr(layer, "_projection_graphs", None)
    if graphs is None:
        graphs = layer._projection_graphs = {}
    key = (name, len(lengths), tuple(value is None for value in inputs))
    if key not in graphs:
        graphs[key] = ProjectionGraph(function, inputs)
    return graphs[key](*inputs)
