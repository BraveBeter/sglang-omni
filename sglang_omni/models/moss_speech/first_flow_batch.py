# SPDX-License-Identifier: Apache-2.0
"""Reference-aware CUDA graphs for matching first streaming Flow hops."""

import torch

from sglang_omni.models.moss_speech.flow_precision import FlowCondition
from sglang_omni.models.moss_speech.flow_prefix import ReferencePrefix
from sglang_omni.models.moss_speech.flow_prefix_graph import StreamingReferenceGraphs
from sglang_omni.models.moss_speech.streaming_flow_graph import StreamingEstimatorGraphs


class FirstFlowGraphUnavailable(RuntimeError):
    """The device cannot capture a graph for this reference and batch shape."""


class FirstFlowBatch(StreamingEstimatorGraphs):
    def __init__(self, production: StreamingReferenceGraphs, rows: int) -> None:
        super().__init__(production.estimator, production.owner_stream.device)
        self.production = production
        self.reference = production.reference
        source_views = production.step_views
        assert self.reference is not None and source_views is not None
        key_count = len(source_views.key_values)
        tail_count = len(source_views.convolution_tails)
        sources = (
            self.reference.key_values[:key_count]
            + self.reference.convolution_tails[:tail_count]
        )
        shapes = [(rows * t.shape[0], *t.shape[1:]) for t in sources]
        sizes = [rows * t.numel() for t in sources]
        self.bank = sources[0].new_empty((production.steps, sum(sizes)))
        self.buffer = torch.empty_like(self.bank[0])
        views = [
            part.view(shape)
            for part, shape in zip(self.buffer.split(sizes), shapes, strict=True)
        ]
        self.views = ReferencePrefix(
            reference_key=b"",
            mel_frames=self.reference.mel_frames,
            key_values=views[:key_count],
            convolution_tails=views[key_count:],
        )
        for step in range(production.steps):
            tensors = (
                self.reference.key_values[step * key_count : (step + 1) * key_count]
                + self.reference.convolution_tails[
                    step * tail_count : (step + 1) * tail_count
                ]
            )
            for part, tensor, shape in zip(
                self.bank[step].split(sizes), tensors, shapes, strict=True
            ):
                part.view(shape).copy_(tensor.repeat_interleave(rows, dim=0))

    def run_estimator(self, x, mask, mu, t, spks, cond) -> torch.Tensor:
        with self.production.prefill.read_step(self.views):
            return super().run_estimator(x, mask, mu, t, spks, cond)

    def first_mels(
        self, decoder, rows: list[FlowCondition], prompt_frames: int
    ) -> tuple[torch.Tensor, ...]:
        size = len(rows)
        offset = self.views.mel_frames
        mu = torch.cat([r.mu[..., offset:] for r in rows])
        mask = torch.cat([r.mask[..., offset:] for r in rows])
        cond = torch.cat([r.cond[..., offset:] for r in rows])
        spks = torch.cat([r.spks for r in rows])
        frames = mu.shape[-1]
        x = decoder.rand_noise[..., offset : offset + frames].to(mu).repeat(size, 1, 1)
        mu = torch.cat((mu, torch.zeros_like(mu)))
        mask = mask.repeat(2, 1, 1)
        spks = torch.cat((spks, torch.zeros_like(spks)))
        cond = torch.cat((cond, torch.zeros_like(cond)))
        steps = self.production.steps
        grid = torch.linspace(0, 1, steps + 1, device=x.device, dtype=x.dtype)
        if decoder.t_scheduler == "cosine":
            grid = 1 - torch.cos(grid * 0.5 * torch.pi)
        else:
            pass
        t = grid[0].unsqueeze(0)
        dt = grid[1] - grid[0]
        for step in range(steps):
            inputs = (
                x.repeat(2, 1, 1),
                mask,
                mu,
                t.expand(2 * size).contiguous(),
                spks,
                cond,
            )
            self.buffer.copy_(self.bank[step])
            graph = self.graphs.get(frames)
            if graph is None:
                try:
                    graph = self.capture(inputs)
                except RuntimeError as exc:
                    raise FirstFlowGraphUnavailable(str(exc)) from exc
                self.graphs[frames] = graph
            else:
                pass
            for destination, source in zip(graph.inputs, inputs, strict=True):
                destination.copy_(source)
            graph.graph.replay()
            positive, negative = graph.output.chunk(2)
            x = x + dt * (
                (1.0 + decoder.inference_cfg_rate) * positive
                - decoder.inference_cfg_rate * negative
            )
            t = t + dt
            if step + 1 < steps:
                dt = grid[step + 2] - t
            else:
                pass
        return x[..., prompt_frames - offset :].float().split(1)
