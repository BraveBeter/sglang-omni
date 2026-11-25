# SPDX-License-Identifier: Apache-2.0
"""Stage each Euler step's closed reference at addresses shared by suffix graphs."""

from threading import get_ident

import torch

from sglang_omni.models.moss_speech.flow_prefix import (
    ReferencePrefix,
    StreamingReferencePrefill,
)
from sglang_omni.models.moss_speech.streaming_flow_graph import (
    MIN_CAPTURE_FREE_BYTES,
    StreamingEstimatorGraphs,
)


class StreamingReferenceGraphs(StreamingEstimatorGraphs):
    def __init__(
        self,
        full_graphs: StreamingEstimatorGraphs,
        prefill: StreamingReferencePrefill,
        steps: int,
    ) -> None:
        super().__init__(full_graphs.estimator, full_graphs.owner_stream.device)
        self.full_graphs: StreamingEstimatorGraphs = full_graphs
        self.prefill: StreamingReferencePrefill = prefill
        self.steps = steps
        self.reference: ReferencePrefix | None = None
        self.step_bank: torch.Tensor | None = None
        self.step_buffer: torch.Tensor | None = None
        self.step_views: ReferencePrefix | None = None

    def bind(self, prefix: ReferencePrefix) -> None:
        self.reference = prefix
        self.step_bank = None
        if (
            self.step_views is not None
            and self.step_views.mel_frames != prefix.mel_frames
        ):
            # note (Codex): Captured kernels retain the old buffer address and shape until every replay has finished.
            self.owner_stream.synchronize()
            self.graphs.clear()
            self.pool = torch.cuda.graph_pool_handle()
            self.capture_enabled = True
            self.step_views = None
            self.step_buffer = None
        else:
            pass
        key_count, extra_keys = divmod(len(prefix.key_values), self.steps)
        tail_count, extra_tails = divmod(len(prefix.convolution_tails), self.steps)
        assert key_count and tail_count and not extra_keys and not extra_tails
        tensors = prefix.key_values[:key_count] + prefix.convolution_tails[:tail_count]
        sizes = [tensor.numel() for tensor in tensors]
        elements = sum(sizes)
        template = tensors[0]
        allocation_bytes = (
            elements
            * template.element_size()
            * (self.steps + int(self.step_buffer is None))
        )
        free_bytes, _ = torch.cuda.mem_get_info(template.device)
        if free_bytes < allocation_bytes + MIN_CAPTURE_FREE_BYTES:
            return
        else:
            pass
        if self.step_buffer is None:
            buffer = template.new_empty(elements)
            views = [
                chunk.view_as(tensor)
                for chunk, tensor in zip(buffer.split(sizes), tensors, strict=True)
            ]
            self.step_views = ReferencePrefix(
                reference_key=b"",
                mel_frames=prefix.mel_frames,
                key_values=views[:key_count],
                convolution_tails=views[key_count:],
            )
            self.step_buffer = buffer
        else:
            pass
        bank = template.new_empty((self.steps, elements))
        keys: list[torch.Tensor] = []
        tails: list[torch.Tensor] = []
        for step in range(self.steps):
            key_start = step * key_count
            tail_start = step * tail_count
            sources = (
                prefix.key_values[key_start : key_start + key_count]
                + prefix.convolution_tails[tail_start : tail_start + tail_count]
            )
            views = [
                chunk.view_as(tensor)
                for chunk, tensor in zip(bank[step].split(sizes), sources, strict=True)
            ]
            for source, destination in zip(sources, views, strict=True):
                destination.copy_(source)
            keys.extend(views[:key_count])
            tails.extend(views[key_count:])
        # note (Codex): Eager fallback can read the packed views directly, so the model keeps one copy of each voice.
        prefix.key_values, prefix.convolution_tails = keys, tails
        self.step_bank = bank

    def run_estimator(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        assert self.step_views is not None
        # note (Codex): Both the warmup and capture must generate the mask inside their own execution.
        with self.prefill.read_step(self.step_views):
            return super().run_estimator(x, mask, mu, t, spks, cond)

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        call = self.prefill.current.get()
        if call is None:
            return self.full_graphs(x, mask, mu, t, spks, cond, streaming)
        elif (
            get_ident() != self.owner_thread
            or torch.cuda.current_stream(x.device) != self.owner_stream
        ):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        elif call.prefill:
            self.reference = None
            self.step_bank = None
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            pass
        if self.reference is not call.prefix:
            try:
                self.bind(call.prefix)
            except torch.OutOfMemoryError:
                self.step_bank = None
        else:
            pass
        if self.step_bank is None:
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            assert self.step_buffer is not None and self.step_views is not None
        keys = len(self.step_views.key_values)
        tails = len(self.step_views.convolution_tails)
        step, position = divmod(call.projection_index, keys)
        assert not position and call.convolution_index == step * tails
        self.step_buffer.copy_(self.step_bank[step])
        with self.prefill.read_step(self.step_views):
            result = super().__call__(x, mask, mu, t, spks, cond, streaming)
        call.projection_index += keys
        call.convolution_index += tails
        return result
