# SPDX-License-Identifier: Apache-2.0
"""Append closed training chunks to request-owned histories at every Euler step."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod
from threading import get_ident

import torch
import torch.nn.functional as F

from sglang_omni.models.moss_speech.flow_history_graph import PrivateHistoryPrefill
from sglang_omni.models.moss_speech.flow_prefix import (
    ReferencePrefix,
    StreamingReferencePrefill,
)
from sglang_omni.models.moss_speech.flow_prefix_graph import StreamingReferenceGraphs
from sglang_omni.models.moss_speech.streaming_flow_graph import (
    MAX_GRAPH_FRAMES,
    MIN_CAPTURE_FREE_BYTES,
    StreamingEstimatorGraphs,
)


@dataclass
class PagedHistory:
    pages: tuple[torch.Tensor, ...]


class PageLayout:
    def __init__(self, prefix: ReferencePrefix, steps: int, width: int):
        keys = len(prefix.key_values) // steps
        tails = len(prefix.convolution_tails) // steps
        self.shapes = [
            (v.size(0), width, *v.shape[2:]) for v in prefix.key_values[:keys]
        ]
        self.shapes += [v.shape for v in prefix.convolution_tails[:tails]]
        self.sizes = [prod(s) for s in self.shapes]
        self.elements = sum(self.sizes)
        self.keys = keys
        self.width = width

    def view(self, storage: torch.Tensor) -> ReferencePrefix:
        tensors = [
            t.view(s)
            for t, s in zip(storage.split(self.sizes), self.shapes, strict=True)
        ]
        return ReferencePrefix(
            reference_key=b"",
            mel_frames=self.width,
            key_values=tensors[: self.keys],
            convolution_tails=tensors[self.keys :],
        )


@dataclass
class PageStep:
    shared: ReferencePrefix
    previous: tuple[ReferencePrefix, ...]
    append: tuple[ReferencePrefix, ...]


class PagedHistoryPrefill(PrivateHistoryPrefill):
    def __init__(self, decoder: torch.nn.Module, steps: int):
        super().__init__(decoder)
        self.steps = steps
        self.layout: PageLayout | None = None
        self.active_pages: ContextVar[PageStep | None] = ContextVar(
            "moss_flow_pages", default=None
        )

    @contextmanager
    def apply_pages(self, pages: PageStep) -> Iterator[None]:
        token = self.active_pages.set(pages)
        try:
            with self.read_step(pages.shared):
                yield
        finally:
            self.active_pages.reset(token)

    def project(
        self, layer: torch.nn.Linear, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        pages = self.active_pages.get()
        if pages is None:
            return super().project(layer, hidden_states)
        else:
            call = self.current.get()
            assert call is not None
        index = call.projection_index
        call.projection_index += 1
        projected = F.linear(hidden_states, layer.weight, layer.bias)
        for page_number, page in enumerate(pages.append):
            start = page_number * self.chunk_frames
            page.key_values[index].copy_(
                projected[:, start : start + self.chunk_frames]
            )
        return torch.cat(
            (
                pages.shared.key_values[index],
                *(page.key_values[index] for page in pages.previous),
                projected,
            ),
            dim=1,
        )

    def convolve(self, layer: torch.nn.Conv1d, channels: torch.Tensor) -> torch.Tensor:
        pages = self.active_pages.get()
        if pages is None:
            return super().convolve(layer, channels)
        else:
            call = self.current.get()
            assert call is not None
        index = call.convolution_index
        call.convolution_index += 1
        for page_number, page in enumerate(pages.append, 1):
            end = page_number * self.chunk_frames
            page.convolution_tails[index].copy_(
                channels[..., end - layer.causal_padding : end]
            )
        prefix = pages.previous[-1] if pages.previous else pages.shared
        return F.conv1d(
            torch.cat((prefix.convolution_tails[index], channels), dim=-1),
            layer.weight,
            layer.bias,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
        )

    def solve_euler(
        self, full_solver, noise, timesteps, mu, mask, spks, cond, streaming
    ):
        call = self.current.get()
        tx = self.transaction.get()
        if call is None or call.prefill or tx is None:
            return StreamingReferencePrefill.solve_euler(
                self, full_solver, noise, timesteps, mu, mask, spks, cond, streaming
            )
        else:
            shared = call.prefix
        history = tx.history
        previous = () if history is None else history.pages
        suffix = mu.size(-1) - shared.mel_frames
        assert suffix > len(previous) * self.chunk_frames
        wanted = min(
            MAX_GRAPH_FRAMES // self.chunk_frames,
            max(0, suffix // self.chunk_frames - 1),
        )
        # note (Codex): Start at the validated four-chunk snapshot; shorter requests keep the native path.
        if tx.final or (not previous and wanted < 4):
            adding = 0
        else:
            adding = max(0, wanted - len(previous))
        if self.layout is None:
            self.layout = PageLayout(shared, self.steps, self.chunk_frames)
        else:
            pass
        if adding:
            template = shared.key_values[0]
            required = (
                adding * self.steps * self.layout.elements * template.element_size()
            )
            if (
                torch.cuda.mem_get_info(template.device)[0]
                >= MIN_CAPTURE_FREE_BYTES + required
            ):
                try:
                    storage = template.new_empty(
                        (adding, self.steps, self.layout.elements)
                    )
                except torch.OutOfMemoryError:
                    pass
                else:
                    tx.candidate = PagedHistory(previous + storage.unbind())
                    tx.keep_frames = adding * self.chunk_frames
            else:
                pass
        else:
            pass
        if previous:
            tx.read_history = True
            call.prefix = ReferencePrefix(
                reference_key=shared.reference_key,
                mel_frames=shared.mel_frames + len(previous) * self.chunk_frames,
                key_values=shared.key_values,
                convolution_tails=shared.convolution_tails,
            )
        else:
            pass
        try:
            return StreamingReferencePrefill.solve_euler(
                self, full_solver, noise, timesteps, mu, mask, spks, cond, streaming
            )
        finally:
            call.prefix = shared


class PagedEstimatorGraphs(StreamingEstimatorGraphs):
    def __init__(
        self,
        native: StreamingReferenceGraphs,
        prefill: PagedHistoryPrefill,
        pages: PageStep,
    ):
        super().__init__(native.estimator, native.owner_stream.device)
        self.prefill = prefill
        self.pages = pages

    def run_estimator(self, *inputs: torch.Tensor) -> torch.Tensor:
        with self.prefill.apply_pages(self.pages):
            return super().run_estimator(*inputs)


class StreamingPagedHistoryGraphs:
    def __init__(self, native: StreamingReferenceGraphs, prefill: PagedHistoryPrefill):
        self.native = native
        self.prefill = prefill
        self.reference_buffer: torch.Tensor | None = None
        self.storage: list[torch.Tensor] = []
        self.views: list[ReferencePrefix] = []
        self.runners: dict[tuple[int, int], PagedEstimatorGraphs] = {}

    def prepare_reference(self, shared: ReferencePrefix) -> None:
        native = self.native
        if self.reference_buffer is not None and (
            native.step_buffer is not self.reference_buffer
            or native.step_views is None
            or native.step_views.mel_frames != shared.mel_frames
        ):
            native.owner_stream.synchronize()
            self.runners.clear()
            self.reference_buffer = None
        else:
            pass
        if native.reference is shared and native.step_bank is None:
            # note (Codex): A previous hop could not reserve this voice's bank.
            native.reference = None
        else:
            pass

    def stage(
        self, shared: ReferencePrefix, previous: int, adding: int, *, first_step: bool
    ) -> PagedEstimatorGraphs | None:
        native = self.native
        if first_step:
            self.prepare_reference(shared)
        else:
            pass
        if native.reference is not shared:
            native.bind(shared)
        else:
            pass
        if (
            native.step_bank is None
            or native.step_buffer is None
            or native.step_views is None
        ):
            return None
        elif native.step_buffer is not self.reference_buffer:
            self.reference_buffer = native.step_buffer
        else:
            pass
        key = (previous, adding)
        runner = self.runners.get(key)
        if runner is not None:
            return runner
        else:
            layout = self.prefill.layout
            assert layout is not None
        template = native.step_buffer
        while len(self.storage) < previous + adding:
            if (
                torch.cuda.mem_get_info(template.device)[0]
                < MIN_CAPTURE_FREE_BYTES + layout.elements * template.element_size()
            ):
                return None
            else:
                page = template.new_empty(layout.elements)
                self.storage.append(page)
                self.views.append(layout.view(page))
        current = native.step_views
        view = ReferencePrefix(
            reference_key=b"",
            mel_frames=current.mel_frames + previous * layout.width,
            key_values=current.key_values,
            convolution_tails=current.convolution_tails,
        )
        step = PageStep(
            view,
            tuple(self.views[:previous]),
            tuple(self.views[previous : previous + adding]),
        )
        runner = PagedEstimatorGraphs(native, self.prefill, step)
        # note (Codex): All widths execute serially on the same stream and can reuse graph scratch.
        runner.pool = native.pool
        self.runners[key] = runner
        return runner

    def __call__(self, x, mask, mu, t, spks, cond, streaming=False):
        native = self.native
        tx = self.prefill.transaction.get()
        call = self.prefill.current.get()
        inputs = (x, mask, mu, t, spks, cond)
        if (
            call is None
            or call.prefill
            or tx is None
            or not (tx.read_history or tx.keep_frames)
        ):
            if (
                call is not None
                and call.projection_index == 0
                and (native.reference is not call.prefix or native.step_bank is None)
                and get_ident() == native.owner_thread
                and torch.cuda.current_stream(x.device) == native.owner_stream
            ):
                self.prepare_reference(call.prefix)
            else:
                pass
            return native(*inputs, streaming=streaming)
        else:
            shared = self.prefill.cached
            layout = self.prefill.layout
            assert shared is not None and layout is not None
        previous = () if tx.history is None else tx.history.pages
        append = () if tx.candidate is None else tx.candidate.pages[len(previous) :]
        keys = len(shared.key_values) // native.steps
        tails = len(shared.convolution_tails) // native.steps
        step, extra = divmod(call.projection_index, keys)
        assert not extra and call.convolution_index == step * tails
        runner = None
        if (
            get_ident() == native.owner_thread
            and torch.cuda.current_stream(x.device) == native.owner_stream
        ):
            try:
                runner = self.stage(
                    shared, len(previous), len(append), first_step=step == 0
                )
            except torch.OutOfMemoryError:
                pass
        else:
            pass
        if runner is None:
            view = ReferencePrefix(
                reference_key=b"",
                mel_frames=shared.mel_frames + len(previous) * layout.width,
                key_values=shared.key_values[step * keys : (step + 1) * keys],
                convolution_tails=shared.convolution_tails[
                    step * tails : (step + 1) * tails
                ],
            )
            pages = PageStep(
                view,
                tuple(layout.view(p[step]) for p in previous),
                tuple(layout.view(p[step]) for p in append),
            )
            with self.prefill.apply_pages(pages):
                result = native.estimator(*inputs, streaming=streaming)
        else:
            assert native.step_buffer is not None and native.step_bank is not None
            native.step_buffer.copy_(native.step_bank[step])
            if previous:
                torch._foreach_copy_(  # noqa: leading-underscore
                    self.storage[: len(previous)], [p[step] for p in previous]
                )
            else:
                pass
            with self.prefill.apply_pages(runner.pages):
                result = runner(*inputs, streaming=streaming)
            if append:
                torch._foreach_copy_(  # noqa: leading-underscore
                    [p[step] for p in append],
                    self.storage[len(previous) : len(previous) + len(append)],
                )
            else:
                pass
        if append:
            tx.written_steps += 1
        else:
            pass
        call.projection_index += keys
        call.convolution_index += tails
        return result
