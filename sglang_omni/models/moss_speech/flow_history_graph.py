# SPDX-License-Identifier: Apache-2.0
"""Store a bounded request's closed Flow prefix at each Euler step."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod
from threading import get_ident

import torch
import torch.nn.functional as F

from sglang_omni.models.moss_speech.flow_prefix import (
    ReferencePrefix,
    StreamingReferencePrefill,
)
from sglang_omni.models.moss_speech.flow_prefix_graph import StreamingReferenceGraphs
from sglang_omni.models.moss_speech.streaming_flow_graph import (
    MIN_CAPTURE_FREE_BYTES,
    StreamingEstimatorGraphs,
)

PRIVATE_HISTORY_CHUNKS = 4


@dataclass
class PrivateHistoryBank:
    storage: torch.Tensor
    steps: tuple[ReferencePrefix, ...]


@dataclass
class HistoryTransaction:
    history: PrivateHistoryBank | None
    final: bool
    keep_frames: int = 0
    read_history: bool = False
    candidate: PrivateHistoryBank | None = None
    written_steps: int = 0


class PrivateHistoryPrefill(StreamingReferencePrefill):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__(decoder)
        self.transaction: ContextVar[HistoryTransaction | None] = ContextVar(
            "moss_private_history_request", default=None
        )
        self.reading: ContextVar[ReferencePrefix | None] = ContextVar(
            "moss_private_history_read", default=None
        )
        self.writing: ContextVar[ReferencePrefix | None] = ContextVar(
            "moss_private_history_write", default=None
        )
        self.chunk_frames = decoder.estimator.static_chunk_size

    @contextmanager
    def request_scope(self, transaction: HistoryTransaction) -> Iterator[None]:
        token = self.transaction.set(transaction)
        try:
            yield
        finally:
            self.transaction.reset(token)

    @contextmanager
    def stage(
        self, shared: ReferencePrefix, private: ReferencePrefix, *, write: bool
    ) -> Iterator[None]:
        target = self.writing if write else self.reading
        token = target.set(private)
        try:
            with self.read_step(shared):
                yield
        finally:
            target.reset(token)

    def project(
        self, layer: torch.nn.Linear, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        private = self.reading.get()
        call = self.current.get()
        if private is not None:
            assert call is not None
            index = call.projection_index
            call.projection_index += 1
            projected = F.linear(hidden_states, layer.weight, layer.bias)
            return torch.cat(
                (call.prefix.key_values[index], private.key_values[index], projected),
                dim=1,
            )
        else:
            destination = self.writing.get()
        if destination is None:
            return super().project(layer, hidden_states)
        else:
            assert call is not None
            index = call.projection_index
            projected = super().project(layer, hidden_states)
            start = call.prefix.mel_frames
            destination.key_values[index].copy_(
                projected[:, start : start + destination.mel_frames]
            )
            return projected

    def convolve(self, layer: torch.nn.Conv1d, channels: torch.Tensor) -> torch.Tensor:
        call = self.current.get()
        private = self.reading.get()
        if private is not None:
            assert call is not None
            index = call.convolution_index
            call.convolution_index += 1
            return F.conv1d(
                torch.cat((private.convolution_tails[index], channels), dim=-1),
                layer.weight,
                layer.bias,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
            )
        else:
            destination = self.writing.get()
        if destination is not None:
            assert call is not None
            end = destination.mel_frames
            destination.convolution_tails[call.convolution_index].copy_(
                channels[..., end - layer.causal_padding : end]
            )
        else:
            pass
        return super().convolve(layer, channels)

    def solve_euler(
        self, full_solver, noise, timesteps, mu, mask, spks, cond, streaming
    ):
        call = self.current.get()
        transaction = self.transaction.get()
        if call is None or call.prefill or transaction is None:
            return super().solve_euler(
                full_solver, noise, timesteps, mu, mask, spks, cond, streaming
            )
        else:
            shared = call.prefix
            suffix_frames = mu.shape[-1] - shared.mel_frames
        history = transaction.history
        if history is not None and suffix_frames > history.steps[0].mel_frames:
            transaction.read_history = True
            call.prefix = ReferencePrefix(
                reference_key=shared.reference_key,
                mel_frames=shared.mel_frames + history.steps[0].mel_frames,
                key_values=shared.key_values,
                convolution_tails=shared.convolution_tails,
            )
        elif (
            history is None
            and not transaction.final
            and suffix_frames >= (PRIVATE_HISTORY_CHUNKS + 1) * self.chunk_frames
        ):
            # note (Codex): Delay the write until another full hop follows the private prefix.
            transaction.keep_frames = PRIVATE_HISTORY_CHUNKS * self.chunk_frames
        else:
            pass
        try:
            return super().solve_euler(
                full_solver, noise, timesteps, mu, mask, spks, cond, streaming
            )
        finally:
            call.prefix = shared


class PrivateEstimatorGraphs(StreamingEstimatorGraphs):
    def __init__(
        self,
        native: StreamingReferenceGraphs,
        prefill: PrivateHistoryPrefill,
        private_frames: int,
        *,
        write: bool,
    ) -> None:
        super().__init__(native.estimator, native.owner_stream.device)
        shared = native.step_views
        assert shared is not None
        self.prefill = prefill
        self.steps = native.steps
        self.write = write
        self.private_frames = private_frames
        self.shapes = [
            (v.size(0), private_frames, *v.shape[2:]) for v in shared.key_values
        ] + [v.shape for v in shared.convolution_tails]
        self.sizes = [prod(shape) for shape in self.shapes]
        self.key_count = len(shared.key_values)
        self.buffer = shared.key_values[0].new_empty(sum(self.sizes))
        self.private = self.views(self.buffer)
        self.shared = ReferencePrefix(
            reference_key=b"",
            mel_frames=shared.mel_frames + (0 if write else private_frames),
            key_values=shared.key_values,
            convolution_tails=shared.convolution_tails,
        )

    def views(self, buffer: torch.Tensor) -> ReferencePrefix:
        tensors = [
            v.view(shape)
            for v, shape in zip(buffer.split(self.sizes), self.shapes, strict=True)
        ]
        return ReferencePrefix(
            reference_key=b"",
            mel_frames=self.private_frames,
            key_values=tensors[: self.key_count],
            convolution_tails=tensors[self.key_count :],
        )

    def allocate_bank(self) -> PrivateHistoryBank:
        bank = self.buffer.new_empty((self.steps, self.buffer.numel()))
        return PrivateHistoryBank(
            bank, tuple(self.views(step) for step in bank.unbind())
        )

    def run_estimator(self, *inputs: torch.Tensor) -> torch.Tensor:
        with self.prefill.stage(self.shared, self.private, write=self.write):
            return super().run_estimator(*inputs)


class StreamingPrivateHistoryGraphs:
    def __init__(
        self, native: StreamingReferenceGraphs, prefill: PrivateHistoryPrefill
    ) -> None:
        self.native = native
        self.prefill = prefill
        self.shared_buffer: torch.Tensor | None = None
        self.reader: PrivateEstimatorGraphs | None = None
        self.writer: PrivateEstimatorGraphs | None = None

    def stage_for_request(
        self,
        shared: ReferencePrefix,
        frames: int,
        transaction: HistoryTransaction,
        *,
        write: bool,
    ) -> PrivateEstimatorGraphs | None:
        native = self.native
        if (
            self.shared_buffer is not None
            and native.step_views is not None
            and native.step_views.mel_frames != shared.mel_frames
        ):
            native.owner_stream.synchronize()
            self.reader = self.writer = None
            self.shared_buffer = None
        else:
            pass
        if native.reference is not shared:
            native.bind(shared)
        else:
            pass
        if native.step_bank is None or native.step_buffer is None:
            return None
        else:
            pass
        if self.shared_buffer is not native.step_buffer:
            self.reader = self.writer = None
            self.shared_buffer = native.step_buffer
        else:
            pass
        staged = self.writer if write else self.reader
        allocate_bank = write and transaction.candidate is None
        if staged is not None and not allocate_bank:
            return staged
        else:
            assert native.step_views is not None
        views = native.step_views
        template = views.key_values[0]
        elements = sum(
            v.numel() // shared.mel_frames * frames for v in views.key_values
        ) + sum(v.numel() for v in views.convolution_tails)
        required_bytes = (
            elements
            * template.element_size()
            * (int(staged is None) + native.steps * int(allocate_bank))
        )
        free_bytes, _ = torch.cuda.mem_get_info(template.device)
        if free_bytes < MIN_CAPTURE_FREE_BYTES + required_bytes:
            return None
        elif staged is None:
            staged = PrivateEstimatorGraphs(native, self.prefill, frames, write=write)
            if write:
                self.writer = staged
            else:
                self.reader = staged
        else:
            pass
        if allocate_bank:
            transaction.candidate = staged.allocate_bank()
        else:
            pass
        return staged

    def __call__(self, x, mask, mu, t, spks, cond, streaming=False):
        inputs = (x, mask, mu, t, spks, cond)
        call = self.prefill.current.get()
        transaction = self.prefill.transaction.get()
        native = self.native
        if (
            call is None
            or call.prefill
            or transaction is None
            or not (transaction.keep_frames or transaction.read_history)
        ):
            return native(*inputs, streaming=streaming)
        else:
            shared = self.prefill.cached
            assert shared is not None
        keys = len(shared.key_values) // native.steps
        tails = len(shared.convolution_tails) // native.steps
        step, remainder = divmod(call.projection_index, keys)
        assert not remainder and call.convolution_index == step * tails
        write = bool(transaction.keep_frames)
        history = transaction.history
        if write:
            frames = transaction.keep_frames
        else:
            assert history is not None
            frames = history.steps[step].mel_frames
        owner = (
            get_ident() == native.owner_thread
            and torch.cuda.current_stream(x.device) == native.owner_stream
        )
        staged = None
        if owner:
            try:
                staged = self.stage_for_request(
                    shared, frames, transaction, write=write
                )
            except torch.OutOfMemoryError:
                staged = None
        else:
            pass
        if staged is None and write:
            transaction.keep_frames = 0
            transaction.candidate = None
            return native(*inputs, streaming=streaming)
        elif staged is None:
            assert history is not None
            # note (Codex): A committed request can read its own bank eagerly even when staging cannot be allocated.
            view = ReferencePrefix(
                reference_key=b"",
                mel_frames=shared.mel_frames + frames,
                key_values=shared.key_values[step * keys : (step + 1) * keys],
                convolution_tails=shared.convolution_tails[
                    step * tails : (step + 1) * tails
                ],
            )
            with self.prefill.stage(view, history.steps[step], write=False):
                result = native.estimator(*inputs, streaming=streaming)
        else:
            assert native.step_buffer is not None and native.step_bank is not None
            native.step_buffer.copy_(native.step_bank[step])
            if not write:
                assert history is not None
                staged.buffer.copy_(history.storage[step])
            else:
                pass
            with self.prefill.stage(staged.shared, staged.private, write=write):
                result = staged(*inputs, streaming=streaming)
            if write:
                assert transaction.candidate is not None
                transaction.candidate.storage[step].copy_(staged.buffer)
                transaction.written_steps += 1
            else:
                pass
        call.projection_index += keys
        call.convolution_index += tails
        return result
