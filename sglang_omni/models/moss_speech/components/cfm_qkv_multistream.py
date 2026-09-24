# SPDX-License-Identifier: Apache-2.0
"""Model-local, fixed-math Q/K/V stream ownership for qualified CFM shapes."""
from __future__ import annotations

import ctypes as C
from typing import Any, Callable

from .cfm_block_tail import TailBlock, require

WORKSPACE_BYTES = 1048576
KINDS = ("q", "k", "v")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def identity(streams, contexts):
    need(set(streams) == set(contexts) == set(KINDS), "Three Q/K/V owners required")
    result = []
    for kind in KINDS:
        stream, context = streams[kind], contexts[kind]
        need(not context.closed and not context.failed, "Closed or failed Lt context")
        need(
            context.stream.cuda_stream == stream.cuda_stream
            and context.device == stream.device,
            "Context not prebound to its branch stream/device",
        )
        result.append(
            (
                int(stream.cuda_stream),
                int(context.handle.value),
                int(context.workspace.data_ptr()),
                str(context.device),
                id(context),
                C.addressof(context.handle),
                context.version,
                str(context.path),
            )
        )
    need(
        len({x[0] for x in result}) == 3
        and len({x[4] for x in result}) == 3
        and len({x[5] for x in result}) == 3,
        "Branch streams, context owners and handle slots must be independent",
    )
    # CUDA's default main stream has the valid public handle0. Branch streams
    # are explicit non-default owners; all three must remain distinct above.
    need(
        result[0][0] >= 0
        and all(x[0] > 0 for x in result[1:])
        and all(x[1] > 0 and x[2] > 0 for x in result)
        and len({x[3] for x in result}) == 1,
        "Invalid owner or device",
    )
    # 4337 observed public120804 Create returning the same opaque value to
    # independent owners. It is immutable/non-null, not a unique resource ID.
    need(
        all(x[6] == 120804 for x in result) and len({x[7] for x in result}) == 1,
        "Changed qualified Lt library/version",
    )
    intervals = sorted((x[2], x[2] + WORKSPACE_BYTES) for x in result)
    need(
        all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
        "Concurrent branches must not overlap workspace storage",
    )
    return tuple(result)


class QkvSchedule:
    def __init__(self, api, streams, contexts):
        need(not api.is_current_stream_capturing(), "Prepare events outside capture")
        self.api, self.streams, self.contexts = api, dict(streams), dict(contexts)
        self.initial = identity(self.streams, self.contexts)
        self.ready = api.Event(enable_timing=False, external=False)
        self.done = {
            k: api.Event(enable_timing=False, external=False) for k in ("k", "v")
        }
        self.busy = self.closed = False
        self.calls = 0
        self.input_owner = None
        # Initialize event handles before graph capture. The caller has already
        # established model/plan readiness on all three streams.
        self.ready.record(self.streams["q"])
        for kind in ("k", "v"):
            self.done[kind].record(self.streams[kind])
            self.streams["q"].wait_event(self.done[kind])

    def run(self, value, enqueue):
        need(not self.closed and not self.busy, "Closed or reentrant fork/join")
        need(
            identity(self.streams, self.contexts) == self.initial,
            "Changed branch owner",
        )
        main = self.streams["q"]
        need(
            self.api.current_stream(main.device).cuda_stream == main.cuda_stream,
            "Wrong caller stream",
        )
        self.busy = True
        self.input_owner = value
        started = []
        try:
            self.ready.record(main)
            for kind in ("k", "v"):
                with self.api.stream(self.streams[kind]):
                    self.streams[kind].wait_event(self.ready)
                    started.append(kind)
                    try:
                        enqueue(kind, value)
                    finally:
                        self.done[kind].record(self.streams[kind])
            enqueue("q", value)
            self.calls += 1
        finally:
            # Includes tail writes and any startup-only snapshot on each branch.
            # No host sync: main's later SDPA is ordered after these dependencies.
            try:
                for kind in started:
                    main.wait_event(self.done[kind])
            finally:
                self.busy = False

    def close(self):
        if self.closed:
            return
        need(not self.busy, "Cannot close active schedule")
        self.input_owner = None
        self.closed = True
        # Driver first synchronizes/resets graphs, then closes Lt contexts.


class ParallelTailBlock(TailBlock):
    def __init__(self, *args, schedule, **kwargs):
        super().__init__(*args, **kwargs)
        require(
            self.length in (1392, 1424),
            "Only independently qualified N1392/N1424 are admitted",
        )
        self.schedule = schedule
        self.capture_owners = []

    def _parallel_qkv(self, normalized, record, project):
        import torch

        if torch.cuda.is_current_stream_capturing():
            self.capture_owners.append(normalized)
        slots = dict(q=self.query, k=self.key_slot, v=self.value_slot)

        def enqueue(kind, value):
            slots[kind][:, 1280:, :].copy_(record(kind, project(kind, value)))

        self.schedule.run(normalized, enqueue)

    def run(
        self, hidden: Any, mask: Any, *, key: tuple, observe: Callable | None = None
    ) -> Any:
        import torch
        import torch.nn.functional as F

        require(key == self.key, "Changed block prefix identity")
        length = self.length
        require(
            hidden.shape == (2, length, 256)
            and hidden.is_contiguous()
            and hidden.dtype == self.query.dtype
            and hidden.device == self.query.device,
            "Unknown current block input layout",
        )

        def record(name, value):
            if observe is not None:
                observe(name, value)
            return value

        def project(kind, value):
            result = self.plans[kind](value)
            self._projections += 1
            return result

        current = hidden[:, 1280:, :]
        normalized = record("norm1", self.block.norm1(current))
        self._parallel_qkv(normalized, record, project)
        q, k, v = (
            value.view(2, length, 8, 64).transpose(1, 2)
            for value in (self.query, self.key_slot, self.value_slot)
        )
        q = q[..., 1280:, :]
        require(
            q.stride() == (length * 512, 64, 512, 1)
            and q.storage_offset() == 1280 * 512
            and all(
                value.stride() == (length * 512, 64, 512, 1)
                and value.storage_offset() == 0
                for value in (k, v)
            ),
            "Changed original SDPA layout",
        )
        bias = self.block.attn1.prepare_attention_mask(mask, length, 2).view(
            2, 8, length, length
        )
        bias = bias[..., 1280:, :]
        require(
            int(torch._fused_sdp_choice(q, k, v, bias, 0.0, False))
            == int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
            "Unqualified tail SDPA backend",
        )
        attention = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        self._sdpa += 1
        attention = record(
            "sdpa", attention.transpose(1, 2).reshape(2, -1, 512).to(q.dtype)
        )
        output = record("to_out", project("to_out", attention))
        output = record(
            "scaled",
            self.block.attn1.to_out[1](output) / self.block.attn1.rescale_output_factor,
        )
        residual = record("residual1", output + current)
        normalized = record("norm3", self.block.norm3(residual))
        first = record("ff_first", project("ff_first", normalized))
        activated = record("gelu", self.block.ff.net[0].gelu(first))
        ff = record("ff_last", project("ff", self.block.ff.net[1](activated)))
        output = torch.cat((self.output_prefix, ff + residual), dim=1)
        self._blocks += 1
        return record("output", output)


class QkvBranches:
    """Own two fixed branch streams/contexts; the parent owns the main context.

    prepare/bind are startup-only. Replay executes the already captured graph.
    Keep all event schedules until every graph has been reset.
    """

    def __init__(self, api, context_factory, main_context):
        self.api, self.context_factory = api, context_factory
        self.main = main_context
        self.streams = {}
        self.contexts = {}
        self.schedules = []
        self.closed = False

    def prepare(self):
        need(
            not self.closed and not self.api.is_current_stream_capturing(),
            "Branches prepare outside capture",
        )
        main = self.api.current_stream(self.main.device)
        need(
            main.cuda_stream == self.main.stream.cuda_stream,
            "Main context not bound to startup stream",
        )
        for kind in ("k", "v"):
            if kind not in self.streams:
                self.streams[kind] = self.api.Stream(device=self.main.device)
            stream = self.streams[kind]
            stream.wait_stream(main)
            with self.api.stream(stream):
                if kind not in self.contexts:
                    self.contexts[kind] = self.context_factory(self.main.device)
                self.contexts[kind].prepare_stream()

    def bind(self):
        need(
            not self.closed and set(self.contexts) == {"k", "v"}, "Unprepared branches"
        )
        main = self.api.current_stream(self.main.device)
        need(
            main.cuda_stream == self.main.stream.cuda_stream,
            "Main context not bound to capture stream",
        )
        for stream in self.streams.values():
            stream.wait_stream(main)
        schedule = QkvSchedule(
            self.api, dict(q=main, **self.streams), dict(q=self.main, **self.contexts)
        )
        self.schedules.append(schedule)
        return schedule

    def call(self, kind, function, value):
        need(not self.closed and kind in ("k", "v"), "Unknown or closed branch")
        main = self.api.current_stream(self.main.device)
        need(
            main.cuda_stream == self.main.stream.cuda_stream,
            "Wrong startup caller stream",
        )
        stream = self.streams[kind]
        stream.wait_stream(main)
        try:
            with self.api.stream(stream):
                return function(value)
        finally:
            main.wait_stream(stream)

    @property
    def owners(self):
        return tuple(t for context in self.contexts.values() for t in context.owners)

    def close(self):
        if self.closed:
            return
        # A synchronization failure must preserve every strong reference.
        self.api.synchronize(self.main.device)
        for schedule in self.schedules:
            schedule.close()
        for context in self.contexts.values():
            context.close()
        self.schedules.clear()
        self.contexts.clear()
        self.streams.clear()
        self.closed = True
