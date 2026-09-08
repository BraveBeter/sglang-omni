# SPDX-License-Identifier: Apache-2.0
"""Engine-local request ownership and fair output-modality admission."""
from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import CancelledError, Future
from typing import Any

from sglang_omni.scheduling.types import DeferredAdmission


class MossSpeechRequestLifecycle:
    """Allow same-modality batches and defer the next modality without blocking."""

    def __init__(self, model: Any = None) -> None:
        self.model = model
        self.lock = threading.RLock()
        self.requests: dict[str, Any] = {}
        self.active: dict[str, str] = {}
        self.pending: deque = deque()
        self.cancelled: set[str] = set()
        self.cancelled_order: deque[str] = deque()

    def admit(self, data: Any, modality: str) -> Any:
        rid = data.req.rid
        with self.lock:
            if rid in self.cancelled:
                raise CancelledError(rid)
            if rid in self.requests:
                raise ValueError(f"duplicate live MOSS-Speech request: {rid}")
            self.requests[rid] = data
            if not self.pending and (
                not self.active or modality in self.active.values()
            ):
                self.active[rid] = modality
                return data
            ready: Future = Future()
            self.pending.append((rid, modality, ready))
            return DeferredAdmission(value=data, ready=ready)

    def release(self, rid: str, *, aborted: bool = False) -> None:
        from sglang_omni.models.moss_speech.request_builders import (
            cleanup_ar_request_state,
        )

        with self.lock:
            if aborted and rid not in self.cancelled:
                # Match the scheduler's no-reuse policy for aborted request IDs,
                # including abort while its CPU builder is still executing.
                self.cancelled.add(rid)
                self.cancelled_order.append(rid)
                if len(self.cancelled_order) > 4096:
                    self.cancelled.discard(self.cancelled_order.popleft())
            data = self.requests.pop(rid, None)
            if data is not None:
                cleanup_ar_request_state(data)
            self.active.pop(rid, None)
            retained = deque()
            for pending_rid, modality, future in self.pending:
                if pending_rid == rid:
                    future.cancel()
                else:
                    retained.append((pending_rid, modality, future))
            self.pending = retained
            store = getattr(self.model, "_dual_logits_by_rid", {})
            for key in list(store):
                if key[0] == rid:
                    store.pop(key, None)
            if not self.active and self.pending:
                group = self.pending[0][1]
                while self.pending and self.pending[0][1] == group:
                    next_rid, modality, future = self.pending.popleft()
                    if not future.cancelled():
                        self.active[next_rid] = modality
                        future.set_result(None)

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                "requests": len(self.requests),
                "active": len(self.active),
                "pending": len(self.pending),
            }
