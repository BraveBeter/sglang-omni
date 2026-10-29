# SPDX-License-Identifier: Apache-2.0
"""Own live AR requests and admit consecutive requests of one modality."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import CancelledError, Future

from sglang_omni.models.moss_speech.payload_types import OutputModality
from sglang_omni.models.moss_speech.request_data import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel
from sglang_omni.scheduling.types import DeferredAdmission

CANCELLED_REQUEST_HISTORY_LIMIT = 4096


def cleanup_ar_request_state(request_data: MossSpeechSGLangRequestData) -> None:
    request_data.reset()
    request_data.finished = True


class MossSpeechRequestLifecycle:
    """Own requests with grouped admission or optional mixed-output admission."""

    def __init__(
        self, model: MossSpeechSGLangModel, *, mixed_modalities: bool = False
    ) -> None:
        if not isinstance(mixed_modalities, bool):
            raise ValueError("mixed_modalities must be a bool")
        else:
            pass
        self.model = model
        self.mixed_modalities = mixed_modalities
        self.lock = threading.RLock()
        self.requests: dict[str, MossSpeechSGLangRequestData] = {}
        self.active: dict[str, OutputModality] = {}
        self.pending: deque[tuple[str, OutputModality, Future[None]]] = deque()
        self.cancelled: set[str] = set()
        self.cancelled_order: deque[str] = deque()

    def admit(
        self, request_data: MossSpeechSGLangRequestData, modality: OutputModality
    ) -> MossSpeechSGLangRequestData | DeferredAdmission[MossSpeechSGLangRequestData]:
        assert request_data.req is not None
        request_id = request_data.req.rid
        with self.lock:
            if request_id in self.cancelled:
                raise CancelledError(request_id)
            elif request_id in self.requests:
                raise ValueError(f"Request {request_id} already exists")
            else:
                self.requests[request_id] = request_data
            if self.mixed_modalities or (
                not self.pending
                and (not self.active or modality in self.active.values())
            ):
                self.active[request_id] = modality
                return request_data
            else:
                ready: Future[None] = Future()
                self.pending.append((request_id, modality, ready))
                return DeferredAdmission(value=request_data, ready=ready)

    def release(self, request_id: str, *, aborted: bool = False) -> None:
        with self.lock:
            if aborted and request_id not in self.cancelled:
                self.cancelled.add(request_id)
                self.cancelled_order.append(request_id)
                if len(self.cancelled_order) > CANCELLED_REQUEST_HISTORY_LIMIT:
                    self.cancelled.discard(self.cancelled_order.popleft())
                else:
                    pass
            else:
                pass
            request_data = self.requests.pop(request_id, None)
            if request_data is not None:
                cleanup_ar_request_state(request_data)
            else:
                pass
            self.active.pop(request_id, None)
            retained: deque[tuple[str, OutputModality, Future[None]]] = deque()
            for pending_id, modality, ready in self.pending:
                if pending_id == request_id:
                    ready.cancel()
                else:
                    retained.append((pending_id, modality, ready))
            self.pending = retained
            for logits_owner in list(self.model.dual_logits_by_request_id):
                if logits_owner[0] == request_id:
                    self.model.dual_logits_by_request_id.pop(logits_owner)
                else:
                    pass
            if not self.active and self.pending:
                next_modality = self.pending[0][1]
                while self.pending and self.pending[0][1] == next_modality:
                    pending_id, modality, ready = self.pending.popleft()
                    if not ready.cancelled():
                        self.active[pending_id] = modality
                        ready.set_result(None)
                    else:
                        pass
            else:
                pass

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                "requests": len(self.requests),
                "active": len(self.active),
                "pending": len(self.pending),
            }
