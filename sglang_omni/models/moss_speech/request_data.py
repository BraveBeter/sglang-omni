# SPDX-License-Identifier: Apache-2.0
"""Native engine request state, loaded only when constructing an AR request."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass
class MossSpeechSGLangRequestData(SGLangARRequestData):
    """Model-runner-visible request state; the sglang backend reads the
    base-class bookkeeping (suppress_tokens / synced / feedback queues ...)."""

    # 1-D selected-token stream (mirrors zonos2: the prefill flow iterates it)
    input_ids: Any = None
    # canonical dual-channel prompt rows (L, 2), cpu int64
    prompt_rows: Any = None
    # generated grid rows so far: list[(text, audio)]
    output_rows: list = None
    # FSM mode (0=text, 1=audio); initialized from the prompt's last row
    mode: int = 0
    # per-request sampling knobs (reference semantics; see fsm.py)
    params: Any = None
    effective_seed: int = 0
    generation_steps: int = 0
    # terminal bookkeeping
    finished: bool = False
    stop_reason: Any = None
    # lazily-created per-request RNG (seeded from effective_seed); greedy
    # paths never draw. Owned by this data object; freed with it.
    rng_generator: Any = None

    def __post_init__(self) -> Any:
        if self.output_rows is None:
            self.output_rows = []
        import collections

        if self.pending_feedback_queue is None:
            self.pending_feedback_queue = collections.deque()

    def reset(self) -> None:
        """Idempotent reset for retry: drop generated state, keep prompt."""
        self.output_rows = []
        self.generation_steps = 0
        self.finished = False
        self.stop_reason = None
        self.rng_generator = None
        if self.pending_feedback_queue is not None:
            self.pending_feedback_queue.clear()
        from sglang_omni.models.moss_speech.fsm import initial_mode

        self.mode = initial_mode(self.prompt_rows)
