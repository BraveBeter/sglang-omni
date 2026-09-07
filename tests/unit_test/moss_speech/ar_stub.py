"""Test-only AR stub for the P2 multi-process smoke (NEVER in formal configs).

Replays pre-stored generation grids (P0 fixtures) as the AR output while
preserving the routing/seed/voice fields, so the real preprocessing ->
routing -> terminal chain can be exercised before the native model exists
(P3). It does not register as a native model and does not generate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_ar_stub_executor(
    model_path: str,
    *,
    audio_grid_file: str | None = None,
    text_grid_file: str | None = None,
    **_ignored: Any,
):
    if not audio_grid_file or not text_grid_file:
        raise FileNotFoundError("ar stub requires audio_grid_file and text_grid_file")
    audio_grid = torch.load(audio_grid_file)
    if audio_grid.dim() == 3:
        audio_grid = audio_grid[0]
    text_grid = torch.load(text_grid_file)
    if text_grid.dim() == 3:
        text_grid = text_grid[0]

    def compute(payload: StagePayload) -> StagePayload:
        data = payload.data
        state = data if isinstance(data, MossSpeechState) else MossSpeechState.from_dict(data)
        grid = audio_grid if state.output_modality == "audio" else text_grid
        state.output_grid = grid.tolist()
        # routing-relevant fields are preserved verbatim (voice/effective_seed)
        return StagePayload(request_id=payload.request_id, request=payload.request, data=state.to_dict())

    return SimpleScheduler(compute)
