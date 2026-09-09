"""Tiny probe factories (local CPU wiring debug only)."""

from __future__ import annotations

from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_probe_preprocessing(model_path: str):
    def compute(payload: StagePayload) -> StagePayload:
        from sglang_omni.models.moss_speech.request_builders import (
            normalize_and_validate,
        )

        print(f"[probe-pre] computing {payload.request_id}", flush=True)
        state = normalize_and_validate(payload.request, request_id=payload.request_id)
        state.input_grid = [[1, 512], [2, 512]]
        state.attention_mask = [1, 1]
        state.prompt_grid_len = 2
        print(
            f"[probe-pre] done {payload.request_id} modality={state.output_modality}",
            flush=True,
        )
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=state.to_dict()
        )

    return SimpleScheduler(compute)


def create_probe_ar(model_path: str):
    def compute(payload: StagePayload) -> StagePayload:
        from sglang_omni.models.moss_speech.payload_types import MossSpeechState

        print(f"[probe-ar] computing {payload.request_id}", flush=True)
        state = (
            payload.data
            if isinstance(payload.data, MossSpeechState)
            else MossSpeechState.from_dict(payload.data)
        )
        state.output_grid = (
            [[151645, 0]] if state.output_modality == "text" else [[151667, 100]]
        )
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=state.to_dict()
        )

    return SimpleScheduler(compute)


def create_probe_text_terminal(model_path: str):
    def compute(payload: StagePayload) -> StagePayload:
        from sglang_omni.models.moss_speech.payload_types import MossSpeechState

        state = (
            payload.data
            if isinstance(payload.data, MossSpeechState)
            else MossSpeechState.from_dict(payload.data)
        )
        print(
            f"[probe-terminal] got {payload.request_id} modality={state.output_modality}",
            flush=True,
        )
        state.generated_text = "probe done"
        state.__dict__["probe"] = "done"
        data = state.to_dict()
        data["probe"] = "done"
        return StagePayload(
            request_id=payload.request_id, request=payload.request, data=data
        )

    return SimpleScheduler(compute)
