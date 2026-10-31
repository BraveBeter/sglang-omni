# SPDX-License-Identifier: Apache-2.0
"""Request-owned dual-channel generation state."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from typing import Literal

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams as EngineSamplingParams

from sglang_omni.models.moss_speech.audio_tasks import RequestValidationError
from sglang_omni.models.moss_speech.fsm import (
    IM_END,
    TEXT_ENDOFTEXT,
    Mode,
    MossSamplingParams,
    initial_mode,
)
from sglang_omni.models.moss_speech.hf_config import (
    MAX_CONTEXT_TOKENS,
    MAX_GENERATION_TOKENS,
)
from sglang_omni.models.moss_speech.payload_types import (
    AUDIO_VOCAB,
    MODALITY_PAD_TOKEN_ID,
    TEXT_VOCAB,
    MossSpeechState,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_REPETITION_PENALTY = 1.1


@dataclass
class MossSpeechSGLangRequestData(SGLangARRequestData):
    prompt_rows: torch.Tensor | None = None
    output_rows: list[tuple[int, int]] = field(default_factory=list)
    mode: Mode = 0
    params: MossSamplingParams = field(default_factory=MossSamplingParams)
    effective_seed: int = 0
    stream_grid_cursor: int = 0
    stream_token_cursor: int = 0
    stream_audio_ended: bool = False
    finished: bool = False
    stop_reason: Literal["stop_token", "length"] | None = None
    rng_generator: torch.Generator | None = None

    def reset(self) -> None:
        """Drop generated state while retaining the prompt and sampling seed."""
        assert self.prompt_rows is not None
        self.output_rows.clear()
        self.output_ids.clear()
        self.generation_steps = 0
        self.stream_grid_cursor = 0
        self.stream_token_cursor = 0
        self.stream_audio_ended = False
        self.finished = False
        self.finish_reason = None
        self.stop_reason = None
        self.rng_generator = None
        self.pending_feedback_queue.clear()
        self.mode = initial_mode(self.prompt_rows)


def build_sglang_moss_request(
    state: MossSpeechState, *, payload: StagePayload
) -> MossSpeechSGLangRequestData:
    temperature = (
        DEFAULT_TEMPERATURE if state.temperature is None else state.temperature
    )
    top_p = DEFAULT_TOP_P if state.top_p is None else state.top_p
    top_k = DEFAULT_TOP_K if state.top_k is None else state.top_k
    penalty = (
        DEFAULT_REPETITION_PENALTY
        if state.repetition_penalty is None
        else state.repetition_penalty
    )
    max_new_tokens = (
        DEFAULT_MAX_NEW_TOKENS if state.max_new_tokens is None else state.max_new_tokens
    )
    if (
        not all(math.isfinite(value) for value in (temperature, top_p, penalty))
        or temperature < 0
        or not 0 < top_p <= 1
        or top_k < -1
        or penalty <= 0
        or not 1 <= max_new_tokens <= MAX_GENERATION_TOKENS
        or state.stop
    ):
        raise RequestValidationError("invalid native sampling parameters")
    else:
        pass
    parameters = MossSamplingParams(
        do_sample=temperature > 0 and top_k != 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=penalty,
        max_new_tokens=max_new_tokens,
    )
    prompt_rows = torch.as_tensor(state.input_grid, dtype=torch.long)
    if prompt_rows.ndim != 2 or prompt_rows.shape[1] != 2 or not len(prompt_rows):
        raise RequestValidationError("native input_grid must have shape (L, 2)")
    else:
        pass
    if state.attention_mask:
        attention_mask = torch.as_tensor(state.attention_mask)
        if (
            attention_mask.ndim != 1
            or attention_mask.numel() != len(prompt_rows)
            or not ((attention_mask == 0) | (attention_mask == 1)).all()
        ):
            raise RequestValidationError("invalid native attention_mask")
        elif (
            not attention_mask.any() or (attention_mask[1:] < attention_mask[:-1]).any()
        ):
            raise RequestValidationError(
                "native attention_mask must be nonempty left padding"
            )
        else:
            prompt_rows = prompt_rows[attention_mask.bool()]
    else:
        pass
    if ((prompt_rows[:, 0] < 0) | (prompt_rows[:, 0] >= TEXT_VOCAB)).any() or (
        (prompt_rows[:, 1] < 0) | (prompt_rows[:, 1] >= AUDIO_VOCAB)
    ).any():
        raise RequestValidationError(
            "native input_grid contains out-of-vocabulary token ids"
        )
    else:
        pass
    if len(prompt_rows) + max_new_tokens > MAX_CONTEXT_TOKENS:
        raise RequestValidationError(
            f"Prompt plus generation budget exceeds {MAX_CONTEXT_TOKENS} tokens",
            reason="context_too_long",
        )
    else:
        pass
    selected_tokens = [
        audio_token if text_token == MODALITY_PAD_TOKEN_ID else text_token
        for text_token, audio_token in prompt_rows.tolist()
    ]
    engine_sampling = EngineSamplingParams(
        max_new_tokens=max_new_tokens, temperature=0.0
    )
    engine_sampling.normalize(None)
    engine_sampling.verify(TEXT_VOCAB)
    engine_request = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=array("q", selected_tokens),
        sampling_params=engine_sampling,
        eos_token_ids={TEXT_ENDOFTEXT, IM_END},
        vocab_size=TEXT_VOCAB,
    )
    engine_request.tokenizer = None
    return MossSpeechSGLangRequestData(
        req=engine_request,
        input_ids=torch.tensor(selected_tokens, dtype=torch.long),
        prompt_rows=prompt_rows,
        mode=initial_mode(prompt_rows),
        params=parameters,
        effective_seed=state.effective_seed,
        max_new_tokens=max_new_tokens,
        enforce_request_limits=True,
        stage_payload=payload,
    )
