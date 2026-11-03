# SPDX-License-Identifier: Apache-2.0
"""Build the per-request MIMO text/audio grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.moss_speech.payload_types import (
    AUDIO_PAD_TOKEN_ID,
    EOSP_TOKEN_ID,
    MODALITY_PAD_TOKEN_ID,
    SOSP_TOKEN_ID,
    CanonicalTurn,
    OutputModality,
)

DEFAULT_SYSTEM_PROMPTS = {
    "text": "You are a helpful assistant. Respond with text outputs.",
    "audio": "You are a helpful assistant. Respond with spoken outputs.",
}


class ContractViolation(ValueError):
    """An input cannot be represented by the model's token grid."""


@dataclass(kw_only=True)
class CanonicalInput:
    grid: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int


class MossSpeechGridProcessor:
    """Construct canonical segments from text and previously encoded audio."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def text_segment(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        text_channel = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        return torch.cat(
            [text_channel, torch.full_like(text_channel, AUDIO_PAD_TOKEN_ID)], dim=0
        )

    @staticmethod
    def audio_segment(codes: Sequence[int]) -> torch.Tensor:
        if not codes:
            raise ContractViolation("audio turn with zero codes")
        else:
            pass
        code_count = len(codes)
        text_channel = torch.full(
            (1, code_count + 2), MODALITY_PAD_TOKEN_ID, dtype=torch.long
        )
        text_channel[0, 0] = SOSP_TOKEN_ID
        audio_channel = torch.full(
            (1, code_count + 2), AUDIO_PAD_TOKEN_ID, dtype=torch.long
        )
        audio_channel[0, 1 : code_count + 1] = torch.tensor(list(codes))
        audio_channel[0, code_count + 1] = EOSP_TOKEN_ID
        return torch.cat([text_channel, audio_channel], dim=0)

    def build(
        self,
        turns: Sequence[CanonicalTurn],
        audio_codes: Sequence[Sequence[int]],
        output_modality: OutputModality,
    ) -> CanonicalInput:
        if output_modality not in ("text", "audio"):
            raise ContractViolation(f"unsupported output modality {output_modality!r}")
        else:
            pass
        segments: list[torch.Tensor] = []
        remaining_codes = iter(audio_codes)
        for turn in turns:
            role = turn["role"]
            if role not in ("user", "assistant", "system"):
                raise ContractViolation(f"unsupported role {role!r}")
            else:
                pass
            segments.append(self.text_segment(f"<|im_start|>{role}\n"))
            if turn["kind"] == "audio":
                codes = next(remaining_codes, None)
                if codes is None:
                    raise ContractViolation("missing codes for audio turn")
                else:
                    segments.append(self.audio_segment(codes))
            else:
                content = turn["text"]
                if content is None:
                    raise ContractViolation("text turn without content")
                else:
                    segments.append(self.text_segment(content))
            segments.append(self.text_segment("<|im_end|>\n"))

        # note (xrluan): The trained template appends its default system turn after the conversation.
        if not turns or turns[0]["role"] != "system":
            segments.append(self.text_segment("<|im_start|>system\n"))
            segments.append(self.text_segment(DEFAULT_SYSTEM_PROMPTS[output_modality]))
            segments.append(self.text_segment("<|im_end|>\n"))
        else:
            pass
        prefix = "<|im_start|>assistant\n"
        if output_modality == "audio":
            prefix += "<|object_ref_start|>"
        else:
            pass
        segments.append(self.text_segment(prefix))
        grid = torch.cat(segments, dim=1).permute(1, 0).unsqueeze(0).contiguous()
        return CanonicalInput(
            grid=grid,
            attention_mask=torch.ones(1, grid.shape[1], dtype=torch.long),
            prompt_len=grid.shape[1],
        )
