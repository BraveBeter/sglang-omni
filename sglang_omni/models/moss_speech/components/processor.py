# SPDX-License: Apache-2.0
"""Canonical input builder for MOSS-Speech (P2/T2.3).

Model-input semantics only: official chat-template segment construction and
the (B, L, 2) token grid with left-padded collate, faithful to the locked
reference processor (`processing_moss_speech.py` @ cff0d0b, see
docs/design/moss_speech/p0/MODEL_CONTRACT.md §3). Codec encoding lives in
the P1 adapter and is NOT duplicated here — this module consumes per-turn
codes produced by preprocessing.

Reference quirks preserved verbatim:
- the default system turn is appended AFTER all conversation turns (not
  first) when the conversation has no system turn;
- processor default system prompts differ from the interface defaults
  (interface prompts arrive as explicit system turns);
- the repo's chat_template.jinja is NOT used;
- text tokenization uses add_special_tokens=False (no truncation here;
  length limits are enforced by request validation).

Grid convention: channels-last (B, L, 2) at the processor/generate
boundary; the (B, 2, L) conversion is the P3 runner's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from ..payload_types import (
    AUDIO_PAD_TOKEN_ID,
    EOSP_TOKEN_ID,
    MODALITY_PAD_TOKEN_ID,
    SOSP_TOKEN_ID,
)

# Reference processor defaults (used only when no system turn is present).
DEFAULT_SYSTEM_PROMPTS = {
    "text": "You are a helpful assistant. Respond with text outputs.",
    "audio": "You are a helpful assistant. Respond with spoken outputs.",
}


class ContractViolation(ValueError):
    """Raised when canonical input construction violates the model contract."""


@dataclass
class CanonicalInput:
    grid: torch.Tensor  # (B, L, 2) int64
    attention_mask: torch.Tensor  # (B, L) int64
    prompt_len: int  # L (unpadded single-sample length)


class MossSpeechGridProcessor:
    """Builds canonical token grids from canonical turns + per-turn codes."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ContractViolation("tokenizer must define pad_token_id")

    # ------------------------------------------------------------------ text
    def _text_ids(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)

    @staticmethod
    def _audio_segment(codes: Sequence[int]) -> torch.Tensor:
        """Audio turn grid: text-ch [sosp, pad×(n-1)] + audio-ch [pad, codes, eosp]."""
        n = len(codes)
        if n == 0:
            raise ContractViolation("audio turn with zero codes")
        text_ch = torch.full((1, n + 2), MODALITY_PAD_TOKEN_ID, dtype=torch.long)
        text_ch[0, 0] = SOSP_TOKEN_ID
        audio_ch = torch.full((1, n + 2), AUDIO_PAD_TOKEN_ID, dtype=torch.long)
        audio_ch[0, 1 : n + 1] = torch.tensor(list(codes), dtype=torch.long)
        audio_ch[0, n + 1] = EOSP_TOKEN_ID
        return torch.cat([text_ch, audio_ch], dim=0)  # (2, n+2)

    @staticmethod
    def _text_segment(text: str, ids: Optional[torch.Tensor]) -> torch.Tensor:
        tokenized = ids if ids is not None else torch.empty(1, 0, dtype=torch.long)
        audio_ch = torch.full_like(tokenized, AUDIO_PAD_TOKEN_ID)
        return torch.cat([tokenized, audio_ch], dim=0)

    # ------------------------------------------------------------------ build
    def build(
        self,
        turns: Sequence[dict],
        audio_codes: Sequence[Sequence[int]],
        output_modality: str,
    ) -> CanonicalInput:
        """Build a single-sample canonical input.

        `turns`: canonical turn dicts (role/kind/text). Audio turns carry no
        text; their codes come from `audio_codes` in turn order.
        """
        if output_modality not in ("text", "audio"):
            raise ContractViolation(f"unsupported output modality {output_modality!r}")
        segments: List[torch.Tensor] = []
        code_iter = iter(audio_codes)
        has_system = bool(turns) and turns[0].get("role") == "system"

        for turn in turns:
            role = turn.get("role")
            if role not in ("user", "assistant", "system"):
                raise ContractViolation(f"unsupported role {role!r}")
            segments.append(
                self._text_segment(
                    f"<|im_start|>{role}\n", self._text_ids(f"<|im_start|>{role}\n")
                )
            )
            kind = turn.get("kind", "text")
            if kind == "audio":
                codes = next(code_iter, None)
                if codes is None:
                    raise ContractViolation("missing codes for audio turn")
                segments.append(self._audio_segment(codes))
            else:
                content = turn.get("text")
                if content is None:
                    raise ContractViolation("text turn without content")
                segments.append(self._text_segment(content, self._text_ids(content)))
            segments.append(
                self._text_segment("<|im_end|>\n", self._text_ids("<|im_end|>\n"))
            )

        if not has_system:
            default = DEFAULT_SYSTEM_PROMPTS[output_modality]
            segments.append(
                self._text_segment(
                    "<|im_start|>system\n", self._text_ids("<|im_start|>system\n")
                )
            )
            segments.append(self._text_segment(default, self._text_ids(default)))
            segments.append(
                self._text_segment("<|im_end|>\n", self._text_ids("<|im_end|>\n"))
            )

        prefix = "<|im_start|>assistant\n"
        if output_modality == "audio":
            prefix += "<|object_ref_start|>"
        segments.append(self._text_segment(prefix, self._text_ids(prefix)))

        grid = torch.cat(segments, dim=1)  # (2, L)
        grid = grid.permute(1, 0).unsqueeze(0).contiguous()  # (1, L, 2)
        return CanonicalInput(
            grid=grid,
            attention_mask=torch.ones(1, grid.shape[1], dtype=torch.long),
            prompt_len=int(grid.shape[1]),
        )

    # ------------------------------------------------------------------ batch
    def collate(self, samples: Sequence[CanonicalInput]) -> CanonicalInput:
        """Left-padded batch collate (reference semantics)."""
        if not samples:
            raise ContractViolation("empty batch")
        max_len = max(s.grid.shape[1] for s in samples)
        grids, masks = [], []
        for s in samples:
            length = s.grid.shape[1]
            pad = torch.full(
                (1, max_len - length, 2), AUDIO_PAD_TOKEN_ID, dtype=torch.long
            )
            pad[:, :, 0] = self.pad_token_id
            grids.append(torch.cat([pad, s.grid], dim=1))
            masks.append(
                torch.cat(
                    [
                        torch.zeros(1, max_len - length, dtype=torch.long),
                        s.attention_mask,
                    ],
                    dim=1,
                )
            )
        return CanonicalInput(
            grid=torch.cat(grids, dim=0),
            attention_mask=torch.cat(masks, dim=0),
            prompt_len=max_len,
        )
