# SPDX-License-Identifier: Apache-2.0
"""MOSS-Speech FSM and dual-channel sampling (locked reference semantics).

Every rule here is code-verified against the reference revision
(modeling_moss_speech.py, MossSpeechGenerationMixin; see
docs/design/moss_speech/p3/01_native_design.md §2.2-2.3):

- Mode transition is evaluated BEFORE the forward of each step, reading the
  last appended grid row:
      text-mode & text_ch == sosp -> audio
      audio-mode & audio_ch == eosp -> text
- Initial mode from the prompt's last row: text_ch == modality_pad -> audio;
  then audio_ch == audio_pad -> text (audio branch first, text overrides).
- Audio-channel constraints applied to raw logits BEFORE processors:
      audio[16385:] = -inf (hard);  audio[16384] = -inf while
      generating_length < min_new_tokens.
- Per-channel repetition penalty consumes the FULL per-channel history
  (prompt rows + generated rows, including the modality_pad values written
  into the text channel during audio segments).
- Sampling: per-channel argmax (greedy; ties resolve to the LOWEST index,
  matching torch.argmax — 159/1154 reference steps carry exact ties) or
  softmax -> multinomial (seeded per request).
- In audio mode the sampled TEXT token is overwritten with modality_pad
  before the row is appended; in text mode the audio sample is kept in the
  row but ignored by embedding selection.
- Stop: text channel of the last appended row == <|endoftext|>(151643) or
  im_end(151645); the terminating row stays in the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

MODALITY_PAD = 151667
SOSP = 151646
EOSP = 16384
AUDIO_PAD = 512
TEXT_ENDOFTEXT = 151643
IM_END = 151645
AUDIO_FORBIDDEN_FROM = 16385
TEXT_VOCAB = 151680
AUDIO_VOCAB = 16512

MODE_TEXT = 0
MODE_AUDIO = 1


@dataclass(frozen=True)
class MossSamplingParams:
    """Per-request sampling knobs (reference interface applies one set to
    both channels; kept per-channel here for fidelity of the processor
    pipeline)."""

    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    max_new_tokens: int = 200


def initial_mode(prompt_rows: torch.Tensor) -> int:
    """Mode inferred from the prompt's last row (reference order: the
    audio branch is applied first, the text branch second and can win)."""
    last = prompt_rows[-1]
    mode = MODE_TEXT
    if int(last[0]) == MODALITY_PAD:
        mode = MODE_AUDIO
    if int(last[1]) == AUDIO_PAD:
        mode = MODE_TEXT
    return mode


def next_mode(last_row: torch.Tensor, current_mode: int) -> int:
    """Transition evaluated before each forward from the last appended row."""
    if current_mode == MODE_TEXT and int(last_row[0]) == SOSP:
        current_mode = MODE_AUDIO
    if current_mode == MODE_AUDIO and int(last_row[1]) == EOSP:
        current_mode = MODE_TEXT
    return current_mode


def apply_audio_constraints(
    audio_logits: torch.Tensor, generating_length: int, min_new_tokens: int
) -> torch.Tensor:
    """In-place reference rule on the (V_a,) audio logits of one step."""
    audio_logits[AUDIO_FORBIDDEN_FROM:] = float("-inf")
    if generating_length < min_new_tokens:
        audio_logits[EOSP] = float("-inf")
    return audio_logits


def repetition_penalty_scores(
    logits: torch.Tensor, history: torch.Tensor, penalty: float
) -> torch.Tensor:
    """HF RepetitionPenaltyLogitsProcessor math on a single channel row.

    history: 1-D long tensor of that channel's tokens so far (prompt +
    generated). penalty == 1.0 is a no-op.
    """
    if penalty == 1.0 or history.numel() == 0:
        return logits
    seen = torch.unique(history)
    scores = logits.clone()
    sel = scores[seen]
    scores[seen] = torch.where(sel < 0, sel * penalty, sel / penalty)
    return scores


def sample_channel(
    scores: torch.Tensor, do_sample: bool, generator: Optional[torch.Generator] = None
) -> int:
    """Greedy argmax (lowest index on exact ties — torch.argmax semantics) or
    seeded multinomial over softmax(scores)."""
    if not do_sample:
        return int(scores.argmax())
    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def sample_row(
    text_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    params: MossSamplingParams,
    text_history: torch.Tensor,
    audio_history: torch.Tensor,
    generating_length: int,
    mode: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, int]:
    """One generation step's (text_token, audio_token) BEFORE the audio-mode
    overwrite; callers apply :func:`finalize_row` afterwards.

    Order matches the reference: float upcast -> audio constraints ->
    per-channel repetition penalty -> warpers (sampling only) -> sample ->
    audio-mode text overwrite.
    """
    t = text_logits.detach().to(torch.float32).clone()
    a = audio_logits.detach().to(torch.float32).clone()
    a = apply_audio_constraints(a, generating_length, params.min_new_tokens)
    t = repetition_penalty_scores(t, text_history, params.repetition_penalty)
    a = repetition_penalty_scores(a, audio_history, params.repetition_penalty)
    if params.do_sample:
        t = apply_warpers(t, params)
        a = apply_warpers(a, params)
    text_tok = sample_channel(t, params.do_sample, generator)
    audio_tok = sample_channel(a, params.do_sample, generator)
    return text_tok, audio_tok


def apply_warpers(logits: torch.Tensor, params: MossSamplingParams) -> torch.Tensor:
    """temperature -> top_k -> top_p (reference _setup_processors order)."""
    out = logits
    if params.temperature != 1.0:
        out = out / max(params.temperature, 1e-6)
    if params.top_k is not None and params.top_k > 0:
        k = min(int(params.top_k), out.shape[-1])
        if k < out.shape[-1]:
            threshold = torch.topk(out, k).values[-1]
            out = out.masked_fill(out < threshold, float("-inf"))
    if params.top_p is not None and 0.0 < params.top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(out, descending=True)
        probs = torch.softmax(sorted_scores, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        keep = cum - probs < params.top_p
        keep[0] = True
        drop_idx = sorted_idx[~keep]
        out = out.clone()
        out[drop_idx] = float("-inf")
    return out


def finalize_row(text_tok: int, audio_tok: int, mode: int) -> Tuple[int, int]:
    """Audio mode overwrites the text channel with modality_pad; text mode
    keeps the (ignored) audio sample in the row."""
    if mode == MODE_AUDIO:
        return MODALITY_PAD, audio_tok
    return text_tok, audio_tok


def selected_token(text_tok: int, audio_tok: int) -> int:
    """1-D scheduler token: the row's embedding-selected channel value
    (text unless the text channel is modality_pad)."""
    row_text, row_audio = finalize_row(text_tok, audio_tok, MODE_TEXT)
    del row_text
    return int(audio_tok) if text_tok == MODALITY_PAD else int(text_tok)


def row_selected_token(row: Tuple[int, int]) -> int:
    """Selected token of an already-finalized grid row."""
    return int(row[1]) if row[0] == MODALITY_PAD else int(row[0])


def row_is_audio(row: Tuple[int, int]) -> bool:
    return row[0] == MODALITY_PAD


def stop_hit(row: Tuple[int, int]) -> bool:
    """Reference MIMOStoppers watch the TEXT channel of the last row."""
    return row[0] in (TEXT_ENDOFTEXT, IM_END)


def channel_histories(
    prompt_rows: torch.Tensor, output_rows: List[Tuple[int, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full per-channel histories (prompt + generated) for the penalty."""
    gen = torch.tensor(output_rows, dtype=torch.long).reshape(-1, 2)
    full = torch.cat([prompt_rows.to(torch.long), gen], dim=0)
    return full[:, 0].contiguous(), full[:, 1].contiguous()
