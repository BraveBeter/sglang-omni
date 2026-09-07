# MOSS-Speech P0 Golden Fixtures

Parity ground truth exported from the locked reference (see
`docs/design/moss_speech/MODEL_CONTRACT.md` and `p0/01_version_lock.md`).

## Generation settings

- greedy (`do_sample=False`, repetition_penalty=1.1, max_new_tokens=200, min_new=0), seed 0
- reference env: transformers 4.57.1, torch 2.9.1+cu128, A800-SXM4-80GB (bf16)
- exported by `scripts/moss_speech/p0/export_fixtures.py`; determinism gate: rerun tokens bit-equal, sanitized logits max-abs diff = 0.0

## Per-case contents

| file | content |
|---|---|
| `canonical_input.json` | processor output: `input_ids` (L,2) channels-last grid + `attention_mask` (left-padded batch of 1) |
| `tokens_grid.pt` | generated grid (L_new, 2), prompt-stripped (`output_only=True`) |
| `logits_first16.pt` | per-step concat(text_logits[151680], audio_logits[16512]) fp32 — **masked capture**: raw head outputs upcast to fp32 with the audio constraint `audio[16385:] = -inf` already applied; pre-processor (no repetition penalty) and pre-warper. Corrected 2026-09-07 against `_generate_next_tokens_with_scores` (MOD L725–741): `output_logits` records `next_token_logits` AFTER the audio mask. P0 exported with min_new_tokens=0, so eosp is NOT masked here. The same-precision BF16 re-export with all three capture points lives in workspace `artifacts/p3/reference/` |
| `meta.json` | task, conversation, audio codes (full length), input-audio codec codes, audio metadata (24 kHz mono), text |
| `text.txt` / `audio.wav` | decoded outputs |

Cases: `t2t_short`, `t2s_cn`, `s2s_cn`, `s2t_cn`, `mixed_multiturn` (fixed-literal text assistant turn + audio user turn).

## Notes for P3 parity consumers

- The audio channel's `[16385:]` region is already −inf **in the stored tensors** (masked capture, see above); sanitize with `nan_to_num` before diffing (−inf−(−inf)=NaN).
- Full 64-step logits dumps + `_rerun` determinism artifacts live outside git in the workspace `artifacts/p0/fixtures/`.
- `mixed_multiturn` exercises per-turn processor dispatch over mixed text/audio history; `s2*` cases exercise codec encode of user audio.
