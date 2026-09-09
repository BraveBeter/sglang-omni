# MOSS-Speech P0 Golden Fixtures

Historical FP32 fixtures exported from MOSS-Speech revision
`cff025bb41d8459d59abac0b5e44aba7f659ec9e`, codec revision
`eeec733e4e1dea7da444d332d8e1621ef257414c`, and the reference driver
`OpenMOSS/MOSS-Speech@1ea408a10d07b9fdc7a27bce19d1211b62067784`.

## Generation settings

- greedy (`do_sample=False`, repetition_penalty=1.1, max_new_tokens=200, min_new=0), seed 0
- reference env: transformers 4.57.1, torch 2.9.1+cu128, A800-SXM4-80GB. These P0 exports used FP32 AR computation; FP32 file storage alone does not determine computation dtype (corrected by P3 baseline audit).
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

- The audio channel's `[16385:]` region is already −inf **in the stored tensors** (masked capture, see above); first compare finite/inf/sign patterns and reject NaNs, then compare only finite entries. Do not use `nan_to_num` to hide mismatches (−inf−(−inf)=NaN).
- Full 64-step logits dumps + `_rerun` determinism artifacts live outside git in the workspace `artifacts/p0/fixtures/`.
- `mixed_multiturn` exercises per-turn processor dispatch over mixed text/audio history; `s2*` cases exercise codec encode of user audio.

- P3 native greedy targets are the separate explicit-BF16 `artifacts/p3/reference/` grids, not these P0 FP32 grids. Missing long-case captures are supplemented in `artifacts/p3/reference_complete_3838/`, with original grid and capture hashes unchanged. P0 files are preserved for processor/codec and historical regression.
- Native parity requires an independently generated BF16 reference. These P0 FP32 grids are not its expected output; no native result replaces a reference value.

## Archived large captures

The five unused `logits_first16.pt` captures (53,829,570 bytes) are no longer
in the current tree. Their immutable source commit, paths, sizes and SHA256
are listed in `sglang-omni/tests/fixtures/moss_speech/archived_logits.json`.
They remain available in Git history; this cleanup does not rewrite history
or claim to shrink existing clones. From the workspace root, restore one with:

```bash
git -C sglang-omni show 3bb5ba4:tests/fixtures/moss_speech/t2t_short/logits_first16.pt > /tmp/logits_first16.pt
sha256sum /tmp/logits_first16.pt
```

Check the hash against the manifest before using the file. Fetch the manifest's
source commit first if your clone is shallow. These FP32 P0 captures are
historical diagnostics, not the current BF16 parity target. The two small WAVs
remain as codec regression outputs (1,528,408 bytes total), alongside token grids
and canonical inputs. They do not require an external asset service.
