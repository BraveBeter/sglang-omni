# P0-03 Reference Four-Mode Runs (T0.4)

Date: 2026-09-04. Node: slurmd-6 (A800-SXM4-80GB). Driver: `sglang-omni/scripts/moss_speech/p0/run_reference.py` (sbatch log `artifacts/p0/sbatch_ref_3627.log`, outputs `artifacts/p0/runs_reference_greedy/`).

## Configuration

- sampling: **greedy** (do_sample=False, rep_penalty 1.1, max_new 200), seed 0
- model/codec: locked local dirs (`models/MOSS-Speech`, `models/MOSS-Speech-Codec`)
- voice prompt for audio outputs: `repos/MOSS-Speech/assets/prompt-cn.wav|prompt-en.wav` (official default path in code is buggy — see contract §7.1)
- runtime shims (reference code untouched): no-op streamer (transformers 4.57.1 kwarg filtering), soundfile-backed `torchaudio.load`

## Results

| case | task | gen wall s | det rerun | notes |
|---|---|---|---|---|
| t2t_short | text→text | ~1 | ✅ | clean one-sentence self-intro |
| t2t_long | text→text (long prompt) | ~2 | ✅ | 3-bullet summary as instructed |
| t2s_cn | text→speech | ~7 | ✅ | 24 kHz mono wav; decode via prompt-cn |
| t2s_en | text→speech | ~7 | ✅ | 24 kHz mono; prompt-en voice |
| s2t_cn | speech→text | 6.0 | ✅ | answers the spoken gambling-law question in Chinese |
| s2t_en | speech→text | 6.2 | ✅ | answers spoken friendship question in English |
| s2s_cn | speech→speech | 8.3 (+2.2 decode) | ✅ | 15.92 s audio |
| s2s_en | speech→speech | 8.5 (+3.1 decode) | ✅ | 15.92 s audio |
| mixed t1/t2/t3 | text Q/A → audio Q/A → audio Q→text A | — | ✅ | per-turn processor dispatch over mixed history works |

- All 8 single-mode cases: greedy rerun **token-identical** (`det_ok=true`).
- Audio metadata: sr 24000, mono, float32.
- Discovery: `assets/prompt-*.wav` are **spoken questions** (~27 s CN / ~36 s EN, 44.1 kHz), doubling as default voice prompts — not neutral enrollment clips.
- AR generation stops early via `<|im_end|>` on the text channel; 200-step cap hit only by design for long audio cases.

## VRAM (A800-80G, bf16)

- AR peak ≈ 36.8 GiB (weights ≈ 17.1 GiB + KV + activations), codec decode peak ≈ 37.3 GiB (codec loaded on same device); fits 80G comfortably; 24G feasibility deferred to P5 with 40-layer KV accounting from `05_trace_and_kv.md`.

## Reproduction

```bash
sbatch scripts/p0_run_reference.sbatch   # workspace launcher
# or directly on a compute node:
PYTHONPATH=repos/MOSS-Speech:repos/MOSS-Speech/Matcha-TTS \
.venv-p0/bin/python sglang-omni/scripts/moss_speech/p0/run_reference.py \
  --model-path models/MOSS-Speech --codec-path models/MOSS-Speech-Codec \
  --assets-dir repos/MOSS-Speech/assets --out-dir <out> --sampling greedy --seed 0 --check-determinism
```
