# P1-02 Alignment Plan & Frozen Thresholds (T1.3)

> **Written and frozen BEFORE the adapter acceptance run** (protocol order:
> calibration on reference → freeze → adapter export → comparison). Any
> threshold change after seeing adapter results invalidates the run.

## Samples

- **Inputs (encode)**: `repos/MOSS-Speech/assets/prompt-cn.wav` (44.1 kHz, 27.43 s),
  `prompt-en.wav` (44.1 kHz); P0 fixture user audios are these same two assets
  (s2t_cn/s2s_cn/mixed t3 = prompt-cn, mixed t2 = prompt-en), so the corpus is
  the two assets under path / `(wav, sr)` / tensor input forms.
- **Codes (decode)**: extracted from P0 fixture token grids — `t2s_cn`
  (~198 codes, eosp-less truncation), `s2s_cn` (~198, truncation),
  `mixed_multiturn` (105, properly eosp-terminated); synthetic edges:
  single code `[100]`, 500-code repeat of real codes, empty list.
- **Voices**: prompt-cn, prompt-en.
- Environment fingerprint: revision `eeec733e…`, GPU A800-SXM4-80GB, codec FP32,
  adapter `.venv-omni` (torch 2.9.1+cu128, transformers 5.12.1) vs reference
  `.venv-p0` (torch 2.9.1+cu128, transformers 4.57.1).

## Frozen acceptance rules

| # | check | rule (machine-checked) |
|---|---|---|
| A1 | reference encode self-consistency | per-sample codes identical across 2 repeats, batch_size {1, 4, 128} and mixed-length batch — else reference itself is order-sensitive and "batch-safe" is void |
| A2 | adapter encode vs reference | per-code equality, exact, every input form × batch config |
| A3 | reference decode self-consistency | blake2b identical across 2 repeats × seeds {0, 1} × 2 voices (also proves seed-inertness) |
| A4 | voice conditioning tensors | prompt_token exact; prompt_feat / embedding exact (allclose atol=0) |
| A5 | adapter decode vs reference | waveform blake2b equality, every (codes × voice) case; shape=(codes×1920,) ±1920, sr=24000, finite — hard conditions |
| A6 | edges | single code / 500 codes / eosp-truncated: follow A5; empty codes: both sides must raise a clear error; out-of-range: adapter rejects (documented stricter-than-reference deviation, see below) |
| A7 | RNG isolation | python/numpy/torch-CPU/torch-CUDA RNG states bit-identical before vs after every adapter call |
| A8 | request isolation | interleaved decode(codesA, voiceCN) / decode(codesB, voiceEN) × 2 orders == solo-run hashes; failure injection (bad voice shape) → cleanup → next request recovers and matches solo hash; duplicate cleanup safe |
| A9 | decode state | after each finalize-decode, adapter reports zero active sessions |

**Out-of-range deviation (pre-registered):** the reference flow embedding table
is 20480-wide, so codes in [16384, 20480) decode to garbage without error in
the reference; the adapter rejects anything outside [0, 16384) with a clear
`ValueError`. Stricter validation is intentional (codec codes are defined
[0, 16384)); recorded in MODEL_CONTRACT §7 quirks.

**No tolerance invention:** if A5 hash comparison fails on FP32, the run FAILS
and the difference is root-caused (no SNR/mel fallback threshold exists at
FP32; spectral metrics are only recorded diagnostically for the optional
bf16/fp16 experiments, which may not become defaults without their own
frozen criteria).

## Procedure

1. `align_reference_export.py` (`.venv-p0`, reference PYTHONPATH): dumps
   codes / conditioning / waveforms / hashes / RNG snapshots to
   `artifacts/p1/alignment/reference/`.
2. `align_adapter_export.py` (`.venv-omni`, clean env): same manifest via the
   adapter to `artifacts/p1/alignment/adapter/`.
3. `align_compare.py` (CPU): applies A1–A9, emits machine-readable
   `alignment_result.json` (per-case pass/fail + diagnostics).

## RNG findings & evidence-based rule revisions (recorded after run 3655, before rerun 3656)

The first full run surfaced three rule-level issues; revisions are justified
by direct evidence, not by adapter results (adapter cn-voice cases were
already bit-exact when the rules were revised):

1. **Decode consumes global RNG (reference fact).** `HiFTGenerator`'s
   `SineGen2` draws `rand_ini = torch.rand(...)` (hifigan_generator.py:270)
   and additive `randn_like` noise (lines 166/222) at inference. The flow
   ODE uses a fixed construction-time noise buffer, but the vocoder does
   not. Therefore decode outputs differ across call-time seeds and are
   reproducible per seed. Consequences:
   - A3's "seed inertness" was wrong; replaced by "runs clean" +
     per-seed reproducibility via A5.
   - A7's "RNG unchanged after decode" was wrong for BOTH reference and
     adapter; replaced by: encode/voice must not consume RNG (kept) and
     decode consumption is a documented reference fact, with identical
     consumption proven transitively by A5 bit-equality (identical outputs
     under identical pre-call seeds require identical draws).
   - A8 interleaving equality now checked under fixed per-call seeding —
     this is precisely the request-scoped RNG contract P2 must implement
     for the vocoder terminal.
2. **A1 test bug:** `[cn, en] * 64` has `en` as its last item; the rule
   compared it against `single_cn`. Fixed to compare against `single_en`.
3. **Voice codes resample-chain deviation (adapter bug, fixed):** the
   adapter initially encoded voice prompt codes from the 24 kHz-resampled
   waveform; the reference encodes them from the ORIGINAL sample rate
   directly to 16 kHz (its xvector keeps the 24k→16k chain). cn-voice codes
   happened to be insensitive; en-voice codes were not. The adapter now
   follows the reference chain exactly (codes: orig-sr→16k; mel/xvector:
   orig-sr→24k→16k / 24k mel).

No numeric acceptance threshold was relaxed: A5 remains strict blake2b
equality.

## Final result (run 3669, 2026-09-06)

`"pass": true, "failures": []` — 33/33 checks. Highlights:

- A2: per-code equality across path / tuple / tensor forms and batch_size
  {1, 4, 128} incl. mixed lengths (reference batch is per-sample-equivalent,
  so batched encode is safe to expose).
- A4/A5: conditioning tensors and all 20 decode cases (5 code sets × 2
  voices × 2 seeds) are **bit-exact (blake2b equal)** against the reference
  in a different transformers major version (5.12.1 vs 4.57.1).
- A6: empty codes → clear error both sides; out-of-range code → adapter
  rejects (pre-registered stricter-than-reference deviation).
- A7/A8/A9: encode/voice consume no RNG; decode RNG consumption documented
  (HiFT); seeded interleaving == solo; genuine bad-voice injection recovers;
  duplicate cleanup safe; zero sessions after finalize.

Artifacts: `artifacts/p1/alignment/{reference,adapter,alignment_result.json}`.

Low-precision (bf16/fp16) codec: **not run** (optional track, explicitly
non-blocking; V1 default remains FP32 — decision D2 unchanged).
