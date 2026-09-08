# P5 quality report (2026-09-08)

Final evidence: HF reference job3880, native HTTP GPU CI job3883, independent
Whisper small scoring job3884. All32 requests completed; all4001 dual-channel
grid rows, canonical inputs, generated text and finish reasons match exactly.
All16 WAV pairs are byte-identical using reference codes through the qualified
P1 decoder and the same configured voice/seed. This establishes adaptation
agreement for this sample and numerical profile, not production quality.

## Scope and numerical profile

The frozen selection contains8 rows (first4 per language),4 unique source audio
recordings,32 mode requests and24 unique request bodies. Repeated source audio
is retained; these are not32 independent utterances. Greedy seed0, repetition
penalty1.1, max512 generated grid rows; BF16 AR/KV, FP32 codec, A80080GB, TP1,
eager torch_native attention. Matmul TF32 is off, cuDNN TF32 is on.

Reference explicitly sets **logits_to_keep=0**, preserving the P3 full-prefill
LM-head projection profile. Initial3871 used the HF implicit last-only1 path;
3872 correctly failed the full-grid gate on3 inactive-audio step0 rows. All40
hidden layers and final norms were identical; the LM-head GEMM shape caused the
difference. Explicit0 probes3878 restored raw-vector equality. Initial artifacts
are retained and protocol AppendixA records the correction. No frozen P0–P4
expected asset, production forward, sampling policy or tolerance was changed.
Equivalence to HF arbitrary optimization defaults is not claimed.

## Measured quality

English uses shared Whisper text normalization plus WER; Chinese uses shared
normalization plus character error rate. Values below compare to the dataset
annotation and are identical for reference and native. Each cell is a corpus
error numerator divided by reference units, including every selected sample.

| Task | Language | Cases | Reference | Native |
|---|---|---:|---:|---:|
| T2T | en (WER) | 4 | 0/39 = 0.00% | 0/39 = 0.00% |
| T2T | zh (CER) | 4 | 3/80 = 3.75% | 3/80 = 3.75% |
| T2S | en (WER) | 4 | 2/39 = 5.13% | 2/39 = 5.13% |
| T2S | zh (CER) | 4 | 31/80 = 38.75% | 31/80 = 38.75% |
| S2T | en (WER) | 4 | 10/76 = 13.16% | 10/76 = 13.16% |
| S2T | zh (CER) | 4 | 0/50 = 0.00% | 0/50 = 0.00% |

T2S scores use independent ASR of the generated waveform against the requested
target text. The38.75% Chinese T2S CER is high; it combines model reading and ASR
errors in this tiny sample. No listening audit has separated those causes.
S2T scores compare the requested verbatim transcription with source annotations.
T2T compares prompted repetition with target annotations. The model remains a
conversational model, without a dedicated ASR/TTS accuracy guarantee.

S2S is an ordinary spoken answer, so input transcripts are not answer goldens.
All8 reference/native S2S ASR comparisons have zero edits. Both implementations
hit the512-row limit for zh_00_s2s and zh_01_s2s (the same source recording):
2/8 S2S,2/4 Chinese S2S,2/32 total requests. These truncated outputs remain in
all statistics and audio artifacts; they are not successful complete utterances.

Whisper small: openai-whisper20250625, explicit language, temperature0, beam5,
fp16=False, condition_on_previous_text=False. Checkpoint SHA256:
`9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794`.
The evaluator ran only after generator shutdown. Exact (WAV SHA, language)
duplicates share ASR results:12 unique pairs across32 reference/native files.
This avoids redundant scoring without treating duplicated audio as independent.

All16 native WAVs and16 reference WAVs are finite, nonempty, mono24kHz.
Duration p50/max=15.84/40.96s; maximum measured clipping fraction is0.
Per-file duration/RMS/peak/clipping and transcripts remain in the raw evaluation
report. Human listening/MOS is **unassessed**. Local listening samples are
`artifacts/p5/ci_3883/en_00_t2s.wav`, `artifacts/p5/ci_3883/zh_00_t2s.wav`,
`artifacts/p5/ci_3883/en_00_s2s.wav` and `artifacts/p5/ci_3883/zh_00_s2s.wav`
(last sample is length-truncated). Raw audio/text is not redistributed here.

## Local timing and memory

Serial client concurrency1,8 cases per mode, real HTTP E2E; no separate warmup
exclusion. These small-sample percentiles include startup-cold request effects.
RTF is HTTP latency divided by generated audio duration; it is not streaming
latency. HF timing covers encode+AR only and is not used as an E2E speedup.

| Mode | HTTP p50 / p95 (s) | RTF p50 / p95 |
|---|---:|---:|
| T2T | 1.096 / 1.476 | N/A |
| T2S | 5.132 / 6.622 | 1.236 / 1.666 |
| S2T | 1.133 / 1.897 | N/A |
| S2S | 28.287 / 34.097 | 0.847 / 0.883 |

Native serving NVML sampled peak=27.645GiB at0.5s intervals,
not an instantaneous allocator peak. P4 remains the load/concurrency evidence
(144 requests and the separately measured boundary peak29.188GiB). P5 does not
replace that capacity report with serial microbenchmark timing.

Quality/performance gate_thresholds=False and calibrated=False. No quality SLA
or hosted CI threshold was invented. Functional full-grid/coverage/finite-wave
gates pass; Chinese reading accuracy and output truncation remain visible
limitations for consumers and inputs to later, separately scoped evaluation.

Evidence paths and SHA256 are in
`sglang-omni/docs/design/moss_speech/p5/gate_summary.json`.
Protocol: `sglang-omni/docs/design/moss_speech/p5/01_protocol.md`.
Hardware: `sglang-omni/docs/design/moss_speech/p5/03_hardware.md`.
