# P5 qualification protocol v1

Frozen before generation on 2026-09-08; P4 source is 854d5a7. This is a small
adaptation regression benchmark, not a full SeedTTS score or product quality SLA.

Use the first four rows of each en/zh split of the repository's pinned SeedTTS
Arrow revision 27f4c1adee83b5b29b7c4b375f6b976324bda308. No filtering, replacement
or post-generation exclusion. Stage all eight source audios and annotations with
SHA256 before model inference; generate four modes per source (32 requests).
The dataset is external and is not redistributed with the integration.

- T2S: ask the voice assistant to read the target text verbatim, default configured
  voice, no voice cloning. ASR against target text, not the wrapping instruction.
- S2T: explicit verbatim transcription system instruction; source audio against
  its source annotation. The model is a chat model; errors remain in the report.
- S2S: ordinary voice-assistant reply to the same source audio; compare output ASR
  and acoustics to the same-precision reference reply, never to input transcripts.
- T2T: ask for verbatim repetition of target text; compare both the annotation and
  reference output. This is text regression, not general reasoning evaluation.

All use greedy decoding, repetition_penalty=1.1, seed=0, max_new_tokens=512,
TP1, BF16 AR, FP32 codec, prompt<=512/context1024. Preserve final length stops
in metrics. Exact full dual-channel grid, canonical input, text and stop agreement
remain adaptation hard gates under P3 v1/E1. No near-tie exemption or new tolerance.
TF32 matmul is off and cuDNN TF32 remains on, matching the archived P3 environment.
Reference is locked local HF code in .venv-p0, native uses real HTTP with the
formal model stages in .venv-omni. Same reference codes through the P1 decoder
and default voice/seed provide waveform comparison. No injected native outputs.

Run independent OpenAI Whisper small (openai-whisper==20250625, official download
URL plus embedded SHA256) only after generator shutdown; explicit language,
temperature0, beam_size5, fp16=False, condition_on_previous_text=False. Reuse shared benchmark text normalization
and corpus error counting. Report English WER and Chinese CER separately, each
with the actual numerator/denominator. Include every sample, even failed,
empty or truncated ones; missing transcripts are failures, not zero-error rows.
Report waveform finite/nonempty, duration, RMS/peak/clipping, hashes and ASR
agreement; publish sample paths for listening. Human MOS is unassessed.

Quality/performance thresholds are uncalibrated and disabled, represented by
null values with gate_thresholds=False/calibrated=False. Functional tests and
non-finite/empty output checks remain enforced. Real hosted CI calibration needs
independent repeated runs on that actual host; local A800 evidence is labeled.

4090 qualification starts with unchanged P4 YAML and driver. Real startup/OOM
and shutdown evidence may establish unsupported24G. A failed optional hardware
qualification does not enable a misleading YAML or lower precision silently.
Current metadata/license sources are audited without assuming code licenses
cover checkpoints; unresolved rights block weight redistribution/upstream-ready
claims but do not authorize contacting maintainers or publishing model weights.


## Appendix A: reference-path correction v1.1 (2026-09-08)

The original P5 export3871 inadvertently used HF's implicit logits_to_keep=1.
P3's RawLogitsCapture.forward(*args, **kwargs) hides the original forward
signature, so GenerationMixin._supports_logits_to_keep() returns False there:
P3 evaluates the model's default logits_to_keep=0 (full prefill LM heads).
P5's uncaptured call exposed that parameter and HF auto-selected1. This changes
the BF16 head GEMM shape while leaving every hidden activation identical.

3876/3877 show all40 layer outputs and both final norms bit-identical. Three
requests differ only in inactive-audio step0, including one duplicated input;
all32 texts/stop semantics and all16 generated WAVs match. Probe3878 explicitly
sets logits_to_keep=0: both full raw head vectors and selected rows exactly match
native in all three probes, including a previously passing control. Evidence:
artifacts/p5/diagnose_summary.json and artifacts/p5/logits_profile_comparison.json.

Decision: make the existing P3 full-head numerical profile **explicit** with
logits_to_keep=0 in the new P5 reference exporter, then regenerate a versioned
reference and re-run the GPU CI entry. Retain3871 and failed3872 unchanged.
No P0–P4 expected file, tolerance, production forward, or sampling policy changes.
The final quality report must distinguish this configured reference from HF's
implicit optimized default; equivalence to arbitrary HF optimization settings
is not a promised numerical contract. P3's observer should never be treated as
transparent unless its model call parameters are explicitly fixed. This is a
reference configuration correction under Tasks.md's existing policy, not an
argmax tie exemption or a relaxation of the full-grid gate.
