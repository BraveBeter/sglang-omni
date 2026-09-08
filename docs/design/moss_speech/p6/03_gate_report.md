# P6 streaming gate report (2026-09-09)

P6 qualifies opt-in incremental chat streaming on one NVIDIA A800-SXM4-80GB.
The default configuration remains non-streaming. All32 canonical requests and
4001 full dual-channel grid rows match the frozen P5 HF reference. All16 PCM
outputs exactly match the independent streaming-codec reference. The original
offline regression also retains all16 P5 WAVs byte for byte. P7 has not started.

| Gate | Result and accepted evidence |
|---|---|
| G1 codec | Pass: independent reference3890/native3891, both trained chunk5/25 profiles,32 boundary cases with exact per-chunk waveforms; two-seed interleaving, cancellation recovery, RNG restoration and empty sessions |
| G2 incrementality | Pass: full HTTP CI3903,4001 exact grid rows; all16 audio cases have client first PCM before AR end; UTF-8 text and exact sample/code ledgers |
| G3 lifecycle | Pass:3903 completes53 normal requests plus six preflight failures, three disconnects and one controlled post-header error; all22 lifecycle checks, four actual child-process exits; offline3900 passes32 |
| G4 engineering/evidence | Pass: CPU292/fresh114, scoped package declaration3, shared vocoder27;25 Python static checks; actual Slurm CI and ASR3904; executable configs, provenance and source hashes archived |

These suites overlap and are not additive. Hosted GPU/CPU CI execution is not
claimed; the repository CI entry points were executed locally and in Slurm.
The independent CPU environment has no SGLang, CUDA-enabled torch or model assets.
All final GPU jobs completed with Slurm exit0:0, and no owned Slurm job remains.

## Delivered behavior

The registry exposes `MossSpeechStreamingPipelineConfig` through the explicit
`streaming` variant. AR uses its existing sampling/FSM and emits each newly
produced grid row once. Output routing selects only the request's text or audio
terminal. Text buffering withholds incomplete UTF-8; audio uses real chunk-trained
flow and request-owned HiFT overlap/RNG state through `StreamingVocoderBase`.

Speech deltas are PCM16 little-endian mono24kHz or independent WAV containers.
WAV headers must not be concatenated. Normal completion has one finish/usage
and one `[DONE]`. Disconnects release unfinished backend/terminal work. An error
after headers yields sanitized SSE error data, without a success finish or
`[DONE]`. This model-neutral shared fix is independent commit **da6ce46**.
Model implementation, tests and reproduction drivers are commit **e343614**.

The codec profile is **chunk-cudnn-deterministic-v1**: FP32 flow/HiFT, scoped
cuDNN deterministic=True and benchmark=False with caller flags restored. The
streaming vocoder worker alone sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Default offline and AR worker
environments retain their existing behavior. There are no changes to frozen
P3/P5 goldens, AR arithmetic, tolerances or full-grid E1 acceptance.

Only `supports_streaming_vocoder` becomes true. The package declaration changed
after GPU qualification, with before/after SHA and three CPU checks archived in
sglang-omni/docs/design/moss_speech/p6/source_amendments.json. Its serving consumer
is capability logging. The exact GPU-executed source hashes remain unchanged in
the raw report; final inference/config/HTTP/lifecycle code matches those hashes.
The postprocessor requires an explicit declaration receipt for this one change;
without that receipt it rejects the old-run/current-source mismatch.

## Measured performance and memory

The sample is the unchanged P5 bilingual set:8 source rows,4 unique source audio
recordings,32 four-mode requests and24 unique request bodies. Audio statistics
below use the16 canonical audio-output requests, including two length stops.
TTFA starts at HTTP initiation and ends at the first nonempty PCM delta received
by the client. RTF is HTTP end-to-end duration divided by delivered audio duration.

| Metric | p50 | p95 | Maximum |
|---|---:|---:|---:|
| Client TTFA (s) | 2.563 | 2.719 | 3.080 |
| End-to-end RTF | 3.868 | 5.639 | 5.640 |
| Absolute PCM jump at codec seams, normalized full scale | 0.006958 | 0.064209 | 0.301880 |

All16 client first-PCM receipts precede AR completion: the conservative minimum
margin is1.415s. The bound uses backend-input-to-AR-end elapsed time minus client
TTFA; backend input happens after HTTP initiation, so a positive bound proves
early receipt without equating clocks. The raw event and client timings are
retained. Header, role and empty events do not count as audio.

The complete3903 pipeline sampled **26.487GiB NVML peak**. Four repeated waves
with two simultaneous requests each ended at **27122.3125MiB**, with0MiB variation.
All survivors, recovery requests and repeats retain their serial grid/PCM.
Pending-flush disconnects additionally require actual terminal release events,
not an AR-aborted event after AR has already succeeded.

Isolated codec probe3902 separately runs two512-code requests. Both match every
independent reference chunk, and both return to the warmed live baseline exactly
(588.159MiB). Its reserved peak is3.473GiB, live peak3.089GiB, with zero allocator
retries/OOMs and actual expandable-segment snapshots. These are codec-only figures,
not total serving memory. Default allocator probe3898 reserved77.049GiB despite
bounded live tensors; that result is retained and is not the supported preset.

RTF exceeds1: this FP32 prefix-recomputation implementation does not sustain
real-time playback. Seam amplitudes are descriptive, not a calibrated listening
quality threshold. There is no human MOS or latency/quality SLA.

## Quality and offline comparison

ASR3904 uses the unchanged P5 evaluator, official Whisper small checkpoint,
English WER/Chinese CER normalization and truncation policy. All reference/native
transcripts agree. Aggregate scores are also unchanged from offline P5:

| Output task | English WER | Chinese CER |
|---|---:|---:|
| Text to text | 0/39 (0%) | 3/80 (3.75%) |
| Text to speech, ASR transcript | 2/39 (5.13%) | 31/80 (38.75%) |
| Speech to text | 10/76 (13.16%) | 0/50 (0%) |
| Speech to speech | No ground-truth answer transcript | No ground-truth answer transcript |

For S2S, reference/native ASR agreement is measured; no correctness score against
an invented answer is assigned. Two Chinese S2S cases still reach the512-row
budget, share the same source input, and remain counted. `length` explicitly
indicates possible utterance truncation. Tail-flush correctness does not promise
that a budget-limited utterance is linguistically complete.

All16 streaming/offline pairs have equal sample counts and different waveforms,
as expected for separately trained codec profiles. The median aligned waveform
RMSE is0.099748 after PCM16/32768 normalization; this is a descriptive signal
comparison, not a perceptual quality claim. Same-profile reference/native PCM
remains exact. No source audio, transcript, weight or large tensor is committed.
Compact per-case measurements are in
sglang-omni/docs/design/moss_speech/p6/quality_comparison.json.

## Verified support boundary

- A80080GB, TP1, BF16 AR/KV, FP32 codec, eager torch_native attention.
- HTTP streaming chunk5; chunk25 has component evidence only.
- Prompt512/output512/context1024, KV4096, inherited AR running4/waiting20.
  Serial and two-request streaming were measured; a four-stream capacity claim
  is not established. Codec calls remain serial with request-local state.
- P5's real4090 failures still exclude24GB support. No new24GB YAML, low precision,
  CPU offload, TP>1, radix, CUDA Graph, compile or quantization is qualified.
- Uploaded voice/reference-audio service features remain disabled. Speech input
  through chat remains supported; these are different capabilities.
- Weight/HF-code redistribution rights remain unresolved. This does not block
  the authorized fork integration-code delivery, but upstream/weight release
  readiness remains false.

## Reproduction and evidence integrity

All paths below are relative to the workspace root containing sglang-omni/.
Environment, checkpoint revisions and executable commands are in
sglang-omni/docs/cookbook/moss_speech.md and
sglang-omni/docs/design/moss_speech/p6/02_reproduction.md.

| Evidence | Local artifact |
|---|---|
| Independent chunk reference and native comparison | artifacts/p6/codec_reference_3890/report.json; artifacts/p6/codec_compare_3891/report.json |
| Full independent audio reference | artifacts/p6/audio_reference_3894/report.json |
| Final native CI, SSE/PCM/grid/events, PID exits | artifacts/p6/ci_3903/report.json |
| Original offline CI regression | artifacts/p6/o_3900/report.json |
| Independent long-codec allocator qualification | artifacts/p6/memory_3902/report.json |
| Same-profile16 PCM/voice comparisons and32-case ASR | artifacts/p6/quality_input_3904/comparison.json; artifacts/p6/evaluate_3904/report.json |
| Full/fresh CPU, declaration, static and Slurm receipts | artifacts/p6/cpu_full_final.log; artifacts/p6/cpu_fresh_final.log; artifacts/p6/cpu_capability_after.log; artifacts/p6/static_delivery.log; artifacts/p6/slurm_accepted.txt |

sglang-omni/docs/design/moss_speech/p6/gate_summary.json records all hard gates,
measurements, raw-report/PCM hashes and GPU runtime source hashes.
sglang-omni/docs/design/moss_speech/p6/evidence_index.json adds CPU/Slurm/config,
reference-profile and ASR-checkpoint hashes. The delivery manifest indexes final
source and documentation files separately from the GPU-executed source.
Raw artifacts stay in the local workspace; Git carries code and compact evidence.
Root Tasks.md, docs/plan.md, CHANGE.md and continue.md are local workspace records
and are not part of the sglang-omni Git repository.

## Retained failures and corrections

- 3885: original HF streaming helpers reproduce mel-axis and13-code sample-count
  bugs. The valid independent reference uses unchanged locked CosyVoice2 token2wav.
- 3887: strict chunk-weight load exposed epoch/step training metadata; only those
  two keys are excluded, with strict inference-key/shape loading retained.
- 3888:7/32 same-profile waveforms diverged under default cuDNN; independent3889
  reproduces the same drift within one implementation. Separately versioned
  deterministic profile3890/3891 achieves exact equality; old failures remain.
- 3893/3896: canonical/normal requests passed, but the overall reports remain
  failed because a pending-terminal abort check incorrectly required already
  successful AR to become aborted. Real terminal-release events and a new full
  run3903 close that gap; no failed report was edited into a passing result.
- 3895: a112-byte generated Unix socket path exceeded107 bytes before launch;
  a shorter output prefix fixes the wrapper. Original offline3900 then passes.
- 3898: cold-vs-warmed baseline made the first leak probe invalid; both requests
  returned to the same live memory, but allocator reservation grew excessively.
  3901 used an ineffective allocator alias and was cancelled with partial data
  retained. Correct CUDA environment3902 and full CI3903 qualify the final preset.
- 3899: an ASR job dependent on failed3893 was cancelled before execution. Accepted
  ASR3904 depended on final3903 and ran only after serving released the GPU.

These corrections change neither the frozen AR reference nor acceptance
thresholds. Their detailed protocol and diagnostic reasoning remain in
sglang-omni/docs/design/moss_speech/p6/01_streaming_contract.md.
