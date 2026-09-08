# P6 streaming contract and reference protocol

Status: P6 qualification passed on 2026-09-09; P5 source baseline515a734.
The original offline pipeline remains the default. The separately selected
chunk5 streaming variant passed all functional gates; measured scope and retained
failures are recorded in sglang-omni/docs/design/moss_speech/p6/03_gate_report.md.

## Codec profile

Use the locked codec revision eeec733e4e1dea7da444d332d8e1621ef257414c, actual
flow-chunk-5.pt or flow-chunk-25.pt with matching encoder static_chunk_size and
flow estimator static_chunk_size times4. HiFT weights remain flow/hift.pt.
The reference uses the unmodified locked CosyVoice2Model.token2wav implementation
with the corresponding HF-constructed flow/HiFT modules. It constructs no LLM.
Native code reuses the existing explicit flow/HiFT constructors and only adds
session ownership, bounded cursors and RNG isolation. Loading is strict after
excluding exactly the checkpoint training counters epoch and step.

The original HF streaming_inference helper fails on mel axes, and its
stream_inference helper produces24480 instead of24960 samples for13 codes.
These helpers are not used as a valid streaming reference. Job3885 preserves
both findings. Jobs3885/3886 independently exercise short, threshold-adjacent,
first/steady/final lengths with seeds0 and1 for both real chunk profiles.

First hop = chunk + (-prompt_code_length) modulo chunk. Subsequent hops equal
chunk. Nonfinal flow input includes the available cumulative code prefix plus
3 lookahead codes; only newly committed mel after cursor*4 is consumed by HiFT.
Final flow recomputation follows upstream finalize=True/streaming=False.
HiFT preserves8 mel frames /3840 source and speech samples. Its Hamming overlap
uses the upstream float64 window arithmetic with FP32 stores. Withheld samples
are emitted once on subsequent/final calls. Total PCM length is codes*1920 at
24kHz. Long history is bounded by the existing512-code generation limit.

Every request owns its code cursor, voice, CPU/device RNG states and HiFT cache.
Codec calls are serial; interleaved requests restore their RNG stream before
each call and restore the caller's RNG afterward. Completion, errors and abort
release sessions idempotently. Late chunks cannot recreate an aborted terminal
scheduler state. Default voice data is not released by request cleanup.

### Explicit deterministic profile correction (2026-09-08)

Original strict comparison3888 failed7/32 cases (max absolute waveform drift
0.0001465287), despite correct lengths, RNG restoration and empty caches.
Independent same-implementation probe3889 repeats the same seed/input six times:
default cuDNN also drifts0.0001465287; cuDNN deterministic gives exact equality
in all six, as does full deterministic algorithms. This demonstrates an unstable
reference execution setting, not permission to relax the adaptation gate.

Freeze **chunk-cudnn-deterministic-v1** before producing the corrected reference:
FP32 codec, cudnn.deterministic=True, cudnn.benchmark=False, existing cuDNN TF32
setting retained, matmul TF32 off. Native applies the cuDNN flags in a scope
around the codec call and restores them afterward. No AR settings, P3/P5 expected
asset, offline model arithmetic, quality tolerance or acceptance threshold is
changed. Old3886/3888 are retained; new reference3890 is a separate artifact.
Full deterministic algorithms and a new global CUBLAS setting are unnecessary
and are not enabled in production. Same-profile per-chunk waveform equality,
length accounting and interleaved seed equality remain hard gates.

## Pipeline and wire contract

MossSpeechStreamingPipelineConfig is an explicit registry variant. Default
MossSpeechPipelineConfig continues to reject stream=True. Both preserve the
same bounded native AR, grid semantics and output-modality terminal routing.
The streaming variant uses stream_to for text_decode and audio_vocoder, with
stream_done_to_fn selecting only the active terminal for each request; offline
requests send no stream-done signal. Both terminals accept chunks before their
terminal payload. Complete output grids remain available for verification.

The AR builder reads each new output row once and never samples itself. Text
requests send text-token tensors. Audio requests send only valid codes in the
first real audio segment; inactive-text-row audio EOSP cannot end that segment.
First audio metadata includes voice tensors as wire-safe lists, voice key and
seed. Subsequent metadata carries a contiguous row_index. The vocoder inherits
StreamingVocoderBase and does not replay a full waveform in its terminal result.

Text terminal decodes cumulative tokens, holds incomplete trailing UTF-8 and
requires emitted text to remain a prefix. Final valid Unicode text must match
the offline decode. If max_tokens cuts a byte sequence, trailing replacement
characters are withheld; the complete Unicode prefix is emitted, while the
original grid/token usage and length finish reason remain intact. This explicit
length-stop behavior avoids sending broken UTF-8 or hiding consumed tokens.

Chat audio delta defaults to independent WAV containers; explicit
stream=True/audio={format:pcm} selects headerless PCM16 little-endian at24kHz
mono. Concatenate decoded PCM, not WAV headers. Uploaded voices and other audio
options remain unsupported. Actual wire/error/finish/usage behavior and early
PCM relative to AR end must be verified over TCP before delivery.

## Gates and evidence

Tasks.md T6.1–T6.6 is the execution plan. Required evidence includes independent
codec comparison, exact AR grids on P5 cases, four-mode SSE, abort before/after
first chunk and during flush, short/length tails, interleaved recovery, repeated
memory/state cleanup, actual TTFA and E2E RTF, and the original offline regression.
No quality SLA, MOS,24GB support, CUDA Graph/radix/compile, TP>1 or hosted CI
threshold is inferred from successful streaming transport. Changes are recorded
immediately in CHANGE.md and raw experiments remain in artifacts/p6/.

### Shared SSE failure gap

The existing chat generator propagated post-header model errors without an SSE
error body. CPU reproduction captures a real first delta followed by an upstream
exception. A model-neutral wrapper now emits a sanitized error object and ends
the stream, with neither a success finish chunk nor [DONE]. This preserves the
existing failure-without-DONE contract while making the error visible to clients.
Cancellation remains a BaseException and propagates; the owned iterator closes.
The existing generic failure test now asserts the explicit error instead of a
Python RuntimeError, and retains its no-DONE check. This small shared patch is
committed independently as da6ce46, separate from MOSS-specific stream machinery.

## Terminal release evidence and allocator qualification

The initial full HTTP run3893 completed all32 canonical requests (4001 exact
grid rows) and21 additional normal requests. All16 PCM outputs equal the
independent full-audio reference3894. Its three disconnects reached coordinator
abort and surviving/recovery audio remained identical. However, the acceptance
script incorrectly required an AR `model_path_end/status=aborted` event even when
AR had already ended successfully and the audio terminal was still pending.
That one lifecycle check failed in both3893 and the CI-entry smoke3896; the
original reports remain failed and are not silently rewritten.

The model terminal now emits `moss_codec_session_released` after releasing its
codec ownership and clearing its session reference. The event records whether
that request still owns a session, active session count and, on GPU, live and
reserved allocator bytes. It uses the existing optional event recorder and
adds no HTTP response fields. The corrected lifecycle check requires this
terminal receipt after audio has started; before codec creation, an actual AR
abort is sufficient. A new process/run must validate the corrected check.

Both original streaming HTTP runs sampled a79.459GiB NVML peak, despite bounded
request state. Probe3898 distinguishes live tensors (about0.58GiB between calls,
about3.1GiB peak) from accumulated allocator reservation (about77GiB at500 codes).
Both long requests release their sessions and return to the same live footprint.
The initial cold-model comparison is invalid for detecting leaks because first
inference initializes persistent model state; it is retained as a failed probe.
The follow-up explicitly warms up before the unchanged1MiB release threshold.

An allocator experiment tests `expandable_segments:True` without changing the
flow/HiFT arithmetic, dtype, seed or chunk protocol. The installed Torch2.9 CUDA
allocator reads `PYTORCH_CUDA_ALLOC_CONF`; the generic alias in cancelled3901
was ineffective. Snapshot `is_expandable` evidence is required, in addition to
two512-code runs matching the independent reference per chunk, no allocator
retry/OOM, <=8GiB reserved and return to the warmed live baseline within1MiB.
Only after this probe passes may the streaming vocoder's stage-specific env
adopt the setting. AR, preprocessing and default offline remain unaffected.
The new complete HTTP/CI run must measure its actual memory and lifecycle; old
NVML results cannot be relabeled as measurements of the new allocator setting.

Allocator follow-up3902 passed: both512-code runs match every independent
reference chunk, snapshot confirms expandable segments, live bytes after each
release equal the warmed baseline exactly, reserved peak is3.473GiB and there
are no allocator retries or OOMs. The streaming variant now sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only in the audio_vocoder
stage's worker environment. CPU tests prove the other stages and original
pipeline env remain unchanged. Full CI3903 validates the new deployment.
Environment overrides are outside this measured allocator preset.


Full CI3903 passed all32 canonical and21 additional normal requests, all22
lifecycle checks, three disconnects and four actual child-process exits. Client
first PCM preceded AR end in every16 audio case. Independent ASR3904 passed;
offline CI3900 retained all16 P5 WAVs exactly. New full-pipeline NVML peak was
26.487GiB; four repeated two-request waves had identical idle memory. The old
3893/3896 failed reports remain failed. Capability declaration was enabled only
after these gates; its exact before/after SHA and CPU checks are recorded in
sglang-omni/docs/design/moss_speech/p6/source_amendments.json.
