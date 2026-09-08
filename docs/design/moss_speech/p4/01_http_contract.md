# P4 HTTP contract and framework-gap RFC

P3 is published at dc2c517. P4 starts with the approved Tasks.md T4.1–T4.6.

## Verified framework gaps and minimal extensions

The public Client separates messages, sampling parameters and metadata in
OmniRequest. P3 direct-coordinator tests carried a complete GenerateRequest in
inputs. MOSS reconstructs the public form inside its request_builders; routing
also reads metadata.output_modalities. No coordinator/scheduler changes.

MOSS terminals retain the diagnostic state and additionally expose the common
Client text/audio_data/sample_rate/usage fields. Usage counts grid positions,
not the sum of both token channels. stop and length come from native AR.

Non-streaming chat had no disconnect watcher. The common HTTP adapter now
races completion against disconnect, aborts on disconnect/task cancellation,
and cancels the losing waiter. A default-no-op PipelineConfig preflight hook
runs before response headers, including streaming headers; MOSS overrides it
using its existing CPU contract. Invalid/unsupported requests raise ValueError
and map to HTTP 400. P3 GPU code and shared scheduling remain unchanged.

The hook is synchronous CPU validation; audio is decoded on CPU once here and
again in preprocessing for its process-local stash. No codec model runs here.
Avoiding that duplication is a later measured optimization.

## Acceptance

Real loopback TCP /v1/chat/completions + public Client + formal YAML/native AR;
four fixed P0 inputs, greedy dual-grid match, text and PCM WAV match reference
codes through P1 codec with the same voice and seed. Check 24kHz finite nonempty
audio, IDs, one choice, stop/length and grid-count usage. Record actual child
PIDs and exits. Inject no grids or AR outputs into the real requests.

Rejected: streaming, combined output modalities, custom voice/audio config,
nonzero min_p, custom stop, malformed audio, invalid parameters/empty messages.
This phase exposes chat audio input/output, not uploaded voice conditioning:
all optional ModelCapabilities flags remain False, as do realtime/translation.

Lifecycle: wait for actual AR prefill event before closing the TCP request;
assert abort, survivor and recovery; same-seed random requests reproduce under
concurrent arrival, changed seed changes output. Native batch/KV isolation
continues to be guarded by P3 tests.

Load: P1 24-request input manifest (12 text + 6 short speech + 6 long speech),
0.4s arrivals, 3 warmup + 3 measured rounds, with actual native generation of
up to 200 grid rows replacing P1 synthetic decode codes/background AR. The
results are not a like-for-like P1 speedup. Keep Layout A, codec execution 1,
internal encode batch 4; qualify native limits with measured evidence.

Driver: `sglang-omni/scripts/moss_speech/p4/validate_http.py`.
Slurm launcher: `scripts/p4_http.sbatch` (add --workload for load qualification).
CPU regression: `sglang-omni/tests/unit_test/moss_speech/test_http_contract.py`.


## Deployment revision from actual P4 evidence

Run 3862 accepted a roughly 6K-token prompt under the old 10K ceiling and OOMed
with a large automatically allocated KV pool. Its report remains a failure.
Run 3863 passed all HTTP/lifecycle gates with prompt <=512, requested output
<=512, context <=1024 and native running=4/queued=20. Four boundary prompts each
had 512 rows and generated 476 rows (stop), not 512 actual generated rows.

Final tuning caps KV at 4096 positions (4×1024); HTTP3864 and native load3865
both passed with this allocation; waveform soak3867 and CLI3866 also passed. Higher context is a separate expansion gate before
claiming it in a deployment, and does not require changing frozen P3 parity.
The coordinator's existing admission mechanism is reused. Chat now maps its
QueueFullError to 503, preserving 400 for stringified model validation errors
and 500 for actual internal errors.

The broad HTTP regression has a known environmental M4A failure: torchcodec's
shared library references a missing torch CUDA ABI symbol. It reproduces when
the exact unchanged dc2c517 HTTP module is loaded. MOSS WAV tests do not use that
path; retain the failure and repair the dependency separately before general
M4A endpoint qualification. Do not call that broad suite fully passing.
