# MOSS-Speech runtime options

MOSS-Speech supports chat, speech synthesis, and transcription through the
existing `sgl-omni serve` application. The default execution profile remains
single-GPU BF16 AR/KV with FP32 codec, eager `torch_native` attention, and no
prefix cache. The tested device is an A800 80 GB. Quantization, tensor
parallelism, a single CUDA Graph spanning the entire pipeline, and
`torch.compile` remain unsupported;
this configuration does not support a 24 GB GPU. Qualification used
Torch 2.9.1+cu128, SGLang 0.5.16, and Transformers 5.12.1.

## Optional execution settings

The following model options use the values shown through `sgl-omni serve`,
or their underscore names as top-level YAML fields:

| CLI option | Behavior |
| --- | --- |
| `--decode-projection-graphs true` | Capture the norm, projection, RoPE, and MLP sections of each decode layer for batch sizes 1–4. Attention, KV updates, sampling, and request state remain eager. Prefill and other batch sizes use eager execution. |
| `--eager-cpu-metadata true` | Prepare the attention loop's indexing metadata on the CPU once per batch. Model activations and KV tensors remain on the GPU. |
| `--mixed-modalities true` | Allow text-output and audio-output requests in the same AR batch. Each request retains its own sampling, seed, and stopping state. |
| `--prefix-cache true` | Reuse KV through the existing radix cache with keys containing both text and audio tokens. Reuse requires equal original prompt lengths to preserve BF16 kernel shapes. |
| `--codec-optimizations true` | Reuse per-solve attention preparation and fixed codec inputs, omit masked future-key work on qualified A800 streaming shapes, reduce sample transfers, and share HiFT weights. It does not enable streaming request fairness. |
| `--stream-decode-fairness true` | Rotate ready streaming requests after each codec hop. This independent option defaults to false; it can reduce later requests' first-audio wait while increasing completion latency and playback stalls. |
| `--small-first-chunk true` | Use an earlier aligned first hop with chunk25, retaining subsequent 25-code hops. This is a separate, default-off chunk configuration with its own waveform oracle. |
| `--codec-prefix-cache-manifest PATH` | Prepare a bounded fixed-voice CFM prefix bank before readiness. This default-off advanced option requires chunk25 and codec optimizations; it is separate from the AR radix `--prefix-cache` option. |
| `--codec-full-graphs true` | Capture the complete ten-step CFM solve for qualified fixed-voice windows selected by the streaming schedule. Default off; requires chunk25, codec optimizations and a valid prepared manifest. |
| `--decode-full-graphs true` | Prepare complete40-layer B1 AR decode graphs before readiness; unsupported batches, lengths and contexts use the original forward. |
| `--prefill-full-graphs true` | Prepare B1 AR prefill graphs for exact input lengths1–128 and write back to the actual request KV slots; other inputs use the original prefill. |
| `--stream-startup-schedule 6_8_align25` | Use an independent chunk25 schedule with first hops6/8/19 and subsequent25-code hops. Requires chunk25 and excludes `small_first_chunk`. |
| `--codec-final-bias-alignment true` | Prepare the original final attention bias with its qualified storage alignment; requires optimized chunk25. |
| `--codec-final-bias-elision true` | Omit only the qualified all-zero final attention bias while preserving the original selected operator. Requires final-bias alignment. |
| `--codec-qkv-multistream true` | Run qualified tail Q/K/V projections on capture-owned streams inside complete CFM graphs. Requires optimized chunk25, full CFM graphs and their prepared manifest. |
| `--codec-memory-optimizations true` | Reduce encoder attention temporary storage during inference. This is independent of codec preparation and can trade latency for lower peak memory. |
| `--sampling-optimizations true` | Maintain request-local device histories and combine sampling result transfers. This is experimental; gains depend on the sampling settings and workload. |

To enable codec preparation together with the three AR optimizations used by
the optimized evaluation profile, with checkpoints and the voice already staged:

```bash
sgl-omni serve \
  --model-path /path/to/MOSS-Speech \
  --codec-path /path/to/MOSS-Speech-Codec \
  --voice-wav /path/to/voice.wav \
  --codec-optimizations true \
  --decode-projection-graphs true \
  --eager-cpu-metadata true \
  --mixed-modalities true \
  --prefix-cache false \
  --host 127.0.0.1 --port 8000
```

A bare `sgl-omni serve --model-path ...` keeps these options off. The shipped
`examples/configs/moss_speech.yaml` and `moss_speech_streaming.yaml` enable only
`codec_optimizations: true`; stream decode fairness, codec memory, sampling,
the three AR options, and prefix caching remain off. Pass `--codec-optimizations false` to disable codec
preparation explicitly. Enabling the YAML codec setting alone does not reproduce
the optimized evaluation settings: use all three AR flags in the command above
as well, while keeping prefix caching off. Streaming additionally requires the
streaming YAML and the matching chunk profile; RTF depends on that profile and
the request workload.

These options preserve the existing request limits: 512 input grid rows,
512 generated grid rows, context length 1024, and the configured KV pool
capacity. Prefix caching can reduce redundant KV work,
but preserves full-prompt projection and reduction shapes; short prompts
can become slower. Measure it with the intended workload before enabling it.

Partial projection graphs allocate static buffers and incur capture latency
when a batch shape is first encountered. Warm up the batch sizes used in
production and account for this memory. The framework's `cuda graph: False`
log and `supports_cuda_graph=False` describe full-model capture; they do not
indicate whether these model-local projection graphs are active. Do not
combine this option with the framework's full-model graph switches.

Mixed batches increase peak prefill memory relative to processing each
modality group separately. They share the existing scheduler admission limit;
enabling them does not increase the configured maximum running requests.

Codec optimizations retain FP32, the original diffusion step count, and the
selected profile's flow weights and chunk boundaries. The explicit small-first
option below defines a separate chunk configuration. Compare waveforms and
latency within the same configuration, seed, and input/output budgets; these
options alone do not establish real-time streaming.

Streaming request fairness is controlled separately by
`--stream-decode-fairness true` (YAML: `stream_decode_fairness: true`). It defaults
to false, including when codec optimizations are enabled. Offline execution
accepts this option without changing whole-utterance decoding.

When enabled, the streaming vocoder ingests a bounded number of queued messages
and rotates ready requests after each codec hop. It preserves the original audio
emission groups, including inputs containing multiple codes, and sends the
terminal result only after the final waveform is complete. The current hop
remains non-preemptible. Later requests can receive first audio sooner, while
requests already being served can finish later and experience more playback
stalls. This changes service order without reducing full-prefix CFM computation;
measure first audio, completion latency, and playback continuity together before
enabling it for a workload. Pass `--stream-decode-fairness false` to override a
saved configuration that enables it.

## Advanced fixed-voice startup preparation

`small_first_chunk` defaults to false and requires chunk25. For the tested
342-code reference voice, enabling it changes the first hop from 33 to 8 codes;
three lookahead codes are still required. The first emission contains 480 ms of
audio, and subsequent hops remain 25 codes. This changes the chunk configuration
and can change PCM. It retains an independent eager reference; the original
configuration and its exact-output checks remain available.

`codec_prefix_cache_manifest` defaults to unset. An explicit manifest prepares
only N1500 for the original configuration, or separate N1400/N1500 banks with
small-first enabled. It reuses fixed conditioning's attention-prefix outputs
while retaining full QKV projections, ten flow steps, FP32 and HiFT/RNG behavior.
Validation covers A800-SXM4-80GB, SM80, Torch 2.9.1+cu128/CUDA 12.8 and Diffusers
0.37.0, with the exact source and flow/HiFT checkpoints bound in the manifest.
There is no cross-length history reuse or request-time cache construction.

The existing `components.cfm_prefix_cache.build_prepared_manifest` API returns
the JSON manifest. Its keyword arguments are:

| Argument | Required input |
| --- | --- |
| `voice_path` | A CPU tensor artifact containing the exact `prompt_token`, `prompt_feat` and `embedding` conditioning for the deployed reference WAV and voice encoder. Current qualification requires 342 prompt codes; do not fabricate, pad or truncate conditioning to fit. |
| `wave_path` | The corresponding reference WAV. |
| `codec_root` | The local codec checkpoint directory. |
| `source_root` | The MOSS model source directory actually used by the service. |
| `environment` | The same device's `runtime_environment(cuda_device)` result. |
| `warmup_audio_codes` | 36 valid real audio-codebook codes for startup calibration. |

Write the returned JSON from a separate preparation process and let that process
exit before starting the service. Manifest validation checks conditioning/file
hashes, model source, checkpoints and environment; invalid inputs fail startup.
Changing these assets requires a matching new manifest and preparation.

Before readiness, preparation performs native, build and cached codec passes on
those 36 codes, including first-use HiFT work, and checks waveform, sample ledger
and request RNG equality. The original configuration used 2.734 GiB of cache and
3.39 s of preparation in the tested isolated codec; small-first used 5.469 GiB
and 4.97 s. Ready/peak startup allocated GPU memory was 3.32/3.83 GiB and
6.13/6.64 GiB respectively. These figures exclude model loading, other service
stages and the peak of a full 512-code request. Budget their memory separately.

Different request conditioning, other lengths, final noncausal decoding and
changed backend contexts use the ordinary path. Changes to weights, fixed CFM
noise or module training state invalidate the prepared bank; closing the codec
releases it. The two `MOSS prefix cache ready/closed` JSON log messages report
startup costs and distinguish startup from request attention hits without
logging conditioning tensors.

`codec_full_graphs` additionally captures the complete ten-step CFM solve for
these fixed windows, including explicitly qualified FP32 tail projections and
owned K/V prefixes. It defaults to false and requires codec optimizations,
chunk25 and a matching prepared manifest. Startup verifies original full/tail
projections, complete CFM outputs, waveform, layout and RNG before admitting
replay. Other windows use the original solve. Graphs are reset before their
parameter, cache and workspace leases are released.

The qualified implementation retains original block prefixes and evaluates each
tail with the original normalization, attention, FP32 projections, GELU and
residual order. Startup checks all intermediate values as well as complete CFM
and codec outputs. This remains optional; the default is off. After updating
model source, rebuild the prepared-cache manifest against the actual installed
source. A manifest from the prior implementation has stale source hashes and
must fail validation.

This option needs substantially more memory than the attention-only bank above.
The tested small-first isolated codec used 22.72 GiB allocated at readiness and
30.08 GiB at startup peak, with 11.895 s of preparation excluding model loading.
Paired service experiments explicitly used stage placement fractions
0.06/0.54/0.30 for preprocessing/AR/vocoder and typed AR
`runtime.sglang_server_args.mem_fraction_static=0.54`, retaining 4096 KV tokens
and four running requests. The flag does not change default resource settings;
placement fractions do not cap the vocoder allocator.

Compare complete codec output, sample ledgers, RNG and lifecycle behavior against
an eager reference using the same voice and chunk schedule. Historical experiment
matrices and job receipts belong outside the runtime package. Deployment limits
and the current single-GPU example are described in the cookbook.

## Additional exact execution options

All of these settings default to off. `6_8_align25` is a distinct waveform/chunk configuration: with the qualified 342-code reference voice, it emits after 9, 17 and 36 generated audio codes. Its first two PCM chunks contain 320ms and 640ms of audio. Compare every optimization against this configuration's own unoptimized waveform; the existing chunk25 and small-first waveforms remain independent references. The prepared bank/full CFM graphs use N1392/N1424/N1500 for this schedule. FP32 CFM, ten flow steps, request seeds and generation limits are unchanged.

Rebuild the prepared-prefix manifest against the installed MOSS source after upgrading. A manifest from the older source cannot be reused solely because the voice and weights are unchanged. Graph owners retain the original invalidation/fallback and close lifecycle. These features cover bounded AR/CFM subgraphs; preprocessing, sampling, request state, HiFT and HTTP remain outside a single graph for the whole pipeline. They do not establish C2/C4 capacity or guarantee 600ms first audio.

## Speech synthesis and transcription

These endpoints use fixed prompts to lower requests into the same canonical
chat pipeline. Speech synthesis asks the model to repeat the input;
transcription asks for an original-language transcript. These are prompted
behaviors, not guarantees of verbatim speech or perfect transcription.

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"moss-speech","input":"Hello, how are you?","voice":"default","response_format":"wav"}' \
  --output speech.wav

curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=moss-speech \
  -F file=@input.wav \
  -F response_format=json
```

Speech supports the deployed default voice, `speed=1`, and WAV or PCM output.
Custom voices, reference audio, instructions, language controls, and duration
controls are rejected. Omitted sampling parameters retain the model defaults;
explicit temperature, top-p, top-k, repetition penalty, seed, and token limits
are preserved.

Transcription defaults to greedy decoding. Language hints, prompts,
translation, language detection, and timestamps are rejected. Input audio uses
the same size, decoding, and resampling checks as chat audio input.

For streaming, start with the existing `moss_speech_streaming.yaml` configuration
and set `stream=true` on the request. Speech streams raw PCM with
`response_format=pcm`; transcription streams text events. The execution
options above can also be applied to that configuration. Offline and chunked
codec profiles use different weights and are not expected to produce the same
waveform; compare outputs within the same profile and seed.

## Validation

The asset-free CPU contracts run with `scripts/moss_speech/ci/run_cpu.sh`.
Native cache, lifecycle, and full-graph contracts are in
`tests/unit_test/moss_speech/` and require the installed SGLang runtime.
The asset-free CI lane excludes native AR graph imports. CUDA replay cases
need an allocated GPU. Real checkpoint regression uses the existing offline
`scripts/moss_speech/p5/native.py` and streaming
`scripts/moss_speech/p6/validate_http.py` drivers with a local manifest,
independent reference, and a YAML configuration containing the selected
options. Streaming validation supports `--lifecycle` for cancellation and
recovery checks. These drivers compare complete dual-channel grids rather
than only visible text.

The AR benchmark accepts locally staged fixtures and records three measured
rounds after warmup, actual batch sizes, cache hits, graph replay counts,
cancellation/recovery, grid equality, memory, and dependency/source versions:

```bash
python scripts/moss_speech/benchmark.py \
  --model-path /path/to/MOSS-Speech \
  --reference-dir /path/to/independent-reference \
  --profile combined --out /path/to/fresh-results
```

Use `--profile baseline` for the comparison, or `graph`, `metadata`, `mixed`,
and `cache` to isolate individual settings. `--prompt-case s2t_cn` measures
the longer repeated speech-input prefix. The reference directory must contain
the four fixtures generated by the independent reference workflow; it must
not contain model outputs generated by the benchmark under test. These AR
measurements do not include preprocessing, codec, or HTTP latency.

## Measured AR example

On the A800 and dependency versions above, three measured rounds after warmup
gave the following median completion times. B1/B2/B4 use the `t2s_cn` fixture
with 64 generated grid rows per request; the mixed workload uses one request
from each mode and generates 223 rows in total. Times are for the whole batch.

| Workload | Eager baseline | Projection graphs | All four options, warm cache |
| --- | ---: | ---: | ---: |
| B1 | 3.714 s | 1.700 s | 1.335 s |
| B2 | 6.497 s | 3.159 s | 2.438 s |
| B4 | 10.323 s | 5.877 s | 4.512 s |
| Mixed modes | 13.060 s | 5.990 s | 4.032 s |

The long-prefix cache benchmark (372 input rows, B4) reduced warm AR peak
allocated memory from 28.005 GiB to 23.771 GiB, with no latency improvement.
Cold prefill still needs the original memory budget. These measurements
exclude codec memory and HTTP overhead and do not establish 24 GB support
or real-time streaming throughput. Short-prompt first-token latency can
increase with caching even when total generation becomes faster.
