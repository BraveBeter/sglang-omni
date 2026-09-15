# MOSS-Speech runtime options

MOSS-Speech supports chat, speech synthesis, and transcription through the
existing `sgl-omni serve` application. The default execution profile remains
single-GPU BF16 AR/KV with FP32 codec, eager `torch_native` attention, and no
prefix cache. The tested device is an A800 80 GB. Quantization, tensor
parallelism, full-model CUDA Graph, and `torch.compile` remain unsupported;
this configuration does not support a 24 GB GPU. Qualification used
Torch 2.9.1+cu128, SGLang 0.5.16, and Transformers 5.12.1.

## Optional execution settings

The following model options accept explicit `true` or `false` values through
`sgl-omni serve` or their underscore names as top-level YAML fields:

| CLI option | Behavior |
| --- | --- |
| `--decode-projection-graphs true` | Capture the norm, projection, RoPE, and MLP sections of each decode layer for batch sizes 1–4. Attention, KV updates, sampling, and request state remain eager. Prefill and other batch sizes use eager execution. |
| `--eager-cpu-metadata true` | Prepare the attention loop's indexing metadata on the CPU once per batch. Model activations and KV tensors remain on the GPU. |
| `--mixed-modalities true` | Allow text-output and audio-output requests in the same AR batch. Each request retains its own sampling, seed, and stopping state. |
| `--prefix-cache true` | Reuse KV through the existing radix cache with keys containing both text and audio tokens. Reuse requires equal original prompt lengths to preserve BF16 kernel shapes. |

For example, with checkpoints and the configured default voice already staged:

```bash
sgl-omni serve \
  --model-path /path/to/MOSS-Speech \
  --codec-path /path/to/MOSS-Speech-Codec \
  --voice-wav /path/to/voice.wav \
  --decode-projection-graphs true \
  --eager-cpu-metadata true \
  --mixed-modalities true \
  --host 127.0.0.1 --port 8000
```

All options are off by default. They preserve the existing request limits:
512 input grid rows, 512 generated grid rows, context length 1024, and the
configured KV pool capacity. Prefix caching can reduce redundant KV work,
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
Native cache, lifecycle, and projection tests are in
`tests/unit_test/moss_speech/`; the CUDA projection replay cases need an
allocated GPU. Real checkpoint regression uses the existing offline
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
