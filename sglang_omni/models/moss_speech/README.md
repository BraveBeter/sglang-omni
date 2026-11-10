# MOSS-Speech

This pipeline implements native dual-channel generation for text-to-text, text-to-speech, speech-to-text, and speech-to-speech requests. Text and audio terminals handle streaming and non-streaming requests. Audio output uses one configured reference voice; per-request voice cloning is not supported.

The implementation uses eager execution, a request-owned sampling generator, and the MOSS-Speech codec's WhisperVQ encoder, flow decoder, and HiFT vocoder. The checkpoint has 32 shared layers and four layers in each modality tail: its HF configuration describes 36 sequential layers, while paged attention needs 40 distinct KV slots.

## Configuration

Follow the [source installation guide](../../../docs/get_started/installation.md#install-from-source), including its CUDA build prerequisites, then install the model extra from the repository root in the activated environment:

```bash
uv pip install --prerelease=allow -e '.[moss-speech]'
```

The base runtime supplies ONNX Runtime and Diffusers; a second ONNX Runtime distribution is unnecessary.

Download the checkpoints and a reference WAV, then edit model_path, codec_path, and voice_wav in the example configuration. Assets are local paths; the AR tokenizer and configuration load without remote Python code. The [real-checkpoint smoke](#tests) preparation command below downloads these pinned revisions and produces a resolved configuration that can also be passed directly to sgl-omni serve.

| Asset | Validated revision |
|---|---|
| [OpenMOSS-Team/MOSS-Speech](https://huggingface.co/OpenMOSS-Team/MOSS-Speech) | cff025bb41d8459d59abac0b5e44aba7f659ec9e |
| [OpenMOSS-Team/MOSS-Speech-Codec](https://huggingface.co/OpenMOSS-Team/MOSS-Speech-Codec) | eeec733e4e1dea7da444d332d8e1621ef257414c |
| [Reference voice WAV](https://github.com/OpenMOSS/MOSS-Speech/blob/1ea408a10d07b9fdc7a27bce19d1211b62067784/assets/prompt-cn.wav) | MOSS-Speech Git revision 1ea408a10d07b9fdc7a27bce19d1211b62067784 |

Activate the installed environment before launch so standard SGLang CUDA kernel compilation can find its ninja executable. These runtime kernels may compile on the first request.

```bash
sgl-omni serve --config examples/configs/moss_speech.yaml
```

The default placement assigns preprocessing, AR, and audio decoding to GPU 0 in separate processes. The two-GPU example assigns AR to GPU 0 and the codec stages to GPU 1:

```bash
sgl-omni serve --config examples/configs/moss_speech_dual_gpu.yaml
```

Both placements use tensor parallel size one: the dual-GPU configuration places the codec stages on a second device and keeps AR on one GPU. The supported base uses BF16 AR weights/KV storage, torch_native attention, eager execution, and FP32 codec computation. Local real-checkpoint validation used one or two A800-SXM4-80GB GPUs. Other device capacities and tensor parallel sizes have not been validated.

Use the real-checkpoint smoke below to validate startup, output contracts, and recovery on the target GPUs. Measure deployment capacity and performance separately with fixed requests and output budgets.

Set deployment values through typed stage configuration, for example preprocessing.factory.voice_wav, preprocessing.factory.codec_path, audio_vocoder.factory.codec_path, and ar_engine.engine.max_running_requests. The streaming codec uses chunk_size=25, declared once in the vocoder factory configuration. Only the optional codec strategy documented below is included; it remains disabled by default.

### Optional CFM solver graph

The audio_vocoder.factory.cfm_graph_mel_length option defaults to null. Set one total mel-frame length, in multiples of 100 from 100 through 4000, to prepare one CUDA graph at startup with CUDA FP32 eval weights and chunk_size=25. The length includes the complete reference mel and generated mel; choose it explicitly for the configured workload.

Other lengths, final chunks, buffered decoding, changed weights or numerical settings, autocast, and calls on another CUDA stream use the native eager solver. Capture and replay errors propagate. Startup preserves caller RNG, each replay copies all six solver inputs, and returned mel storage remains owned by the request. No additional graph is captured per request. Actual capture and replay must be verified separately from eager fallback and service performance.

`audio_vocoder.factory.broadcast_attention_bias` is an optional strict boolean, defaulting to `false`. When enabled, both streaming and offline codec decoders use a live attention-bias broadcast for plain native CUDA FP32 B2 self-attention, avoiding physical head copies without caching bias values or changing the SDPA backend. Training, autograd, autocast, custom processors/preparers/hooks and unsupported shapes or dtypes retain the stock processor. The public mask-preparation API and checkpoint keys remain unchanged. A first actual optimized call emits `MOSS_CODEC_BIAS_BROADCAST_ACTIVE`; installation alone is not execution evidence. Local operator/Codec checks do not establish HTTP TTFA or RTF benefits.

## Requests

The service defaults to port 8000 and accepts /v1/chat/completions. Input content determines the input modality; modalities selects one output modality, text or audio. For a buffered text response:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello, introduce yourself briefly."}],"modalities":["text"],"stream":false,"temperature":0,"top_k":1,"seed":42,"max_tokens":200}'
```

For speech input and a buffered audio response, use a WAV recording of at most 30 seconds. Run this example with a local input.wav:

```python
import base64
from pathlib import Path

import requests

input_audio = base64.b64encode(Path("input.wav").read_bytes()).decode("ascii")
response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": [{
            "type": "input_audio",
            "input_audio": {"data": input_audio, "format": "wav"},
        }]}],
        "modalities": ["audio"],
        "stream": False,
        "temperature": 0,
        "top_k": 1,
        "seed": 42,
        "max_tokens": 512,
    },
    timeout=300,
)
response.raise_for_status()
audio = response.json()["choices"][0]["message"]["audio"]
Path("output.wav").write_bytes(base64.b64decode(audio["data"]))
```

Set modalities to ["text"] in the speech-input example for speech-to-text, or use a text content string with ["audio"] for text-to-speech. Buffered audio is a base64-encoded mono 24 kHz WAV. For streaming audio, set stream=true and audio={"format":"pcm"}; each SSE choice.delta.audio.data contains base64-encoded little-endian signed 16-bit mono PCM at 24 kHz. Concatenate decoded chunks in arrival order and add a WAV header when saving as WAV.

The optional output budget max_tokens (or native max_new_tokens) counts generated two-channel grid rows, including delimiters. When omitted or null, the server uses 40,960 minus the actual canonical prompt rows after left-padding removal. An explicit positive budget must fit that same remaining context; there is no separate 200-row default or 1000-row generation ceiling. Generation stops early at the model EOS, or returns finish_reason="length" when the effective budget is exhausted. A prompt filling all 40,960 rows is rejected because at least one generation row is required. Streaming text, audio and Codec accumulation remain bounded by the total context, while the AR scheduler enforces each request's effective budget. A successful HTTP response does not establish task compliance or audio quality.

The checkpoint supports a total context of 40,960 two-channel rows. Preprocessing uses exact canonical token counts rather than character estimates; the default engine context is 40,960; the shared KV pool has 40,962 physical slots, including two allocator/admission guards. The model-specific scheduler sentinel accounts for upstream clipping without reducing the logical window. Prompt plus requested generation budget must fit that window. Raising the total context increases KV memory and does not guarantee four simultaneous full-window requests.

## Validation status

The eager base has passed local CPU regressions, native CUDA checks, four-mode streaming/buffered service smoke and strict concurrent-versus-serial comparisons on single- and dual-A800 placements. Wheel/sdist contents and a regular wheel installation were also checked. These are local results; upstream H100 CI has not been executed. Calibrated latency/throughput, maximum concurrency and comprehensive quality evaluation remain separate work.

Third-party source revisions, retained notices and outstanding statement scope are documented in [vendored source provenance](../../vendor/moss_speech_codec/VENDORED_SOURCES.md). Checkpoint terms are separate from source terms; no weights are bundled.

Waiting streaming-codec pushes validate and store codes before returning when the next hop and lookahead are not ready. These pushes skip CPU/CUDA RNG and cuDNN scopes while preserving per-session RNG, context-budget validation, final flushing and failure cleanup. No configuration setting is required. End-to-end performance must be measured for the actual deployment.

The boundary regressions cover the trained chunk profiles, final requests, session isolation and the context limit:

```bash
SGLANG_USE_CPU_ENGINE=1 python -m pytest \
  tests/unit_test/moss_speech/test_codec.py -k unready -v
```

## Output and comparison contract

Each generated row contains a text token and an audio token. A modality-pad text token selects the audio channel; text-channel end-of-text and message-end tokens terminate generation. EOSP closes the audio segment and returns the finite-state machine to text mode; EOSP in an inactive channel does not terminate the request.

Audio decoding keeps every valid active-channel code before the first EOSP. An N-row audio segment containing EOSP has N-1 valid codes; a length-limited segment with N valid codes retains all N. Completion usage counts generated grid rows, including delimiters. Compare both the raw grid and extracted codes, and keep the reported stop or length reason. Do not drop a final valid code to make another decoder agree.

The streaming codec emits 1,920 samples per retained code at 24 kHz. Its final hop flushes held overlap; the sum of emitted chunk samples must equal the final sample ledger. The native pipeline result carries usage and sample-ledger metadata and does not replay audio already sent in deltas. The chat SSE interface exposes audio deltas, finish reason, and final usage; it does not expose the internal sample ledger or raw token grid. The complete-utterance and chunk-trained flow checkpoints are different decoder profiles, so waveform equality between those profiles is not assumed.

## Tests

The unit suite is self-contained and generates tiny model and codec weights locally. Its checked-in processor fixture contains complete input codes and frozen grid provenance; it does not require downloaded model caches or workspace audit files.

```bash
SGLANG_USE_CPU_ENGINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests/unit_test/moss_speech -m 'not accelerator'
```

Set the official CPU platform flag before importing SGLang. GPU cases are selected with -m accelerator and require a supported allocated GPU; collecting or skipping them does not establish a GPU validation result. Full checkpoint and tokenizer parity, four-mode service execution, quality, and performance remain separate integration checks.

The real-checkpoint service smoke consumes a resolved single- or two-GPU configuration and pinned English/Chinese speech inputs. Prepare its assets on a machine with network access, then run inside an allocated GPU environment:

```bash
python .github/scripts/prepare_moss_speech_ci.py \
  --output-directory /path/to/moss-smoke --gpu-count 1
export MOSS_SPEECH_CI_CONFIG=/path/to/moss-smoke/moss_speech.yaml
export MOSS_SPEECH_CI_AUDIO_EN=/path/to/moss-smoke/input-en.wav
export MOSS_SPEECH_CI_AUDIO_ZH=/path/to/moss-smoke/input-cn.wav
export MOSS_SPEECH_CI_OUTPUT_DIR=/path/to/moss-smoke/results
python -m pytest tests/test_model/test_moss_speech_ci.py -m accelerator -v -s -x
```

The prepared speech inputs retain the first 20 seconds of each pinned reference recording, with source hashes and selected frame ranges recorded separately. The configured synthesis voice uses the complete Chinese reference WAV. The fixture rejects invalid or overlong input recordings before starting a server.

The smoke covers both languages, all four input/output modes, streaming and non-streaming, natural text EOS, audio length truncation, disconnect recovery, and invalid-request recovery. It retains HTTP responses, generated WAVs, usage, timing observations, asset hashes, and server logs. A first PCM event can contain silence; non-streaming TTFA and TTFT are unobserved. These observations have no calibrated performance or quality thresholds, and the chat API does not expose the native token grid for exact retained-code comparison.

The central Omni workflow selects this dedicated smoke with the run-omni-ci and run-moss-speech labels. It does not add MOSS-Speech to the random model selection. Local checks and workflow wiring do not establish a successful remote CI run.


## Full decode graph experiment

Set `ar_engine.factory.decode_graph: true` to prepare an exact-length B1 decode bank for
sequence lengths 1 through 512 on the qualified Torch 2.14.1+cu130 / CUDA 13.0 / SM80 BF16 runtime.
The default is false. SGLang's generic graph switch remains disabled. Prefill,
sampling and request cleanup retain their eager paths; other batch sizes, changed
weights, hooks, attention policy, pools or CUDA stream use eager execution.
Each graph owns staging and cloned outputs and retains native SDPA shapes. Longer
sequences use eager execution. Startup and resident graph memory must be measured. It
writes every shared/text/audio KV layer at the scheduler's allocated locations.
Startup restores reserved slot zero. Shutdown synchronizes and releases capture
resources. This candidate requires correctness and service measurement before use;
local graph speed does not establish an HTTP latency or quality improvement.

Set `ar_engine.factory.decode_graph_serial_batch: true` together with `decode_graph: true` to opt into request-local serial replay for engine batches of exactly two or four requests. This separate flag defaults to false; the existing B1 path is unchanged. Every request reuses its exact-length B1 entry without padding or a batch-by-length graph bank. All rows, original embedding IDs and disjoint KV destinations are validated before the first replay; unsupported metadata, lengths above 512, batch three and runtime changes fall back as a whole. Replay errors propagate rather than retrying eager after partial KV mutation. Returned dual-head and hidden tensors own their storage. Actual execution emits `MOSS serial batch decode graph replay active batch=2` or `batch=4`; startup and HTTP concurrency do not prove activation. This path requires its own full-checkpoint/HF correctness and end-to-end measurements; model-local timing improvements are not TTFA or RTF results.


### Optional native norm pointwise arithmetic

Set `ar_engine.factory.native_norm_library` to an explicitly built shared library. The default is `None`. This SM80 experiment supports BF16 tensors of shapes `(1, 4096)`, `(1, 32, 128)` and `(1, 8, 128)`. It retains the original Torch mean reduction, residual addition and BF16 rounding before multiplication by the weight. Both `no_grad` (the default scheduler mode) and `inference_mode` are supported. Training, gradients and unsupported shapes, types, devices or layouts use the original Torch norm. Every output owns its storage.

From this model directory, build the CUDA source and record its hashes:

```bash
nvcc -shared -Xcompiler=-fPIC -O3 -std=c++17 -arch=sm_80 --fmad=false \
  -o /path/to/native_norm.so kernels/decode_norm_native.cu
sha256sum kernels/decode_norm_native.cu /path/to/native_norm.so
```

The loader compares the library's compile-time `CUDART_VERSION` with `torch.version.cuda`; this checks the CUDA runtime headers, rather than the NVCC release string. No library compilation occurs during import or request processing. The wheel and sdist contain the CUDA source, and the operator builds the optional library separately.

Use benchmark-client with an explicit candidate SHA and pinned library hash for correctness before comparing service performance. Attention, KV and projection remain eager Torch. Combining this option with another optimization requires its own validation.

When native norm and serial decode Graph are enabled together, the builder installs the norm library before capture. Captures retain normalization owners and library identity; changed normalization configuration or execution policy rejects replay before KV mutation. The combined deployment requires fresh joint complete-head and KV qualification against eager Torch in addition to the independent HF checks.


### Batched decode RoPE reuse

`ar_engine.factory.decode_rope_reuse` defaults to `false`. When enabled, inference-only eager decode batches of at least two requests reuse the original-shape cosine/sine tensors within one forward across the compatible shared and modality-tail layers. Frequency construction, FP32 position multiplication, trigonometric operations, BF16 rounding, attention/KV operations and per-request projection shapes are unchanged. No request tensors are retained between forwards.

Prefill, batch-one decode, training/gradient/autocast execution, forward hooks, or incompatible layer head dimensions, RoPE theta, dtype or device retain the per-layer eager path. Successful eligible forwards increment `model.decode_rope_reuse_forwards`; the first emits `batched RoPE reuse active`. These indicate path activation, not a performance guarantee. Independent-HF correctness and matched end-to-end benchmarks must qualify each final candidate/configuration before deployment.

## Optional asynchronous Codec

Set --audio_vocoder.factory.codec_flow_slots 2 for two bounded B1 Flow workers, or 1 for the submission/completion ablation; the default 0 keeps synchronous decoding. Every worker owns its Flow state, CUDA stream and CFM Graph. This is concurrent B1 execution, with each request's two CFG rows preserved. HiFT and per-request RNG/cache/PCM commits remain on one serving owner. The scheduler ingests queued codes between bounded completion steps and wakes on finished work, with at most one in-flight hop per request. Cancellation discards late results without reusing an occupied slot. The existing cfm_graph_mel_length applies independently to each worker; graph capture happens on that worker's stream. This experimental path requires separate correctness, memory, throughput and playback-continuity qualification before deployment.
