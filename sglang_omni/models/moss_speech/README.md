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

Set deployment values through typed stage configuration, for example preprocessing.factory.voice_wav, preprocessing.factory.codec_path, audio_vocoder.factory.codec_path, and ar_engine.engine.max_running_requests. The streaming codec uses chunk_size=25, declared once in the vocoder factory configuration. Optional optimization flags from the experimental branch are not part of this pipeline.

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

The checkpoint supports a total context of 40,960 two-channel rows. Preprocessing uses exact canonical token counts rather than character estimates; the default engine context is 40,960; the shared KV pool has 40,962 physical slots, including two allocator/admission guards. The model-specific scheduler sentinel accounts for upstream clipping without reducing the logical window. Prompt plus requested generation budget must fit that window. The independent output budget is 1–1000 rows (default 200); it is not the model context size. Raising the total context increases KV memory and does not guarantee four simultaneous full-window requests.

## Validation status

The eager base has passed local CPU regressions, native CUDA checks, four-mode streaming/buffered service smoke and strict concurrent-versus-serial comparisons on single- and dual-A800 placements. Wheel/sdist contents and a regular wheel installation were also checked. These are local results; upstream H100 CI has not been executed. Calibrated latency/throughput, maximum concurrency and comprehensive quality evaluation remain separate work.

Third-party source revisions, retained notices and outstanding statement scope are documented in [vendored source provenance](../../vendor/moss_speech_codec/VENDORED_SOURCES.md). Checkpoint terms are separate from source terms; no weights are bundled.

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
