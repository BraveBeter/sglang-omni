# MOSS-Speech

MOSS-Speech provides text/speech input and text/speech output, with an opt-in
streaming profile, through
`POST /v1/chat/completions`. It is a conversation model; speech transcription and
verbatim reading are prompting tasks, without a dedicated ASR/TTS quality claim.
Two streaming deployment presets are available: one A800 80 GB, and two A800
80 GB GPUs with the AR stage separated from the input encoder and codec. Both
use TP1, BF16 AR/KV and FP32 codec. The two-GPU preset also enables the measured
AR/CFM graph and fixed-voice optimizations. These are stage placements, not tensor
parallelism or two independent replicas. Codec execution concurrency is 1.

## Installation and model assets

Install the checkout containing MOSS-Speech support in your SGLang-Omni runtime
by following [Installation](../get_started/installation.md). Include the
`moss-speech` extra when installing from source. Commands below run from the
parent directory of the `sglang-omni/` checkout:

```bash
uv pip install -e './sglang-omni[moss-speech]'
sgl-omni serve --help
```

Use the standard `sgl-omni` command in your installed runtime or container.
The `moss-speech` extra supplies codec dependencies; model weights are downloaded
separately.

Download the qualified model and codec revisions on a network-enabled machine:

```bash
hf download OpenMOSS-Team/MOSS-Speech \
  --revision cff025bb41d8459d59abac0b5e44aba7f659ec9e \
  --local-dir models/MOSS-Speech
hf download OpenMOSS-Team/MOSS-Speech-Codec \
  --revision eeec733e4e1dea7da444d332d8e1621ef257414c \
  --local-dir models/MOSS-Speech-Codec
```

Provide a local default-voice WAV, shown as `voice.wav` below. This is a server
asset used to condition speech output; it is separate from user speech input.
The server needs the AR/tokenizer directory, the full codec directory and this
voice file. No reference-source checkout is required for serving. The native
path does not execute `trust_remote_code`.

## Server configuration

Run the following commands from the parent directory of `sglang-omni/`, on an
allocated GPU host. Download assets before using an offline compute node:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
```

The presets contain no machine-specific paths. Supply local assets through the
existing `sgl-omni serve` command; relative paths are relative to the launch
working directory. No generated configuration or model-specific launcher is needed.

| Preset | GPU placement | Streaming schedule | Graphs and caching |
| --- | --- | --- | --- |
| [Single GPU](../../examples/configs/moss_speech_single_gpu.yaml) | Input encoder, AR and codec on GPU 0 | chunk25 | Decode projection graphs; full AR/prefill/CFM graphs off; AR prefix KV cache off |
| [Dual GPU](../../examples/configs/moss_speech_dual_gpu.yaml) | AR on GPU 0; input encoder and codec on GPU 1 | chunk25 with `6_8_align25` startup | Full AR decode/prefill and CFM graphs, qualified norm/QKV/final-bias/native-epoch options, fixed-voice CFM cache; AR prefix KV cache off |

### Single A800 80 GB

```bash
sgl-omni serve \
  --config sglang-omni/examples/configs/moss_speech_single_gpu.yaml \
  --model-path models/MOSS-Speech \
  --codec-path models/MOSS-Speech-Codec \
  --voice-wav voice.wav \
  --model-name moss-speech --host 127.0.0.1 --port 8000
```

This is the single-GPU chunk25 profile evaluated on 2026-09-24. It needs the model,
codec and default voice, with no separate optimization packages. The codec must
include `flow/flow-chunk-25.pt` and the offline decoder assets.

### Two A800 80 GB GPUs

The best measured profile uses Python 3.10, Torch 2.9.1+cu128, SGLang 0.5.16,
Transformers 5.12.1 and Diffusers 0.37.0 on two A800-SXM4-80GB GPUs. The general
project installation currently selects a newer Torch/CUDA stack. For this
explicit profile, create a separate environment and use the supplied overrides:

```bash
uv venv --python 3.10 .venv-moss
source .venv-moss/bin/activate
uv pip install --torch-backend=cu128 \
  --overrides sglang-omni/examples/configs/moss_speech_cu128.txt \
  -e './sglang-omni[moss-speech]'
```

Install CUDA Toolkit 12.8 (`nvcc`) and a C++17 compiler (`g++`) on the build host.
Build in the same Python environment used for serving. Native packages are local
build artifacts: no private binary, source checkout outside this repository, or
reference-model implementation is needed. The source files also ship in the wheel.

The measured voice is `assets/prompt-cn.wav` from the upstream MOSS-Speech
revision `1ea408a10d07b9fdc7a27bce19d1211b62067784`. Download it before going offline:

```bash
curl --fail --location \
  https://raw.githubusercontent.com/OpenMOSS/MOSS-Speech/1ea408a10d07b9fdc7a27bce19d1211b62067784/assets/prompt-cn.wav \
  --output voice.wav
```

On the allocated GPU host, create three fresh output directories using the
installed preparation module. `--nvcc /path/to/nvcc` and `--cxx /path/to/g++` can
select compilers that are not on `PATH`:

```bash
python -m sglang_omni.models.moss_speech.prepare norm --out prepared/norm
python -m sglang_omni.models.moss_speech.prepare epoch --out prepared/epoch
python -m sglang_omni.models.moss_speech.prepare prefix \
  --codec-path models/MOSS-Speech-Codec --voice-wav voice.wav \
  --device 1 --out prepared/prefix

export MOSS_NORM_MANIFEST=prepared/norm/manifest.json
export MOSS_PREFIX_MANIFEST=prepared/prefix/manifest.json
export MOSS_EPOCH_MANIFEST=prepared/epoch/manifest.json
```

The norm build compares all intermediate tensors and CUDA graph replays with
Torch arithmetic for the supported shapes. The epoch build checks live metadata
against the Python traversal, including mutations and unsupported overrides.
Prefix preparation encodes the voice, verifies its 342-code shape and compares
original, cached and complete-CFM-graph execution before publishing its manifest.
Failed builds leave diagnostics but no deployable manifest. These bounded checks
do not replace a workload-level correctness and performance benchmark.

Keep each complete output directory. Package paths are relative to their manifest;
source, binary, runtime and validation fingerprints are checked on load. The
prepared voice directory can be relocated together; after source/environment
changes, rerun preparation in fresh directories. A different voice is not an
interchangeable override: this profile currently requires 342 reference codes.
Serving loads and verifies prepared artifacts and never invokes a compiler.

```bash
sgl-omni serve \
  --config sglang-omni/examples/configs/moss_speech_dual_gpu.yaml \
  --model-path models/MOSS-Speech \
  --codec-path models/MOSS-Speech-Codec \
  --voice-wav voice.wav \
  --decode-norm-kernel-manifest "${MOSS_NORM_MANIFEST:?Set the qualified norm manifest}" \
  --codec-prefix-cache-manifest "${MOSS_PREFIX_MANIFEST:?Set the prepared voice manifest}" \
  --codec-native-epoch-manifest "${MOSS_EPOCH_MANIFEST:?Set the qualified epoch manifest}" \
  --model-name moss-speech --host 127.0.0.1 --port 8000
```

The manifest paths can instead be set as the corresponding top-level YAML
fields. GPU IDs are logical indices within `CUDA_VISIBLE_DEVICES`. On a directly
managed host, select one card with `CUDA_VISIBLE_DEVICES=0` or two with
`CUDA_VISIBLE_DEVICES=0,1` before the command. On Slurm, preserve the allocation's
GPU visibility and request one or two GPUs respectively. Neither preset needs
MPS; both keep AR tensor parallelism at 1.

`6_8_align25` emits after 9, 17 and 36 generated audio codes, then uses 25-code
hops. The first two audio chunks contain 320 ms and 640 ms. It has its own
waveform reference and should not be compared byte-for-byte with ordinary
chunk25. Full AR graph coverage is principally batch size 1; unsupported shapes
use the existing fallback. The fixed-voice CFM cache is separate from prefix KV
reuse in AR, which is disabled in both presets.

### Readiness, shutdown and offline output

```bash
curl --fail http://127.0.0.1:8000/health
```

Wait for readiness before sending requests. Ctrl-C or SIGTERM stops the service
and workers. Use `--host 0.0.0.0` when clients connect from another machine.

For the original non-streaming pipeline, use
`--config sglang-omni/examples/configs/moss_speech.yaml` with the same model,
codec and voice arguments. Model-path autodiscovery also selects that offline
pipeline; it rejects `stream=true`. The legacy
`sglang-omni/examples/configs/moss_speech_streaming.yaml` remains a chunk5
compatibility example. Use the new chunk25 presets for deployment; chunk5 is
excluded because of its measured latency.

## Four-mode requests

```python
import base64
from pathlib import Path
import httpx

speech_input = False   # True: Speech->Text or Speech->Speech
speech_output = False  # True: Text->Speech or Speech->Speech
content = "Introduce yourself in one sentence."
if speech_input:
    content = [{"type": "input_audio", "input_audio": {
        "data": base64.b64encode(Path("input.wav").read_bytes()).decode(),
        "format": "wav",
    }}]
system = ("You are a helpful voice assistant. Answer the user's questions with spoken responses."
          if speech_output else
          "You are a helpful assistant. Answer the user's questions with text.")
body = {
    "model": "moss-speech",
    "messages": [{"role": "system", "content": system},
                 {"role": "user", "content": content}],
    "modalities": ["audio" if speech_output else "text"],
    "stream": False, "temperature": 0.0, "top_p": 1.0,
    "top_k": -1, "repetition_penalty": 1.1, "seed": 0, "max_tokens": 200,
}
with httpx.Client(timeout=600) as client:
    response = client.post("http://127.0.0.1:8000/v1/chat/completions", json=body)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if speech_output:
        Path("output.wav").write_bytes(base64.b64decode(choice["message"]["audio"]["data"]))
    else:
        print(choice["message"]["content"])
    print(choice["finish_reason"])
```

Output audio is mono 24 kHz WAV using the configured voice. `modalities` selects
exactly one output terminal. Inline WAV is the tested input interchange;
`audios[]` can bind empty audio parts in order. URLs and uploaded voice selection
are unsupported. Other audio formats depend on the CPU soundfile decoder;
fixing the common endpoint's TorchCodec/M4A dependency does not add MOSS support
for that format.

Omitted sampling values use 0.6/0.95/20/1.1; explicit values are preserved.
`max_tokens` and `max_completion_tokens` are aliases; conflicting values fail.
The default output budget is 200 grid rows. `finish_reason=length` means the
utterance may be incomplete. Usage counts one dual-channel grid row as one token,
including template/input-audio positions. A disconnect aborts outstanding work.

## Streaming requests

Both single- and dual-GPU presets above enable streaming. The vocoder worker
sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and deterministic cuDNN
is scoped to the streaming codec. Keep these settings when reproducing the
measured profiles.

Set `stream=True` on the four-mode request above. For speech output, explicitly
select PCM to obtain headerless signed 16-bit little-endian samples at 24 kHz mono:

```python
import json

body["stream"] = True
if speech_output:
    body["audio"] = {"format": "pcm"}
with httpx.Client(timeout=900) as client, open("output.pcm", "wb") as audio_file:
    with client.stream("POST", "http://127.0.0.1:8000/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            choice = event["choices"][0]
            delta = choice["delta"]
            print(delta.get("content") or "", end="", flush=True)
            if delta.get("audio", {}).get("data"):
                audio_file.write(base64.b64decode(delta["audio"]["data"]))
            if choice.get("finish_reason"):
                print(choice["finish_reason"], event.get("usage"))
```

WAV is also supported, but each audio delta is a complete WAV container; decode
each one and concatenate its samples instead of concatenating file headers.
Text buffers incomplete UTF-8. If a length cap splits a multibyte sequence, only
the valid Unicode prefix is emitted; full token usage and `length` remain intact.
A successful stream has one finish/usage event and one `[DONE]`. A post-header
failure has a sanitized SSE error and ends without success finish or `[DONE]`.
Closing the connection aborts unfinished backend work.

Streaming speech uses separately trained flow weights and prefix/overlap
decoding, so it need not equal the offline waveform. Compare against a reference
using the same chunk and startup schedule.

## Measured deployment results

The 2026-09-24 regression used 38 bilingual modality tasks: 32 derived from eight
SeedTTS source rows, plus six spoken-QA tasks from three OpenAudioBench questions.
Performance used three load rounds per client concurrency. The C1 statistics
below include 57 speech requests, of which six reached the output limit. They
are observations on A800-SXM4-80GB, not production latency guarantees.

| Metric | Single GPU, chunk25 | Dual GPU, `6_8_align25` |
| --- | ---: | ---: |
| First audio, p50 / p95 | 2.299 / 2.338 s | 0.606 / 0.638 s |
| End-to-end RTF, p50 / p95 | 0.890 / 1.127 | 0.638 / 0.811 |
| Pipeline-ready startup | 40.53 s | 54.07 s |
| Peak sampled GPU memory, including startup | 27.88 GiB | AR 30.96 GiB; encoder/codec 49.22 GiB |

Start with client concurrency 1. C2/C4 increased first-audio and completion
latency sharply with little throughput improvement. RTF below 1 does not by
itself guarantee uninterrupted playback. The deployments differ in graph/cache
options and startup scheduling as well as GPU count.

Both streaming profiles matched all 38 reference token grids, decoded text and
finish reasons. The dual-GPU optimized profile also matched all 19 waveforms
from its ordinary-execution reference using the same startup schedule. The
HF offline decoder is not a streaming PCM reference. The separate single-GPU
offline test retained two strict PCM mismatches on length-capped S2S outputs:
native decoded 512 valid codes, while the upstream processor decoded 511.
This does not establish that all offline outputs are equivalent.

## Capacity and numerical contract

The A800 preset allows 512 prompt rows and 512 generated rows per request, a
1,024-row context, four active AR requests, 20 waiting requests and 4,096 total
KV positions. The 25th in-flight request returns
503. Invalid input returns 400/422; exact post-encode overflow returns 400 before
AR. For latency-sensitive use start with client concurrency 1.

The 2026-09-24 single-GPU non-streaming workload peaked at 29.78 GiB. A 24 GB
4090 could not fit the FP32 vocoder working set, even with AR concurrency 1 and
KV 1024; there is no qualified 24 GB serving preset.

The verified runtime is Python 3.10, torch/torchaudio 2.9.1+cu128,
transformers 5.12.1, SGLang 0.5.16 and TorchCodec 0.8.1. The installation guide's
newer default CUDA 13/torch 2.11 stack has not been qualified for this model.
The small bilingual regression sample does not establish a general quality
claim; no human MOS or calibrated quality/performance threshold is provided.

Checkpoint and HF-code redistribution terms remain unresolved; weights are not
bundled. Source revisions, modifications and license notes are retained in
[Vendored sources](../../sglang_omni/models/moss_speech/components/VENDORED_SOURCES.md).

### Text at the output limit

Text responses decode every generated text-channel token, including an ordinary
last token when `finish_reason="length"`. Special tokens are filtered; trailing
Unicode replacement characters from an incomplete byte sequence are withheld,
matching the streaming terminal. Grid/usage counts retain those tokens.

This integration behavior differs from the locked upstream processor, which
unconditionally removes the last grid row before decoding. The service preserves
ordinary final tokens for both streaming and non-streaming responses.
