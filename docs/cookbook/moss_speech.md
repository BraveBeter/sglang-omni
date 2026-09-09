# MOSS-Speech

MOSS-Speech provides text/speech input and text/speech output, with an opt-in
streaming profile, through
`POST /v1/chat/completions`. It is a conversation model; speech transcription and
verbatim reading are prompting tasks, without a dedicated ASR/TTS quality claim.
The measured baseline is one A80080GB, TP1, BF16 native AR/KV, FP32 codec, eager
PyTorch attention. Uploaded voices, radix caching, CUDA Graph, compile,
quantization and TP>1 are disabled. The pipeline uses one inline input encoder and
separate text/audio terminals. Codec execution concurrency is 1.

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

## Start the server

On a GPU host, launch directly with local asset paths:

```bash
sgl-omni serve \
  --model-path models/MOSS-Speech \
  --codec-path models/MOSS-Speech-Codec \
  --voice-wav voice.wav \
  --host 127.0.0.1 --port 8000
```

`--model-path` discovers the MOSS-Speech pipeline automatically. `--codec-path`
and `--voice-wav` are model-specific configuration options handled by the normal
CLI. Relative asset paths are relative to the launch directory. The codec path
may be omitted when it is the sibling `<model-path>-Codec` directory. Explicit
voice configuration takes precedence over `MOSS_SPEECH_VOICE_WAV`, which remains
available for existing deployments.

A portable optional preset is available at
`sglang-omni/examples/configs/moss_speech.yaml`; pass it with `--config` and the
same asset options if you prefer file-based configuration. The file contains
no operator-specific paths. `/health` reports readiness; SIGTERM or Ctrl-C stops
the server and its workers. On Slurm, run the command inside an allocated GPU
job. For offline deployment, download assets first and set `HF_HUB_OFFLINE=1`
and `TRANSFORMERS_OFFLINE=1`.

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

## Explicit streaming variant

For incremental output, select the supplied streaming preset. It uses the same
asset options and installed console command:

```bash
sgl-omni serve \
  --config sglang-omni/examples/configs/moss_speech_streaming.yaml \
  --model-path models/MOSS-Speech \
  --codec-path models/MOSS-Speech-Codec \
  --voice-wav voice.wav \
  --host 127.0.0.1 --port 8000
```

The default pipeline rejects `stream=True`; the streaming preset enables it
explicitly. No config-generation step is needed.

This requires `flow/flow-chunk-5.pt` in addition to the offline codec assets.
Chunk25 is component-tested separately; the HTTP baseline uses chunk5. The
profile keeps BF16 AR/KV and FP32 codec, with deterministic cuDNN scoped to the
streaming codec. The vocoder worker also defaults
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid accumulating unused
CUDA reservations as prefixes grow. Keep this measured setting; overriding it
requires separate memory qualification. It does not change the default offline
computation or the AR worker environment.

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

Streaming speech uses separately trained flow weights and prefix/overlap decoding;
it need not equal the offline waveform. The hard comparison is against an
independent reference with the same streaming profile. Early audio delivery does
not establish real-time playback throughput or a latency SLA. Do not reuse the
offline memory estimate for streaming. On the fixed 16 audio cases, client
TTFA p50/p95 was 2.563/2.719s and E2E RTF p50/p95 was 3.868/5.639. This profile
therefore does not sustain real-time playback. The full streaming run sampled
26.487GiB peak GPU memory, including four two-request waves with identical idle
memory. HTTP concurrency 2 was verified; the inherited AR limit 4 is not a measured
four-stream load claim. Chunk25 and other hardware need separate HTTP qualification.

## Capacity and numerical contract

The A800 preset allows 512 prompt rows and 512 generated rows per request, a
1,024-row context, four active AR requests, 20 waiting requests and 4,096 total
KV positions. The 25th in-flight request returns
503. Invalid input returns 400/422; exact post-encode overflow returns 400 before
AR. For latency-sensitive use start with client concurrency 1.

The measured non-streaming A800 workloads peaked at 27.03–29.19 GiB. A 24 GB
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
