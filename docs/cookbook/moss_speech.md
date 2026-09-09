# MOSS-Speech

MOSS-Speech provides text/speech input and text/speech output, with an opt-in
streaming profile, through
`POST /v1/chat/completions`. It is a conversation model; speech transcription and
verbatim reading are prompting tasks, without a dedicated ASR/TTS quality claim.
The measured baseline is one A80080GB, TP1, BF16 native AR/KV, FP32 codec, eager
PyTorch attention. Uploaded voices, radix caching, CUDA Graph, compile,
quantization and TP>1 are disabled. The pipeline uses one inline input encoder and
separate text/audio terminals. Codec execution concurrency is1.

## Installation and model assets

Install the checkout containing MOSS-Speech support in your SGLang-Omni runtime
by following [Installation](../get_started/installation.md). Include the
`moss-speech` extra when installing from source. Commands below run from the
parent directory of the `sglang-omni/` checkout:

```bash
uv pip install -e './sglang-omni[moss-speech]'
sglang-omni serve --help
```

`sglang-omni` and `sgl-omni` are equivalent installed console commands. Use either
in your existing runtime or container; serving does not require a specially
named virtual environment, a Python module command, or a generated YAML file.
If upgrading an existing checkout, reinstall the package to register the new
`sglang-omni` alias. The `moss-speech` extra supplies the codec dependencies;
model weights are downloaded separately.

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
sglang-omni serve \
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

Output audio is mono24kHz WAV using the configured voice. `modalities` selects
exactly one output terminal. Inline WAV is the tested input interchange;
`audios[]` can bind empty audio parts in order. URLs and uploaded voice selection
are unsupported. Other audio formats depend on the CPU soundfile decoder;
fixing the common endpoint's TorchCodec/M4A dependency does not add MOSS support
for that format.

Omitted sampling values use0.6/0.95/20/1.1; explicit values are preserved.
`max_tokens` and `max_completion_tokens` are aliases; conflicting values fail.
The default output budget is200 grid rows. `finish_reason=length` means the
utterance may be incomplete. Usage counts one dual-channel grid row as one token,
including template/input-audio positions. A disconnect aborts outstanding work.

## Explicit streaming variant

For incremental output, select the supplied streaming preset. It uses the same
asset options and installed console command:

```bash
sglang-omni serve \
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
select PCM to obtain headerless signed16-bit little-endian samples at24kHz mono:

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
P5 offline memory estimate for streaming. On the fixed 16 audio cases, client
TTFA p50/p95 was 2.563/2.719s and E2E RTF p50/p95 was 3.868/5.639. This profile
therefore does not sustain real-time playback. The full streaming run sampled
26.487GiB peak GPU memory, including four two-request waves with identical idle
memory. HTTP concurrency2 was verified; the inherited AR limit4 is not a measured
four-stream load claim. Chunk25 and other hardware need separate HTTP qualification.

All 32 canonical requests preserved the frozen 4001-row AR grid; all 16 PCM
outputs exactly matched the independent streaming reference. P5 and P6 aggregate
ASR scores were unchanged, including Chinese T2S CER38.75% and two S2S length
stops. This small sample and the absence of human listening scores limit quality
claims. Details and retained failures are in
`sglang-omni/docs/design/moss_speech/p6/03_gate_report.md`.

Full component, HTTP, offline-regression and independent ASR reproduction is in
`sglang-omni/docs/design/moss_speech/p6/02_reproduction.md`. The CPU CI entry also
runs P6 contracts. The separate GPU entry is
`sglang-omni/scripts/moss_speech/ci/run_streaming_gpu.sh`; the original
`sglang-omni/scripts/moss_speech/ci/run_gpu.sh` retains the offline profile.

## Capacity and numerical contract

A800 configuration: prompt<=512 grid rows, requested output<=512, context1024,
AR running4, waiting20, total KV4096 positions. The25th in-flight request returns
503. Invalid input returns400/422; exact post-encode overflow returns400 before
AR. For latency-sensitive use start with client concurrency1.

P4 measured144 native requests with no errors and stable post-warmup memory.
The72 measured requests averaged0.1031rps; E2E p50/p95 were124.88/222.75s under
24-request bursts. These figures include queueing and do not establish a
sustainable2.5rps arrival rate. NVML sampled peak was27.03GiB for that workload
and29.19GiB for the boundary run. See the P4 gate report for full conditions.
4090-24GB failed the FP32 vocoder working set even with AR concurrency1 and
KV1024; no24GB serving profile is qualified. Neither the A800 measurements nor
a short successful4090 request establishes capacity for longer outputs. `--probe-24gb` is an experiment generator, not a supported serving preset.

The frozen parity profile explicitly uses BF16 HF AR with **logits_to_keep=0**.
HF's implicit last-only LM-head optimization uses a different GEMM shape and
can break inactive-audio ties. P5 documents and retains that failed comparison;
it neither exempts those tokens nor modifies P3 expected grids. See P5 protocol
AppendixA for the explicit reference configuration.

## Qualification and developer reproduction

The measurements above use the recorded A800 profile: Python3.10,
torch/torchaudio2.9.1+cu128, transformers5.12.1, SGLang0.5.16 and TorchCodec0.8.1.
The installation guide's newer default CUDA13/torch2.11 stack is a separate
dependency profile; these numbers do not qualify that stack. Environment names
used in experiment receipts are not serving requirements.

The fixed regression set contains eight bilingual rows, four unique source
audios and32 mode requests. All32 AR outputs and16 audio outputs agree with their
independent same-profile references. Chinese T2S CER is38.75%; two S2S responses
reach the512-row budget. No human MOS or calibrated quality/performance threshold
is claimed. Checkpoint/HF-code redistribution terms remain unresolved; weights
are not bundled with the integration.

For reproduction and CI rather than ordinary server launch, see:

- `sglang-omni/docs/design/moss_speech/cli_launch.md`: installed console commands,
  noninteractive startup and four-mode launch verification.
- `sglang-omni/docs/design/moss_speech/p5/02_dependencies_and_rights.md`: exact
  tested dependencies and provenance.
- `sglang-omni/docs/design/moss_speech/p5/05_gate_report.md`: offline quality and
  benchmark/CPU/GPU CI commands.
- `sglang-omni/docs/design/moss_speech/p6/02_reproduction.md`: streaming codec,
  full HTTP regression, independent ASR and aggregation commands.
- `sglang-omni/docs/design/moss_speech/p6/03_gate_report.md`: streaming results,
  measured boundaries and retained diagnostic failures.


The developer GPU CI entry accepts explicit assets and an already generated
independent reference report. Run it in the installed runtime on an allocated
GPU; it launches and stops its own service:

```bash
export MOSS_SPEECH_MODEL_DIR="$PWD/models/MOSS-Speech"
export MOSS_SPEECH_CODEC_DIR="$PWD/models/MOSS-Speech-Codec"
export MOSS_SPEECH_VOICE_WAV="$PWD/voice.wav"
export MOSS_SPEECH_MANIFEST="$PWD/artifacts/p5/seedtts/manifest.json"
export MOSS_SPEECH_REFERENCE="$PWD/artifacts/p5/reference/report.json"
export MOSS_SPEECH_CI_OUTPUT="$PWD/artifacts/ci-new"
bash sglang-omni/scripts/moss_speech/ci/run_gpu.sh
# Use run_streaming_gpu.sh with a fresh output directory for the streaming lane.
```

### Text at the output limit

Text responses decode every generated text-channel token, including an ordinary
last token when `finish_reason="length"`. Special tokens are filtered; trailing
Unicode replacement characters from an incomplete byte sequence are withheld,
matching the streaming terminal. Grid/usage counts retain those tokens.

This integration behavior differs from the locked upstream processor, which
unconditionally removes the last grid row before decoding. New reference exports
retain that upstream result in `text`, and record `service_text` with profile
`moss-full-grid-unicode-prefix-v1`. Grid parity and service-text parity are separate
checks. Frozen P0–P6 reports are unchanged; unversioned reports retain their old
comparison semantics. See the [audit remediation](../design/moss_speech/audit_20260909.md).
