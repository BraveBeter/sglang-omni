# MOSS-Speech

MOSS-Speech provides non-streaming text/speech input and text/speech output through
`POST /v1/chat/completions`. It is a conversation model; speech transcription and
verbatim reading are prompting tasks, without a dedicated ASR/TTS quality claim.
The measured baseline is one A80080GB, TP1, BF16 native AR/KV, FP32 codec, eager
PyTorch attention. Streaming, uploaded voices, radix caching, CUDA Graph, compile,
quantization and TP>1 are disabled. The pipeline uses one inline input encoder and
separate text/audio terminals. Codec execution concurrency is1.

## Environment and assets

Commands below run from the workspace root containing the `sglang-omni/` checkout.
Use a dedicated Python3.10 environment. The verified GPU profile is torch and
torchaudio2.9.1+cu128, transformers5.12.1, SGLang0.5.16 and TorchCodec0.8.1.
The existing `moss-speech` optional extra supplies the codec extras; diffusers0.37.0
is also required by the vendored Matcha components. Installing the repository's
newer torch2.11/CUDA13 default stack is a different, unqualified profile; do not
upgrade a frozen parity environment as part of launching the server.

The package/version receipt, CPU installation instructions and compatibility
repair are documented in
`sglang-omni/docs/design/moss_speech/p5/02_dependencies_and_rights.md`.
Run `uv pip install --python .venv-omni/bin/python --no-deps -e sglang-omni`
only after preparing that GPU dependency environment; `--no-deps` does not install
its dependencies or certify an arbitrary existing environment.

Stage assets online before entering a GPU allocation:

- AR/tokenizer: `fnlp/MOSS-Speech` at cff025bb41d8459d59abac0b5e44aba7f659ec9e.
- Codec: `fnlp/MOSS-Speech-Codec` at eeec733e4e1dea7da444d332d8e1621ef257414c,
  including `flow/flow.pt`, `flow/hift.pt`, `flow/campplus.onnx` and config files.
- A configured default voice WAV. Qualification uses
  `repos/MOSS-Speech/assets/prompt-cn.wav` from the locked reference checkout.

The old fnlp URLs resolve to OpenMOSS-Team. No `trust_remote_code` executes in the
native serving path. Keep checkpoints local and do not bundle their weights with
an integration release: checkpoint and HF-code redistribution terms remain
unresolved in the provenance audit. The HF reference exporter alone needs the
locked remote-code files and the separate `.venv-p0` environment.

## Render and launch

Avoid copying the workspace-specific paths from a checked-in example. Generate
an explicit config using your local paths:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/remote-home1/xrluan/.cache/huggingface
export TMPDIR=/dev/shm OMP_NUM_THREADS=1
.venv-omni/bin/python sglang-omni/scripts/moss_speech/p5/make_config.py \
  --model-path models/MOSS-Speech --codec-path models/MOSS-Speech-Codec \
  --voice-wav repos/MOSS-Speech/assets/prompt-cn.wav \
  --runtime-dir artifacts/moss-runtime --output artifacts/moss.yaml
# Run the serving command inside a Slurm GPU allocation.
.venv-omni/bin/python -m sglang_omni.cli serve \
  --config artifacts/moss.yaml --host 127.0.0.1 --port 8000
```

All inference requires an allocated GPU. `/health` reports readiness; use SIGTERM
or Ctrl-C for complete child-process shutdown. The config generator emits full
stage factory arguments because compact `stage_overrides` currently accepts only
runtime overrides. Model, codec and voice paths are independently configurable.

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

## Quality benchmark and CI

Prepare the pinned bilingual sample and independent ASR checkpoint on a node
with network access:

```bash
mkdir -p artifacts/download-tmp
PYTHONPATH="$PWD/sglang-omni" TMPDIR="$PWD/artifacts/download-tmp" \
  .venv-omni/bin/python sglang-omni/scripts/moss_speech/p5/prepare_assets.py \
  --out-dir artifacts/p5/seedtts --asr-dir models/eval
```

The default subset is the first four rows
per language, including repeated source recordings:8 rows,4 unique source audios,
32 mode requests. It is a small regression sample, not a full SeedTTS result.
The independent reference exporter uses `.venv-p0` with the locked MOSS-Speech
and Matcha checkouts on PYTHONPATH. Run reference generation, native generation,
and independent Whisper scoring in separate GPU stages; command templates are
in `sglang-omni/scripts/moss_speech/p5/` and the final gate report.

```bash
# GPU-free; can use the separately installed CPU requirements.
PATH="$PWD/.venv-p5-cpu/bin:$PATH" \
  bash sglang-omni/scripts/moss_speech/ci/run_cpu.sh

# In an allocated GPU job, with the independent reference already produced:
export MOSS_SPEECH_MODEL_DIR="$PWD/models/MOSS-Speech"
export MOSS_SPEECH_CODEC_DIR="$PWD/models/MOSS-Speech-Codec"
export MOSS_SPEECH_VOICE_WAV="$PWD/repos/MOSS-Speech/assets/prompt-cn.wav"
export MOSS_SPEECH_MANIFEST="$PWD/artifacts/p5/seedtts/manifest.json"
export MOSS_SPEECH_REFERENCE="$PWD/artifacts/p5/reference/report.json"
export MOSS_SPEECH_CI_OUTPUT="$PWD/artifacts/p5/ci-new"
PATH="$PWD/.venv-omni/bin:$PATH" \
  bash sglang-omni/scripts/moss_speech/ci/run_gpu.sh
```

Missing assets fail immediately. The workflow runs CPU contracts; the GPU entry
is explicitly operated on staged assets. Local A800 passes do not claim hosted
CI calibration. Quality/performance thresholds remain disabled, without invented
placeholder numbers; functional, full-grid and finite/nonempty waveform gates
remain mandatory. Report English WER/Chinese CER, reference differences and
truncations; human MOS is unassessed.

Final measured quality, including Chinese T2S CER38.75% and two truncated S2S
requests, is in `sglang-omni/docs/design/moss_speech/p5/04_quality.md`.
Reference and native agree exactly on these cases; agreement does not establish
a dedicated transcription/reading quality guarantee. Full reproduction and
evidence: `sglang-omni/docs/design/moss_speech/p5/05_gate_report.md`.
