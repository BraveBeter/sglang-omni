# MOSS-Speech non-streaming chat deployment

Run commands from the workspace root. The model and codec directories must be
materialized locally. Use the checked-in A800 configuration; it uses native
SGLang AR (BF16), FP32 codec, TP=1, torch_native attention and eager execution.
All inference runs inside a Slurm GPU allocation.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/remote-home1/xrluan/.cache/huggingface
export TMPDIR=/dev/shm OMP_NUM_THREADS=1
.venv-omni/bin/python -m sglang_omni.cli serve \
  --config sglang-omni/examples/configs/moss_speech.yaml \
  --host 127.0.0.1 --port 8000
```

The YAML includes local checkpoint and default-voice paths; change those paths
for a different workspace. `/health` indicates readiness. Stop with SIGTERM or
Ctrl-C so the launcher shuts down its stage processes.

## Request and response

```python
import base64
from pathlib import Path
import httpx

messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer the user's questions with text."},
    {"role": "user", "content": "Introduce yourself in one sentence."},
]
body = {
    "model": "moss-speech", "messages": messages,
    "modalities": ["text"], "stream": False,
    "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    "repetition_penalty": 1.1, "seed": 0, "max_tokens": 200,
}
with httpx.Client(timeout=600) as client:
    response = client.post("http://127.0.0.1:8000/v1/chat/completions", json=body)
    response.raise_for_status()
    print(response.json()["choices"][0]["message"]["content"])
```

For speech input, replace the user content with an `input_audio` part:

```python
messages[1]["content"] = [{
    "type": "input_audio",
    "input_audio": {
        "data": base64.b64encode(Path("input.wav").read_bytes()).decode(),
        "format": "wav",
    },
}]
```

For speech output, set `modalities=["audio"]` and the system text to
`You are a helpful voice assistant. Answer the user's questions with spoken responses.`
Read `choices[0].message.audio.data` and base64-decode it to a WAV file. Output
is mono 24kHz. Use these substitutions independently for the four modes.
The voice is the configured default; request-level voice selection is disabled.

`audios[]` can alternatively bind in order to input_audio parts with empty data;
there must be exactly one source per audio turn. Do not also provide inline data.
WAV is the exercised interchange format. The CPU soundfile decoder determines
which other formats can actually be decoded; unsupported files return 400.
HTTP URLs are rejected by this V1 model; there is no remote download path.

Missing sampling fields use model defaults 0.6/0.95/20/1.1. Explicit 0/1 values
are preserved. `max_tokens` and `max_completion_tokens` are aliases; unequal
simultaneous values are rejected. Omitted length defaults to 200. Explicit seed
is independent of request ID; otherwise the ID deterministically supplies it.
`finish_reason=length` means the output reached its grid budget and may contain
an unfinished utterance; it is not a complete speech guarantee. Usage counts
one dual-channel grid row as one token.

## Bounded eager deployment

The formal configuration caps prompt length at 512 grid rows, requested new
rows at 512 and native context at 1024. Prefix/template and encoded audio rows
count toward the prompt limit. Four native AR requests may run; another 20
requests may wait across the whole pipeline. The 25th in-flight request returns
503; use bounded client concurrency and retry after existing requests complete.
The total KV pool is 4096 positions (4 × 1024), plus the allocator padding slot.
Preprocessing and vocoder execute one request at a time; encoder internal batch
size is 4. These are separate from HTTP arrival concurrency.

A previous 6K prompt with the uncapped 37.5GiB KV reservation OOMed on A800.
The current conservative envelope is tested separately; longer context is an
independent expansion task, not guaranteed by the checkpoint's 10K/40K metadata.
P3 full-grid parity is unchanged. This phase does not qualify 24GB GPUs,
streaming, uploaded voices, CUDA graphs, radix caching, quantization or TP>1.

Expected errors: invalid/unsupported chat or post-encode oversized grid → 400;
malformed protocol fields may return 422; in-flight cap → 503; internal failures
→ 500. A non-streaming HTTP disconnect aborts outstanding pipeline work.

## Reproduction and evidence

- `scripts/p4_http.sbatch`: real TCP four-mode/reference and lifecycle checks.
- `scripts/p4_http.sbatch --workload`: native P1-input-composition workload.
- `scripts/p4_cli.sbatch`: the documented CLI, both terminals and shutdown.
- `sglang-omni/scripts/moss_speech/p4/`: portable drivers used by those launchers.

The gate report records final run IDs, source/evidence hashes and measured
memory/latency. P1 synthetic decoder traffic is not a throughput baseline for
P4 actual autoregressive generation.
