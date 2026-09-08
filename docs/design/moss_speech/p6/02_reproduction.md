# P6 reproduction

Qualification passed on 2026-09-09. Accepted outputs and retained failed probes
are listed in sglang-omni/docs/design/moss_speech/p6/03_gate_report.md. Run from the workspace root with
all inference inside a Slurm GPU allocation. The original P5 requirements,
checkpoint revisions, 512/512 request budgets and offline environment apply.

## Independent codec checks

The reference environment is `.venv-p0`; the native environment is `.venv-omni`.
Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=1`,
`HF_HOME=/remote-home1/xrluan/.cache/huggingface`, and `TMPDIR=/dev/shm`.
Only the reference needs the locked MOSS-Speech and Matcha checkouts on PYTHONPATH.

```bash
export PYTHONPATH="$PWD/repos/MOSS-Speech:$PWD/repos/MOSS-Speech/Matcha-TTS:$PWD/sglang-omni"
.venv-p0/bin/python sglang-omni/scripts/moss_speech/p6/reference_codec.py \
  --codec-path models/MOSS-Speech-Codec \
  --state artifacts/p5/ci_3883/en_00_s2s.pt \
  --out-dir artifacts/p6/codec-reference-new

export PYTHONPATH="$PWD/sglang-omni"
.venv-omni/bin/python sglang-omni/scripts/moss_speech/p6/compare_codec.py \
  --codec-path models/MOSS-Speech-Codec \
  --state artifacts/p5/ci_3883/en_00_s2s.pt \
  --reference-dir artifacts/p6/codec-reference-new \
  --out-dir artifacts/p6/codec-native-new
```

Both chunk5 and chunk25 use their actual trained weights. Qualification of the
component with chunk25 does not qualify a chunk25 HTTP deployment. The
`chunk-cudnn-deterministic-v1` correction and retained failures are explained in
`sglang-omni/docs/design/moss_speech/p6/01_streaming_contract.md`.

## Native SSE and offline regression

Use the environment variables shown in the cookbook's P5 GPU CI example, with
the existing frozen reference report and a fresh output directory. Each command
launches and stops its own service; do not run an ASR model in that service's GPU
allocation at the same time.

```bash
export MOSS_SPEECH_REFERENCE="$PWD/artifacts/p5/reference_3880/report.json"
export MOSS_SPEECH_CI_OUTPUT="$PWD/artifacts/p6/http-new"
PATH="$PWD/.venv-omni/bin:$PATH" \
  bash sglang-omni/scripts/moss_speech/ci/run_streaming_gpu.sh

export MOSS_SPEECH_CI_OUTPUT="$PWD/artifacts/p6/offline-new"
PATH="$PWD/.venv-omni/bin:$PATH" \
  bash sglang-omni/scripts/moss_speech/ci/run_gpu.sh
```

The streaming entry runs the fixed32 cases, a separate WAV transport request,
length caps, explicit preflight errors, disconnects before/after first PCM and
after AR completion while the terminal is pending, a controlled failure after
a real backend delta, and repeated two-request waves. Optional `--smoke` limits
the canonical set to eight bilingual cases; it does not establish the32-case
gate. The controlled failure tests SSE/iterator cleanup; CPU codec-fault tests
separately test actual decoder-exception ownership. The terminal-finalization
race is covered by CPU scheduler tests and the real pending-terminal disconnect.
No artificial GPU OOM is required.

The first-PCM gate checks both actual stage events and a conservative client
receipt bound before `model_path_end`, and TTFA
starts at HTTP request initiation. Header/role events do not count. PCM16
little-endian chunks have no headers; WAV chunks are separate containers that
must each be decoded before joining their samples. Exact AR grids, stop reason,
usage, continuous sample counts and reference PCM are functional gates. RTF,
seam amplitude and latency are descriptive metrics without calibrated thresholds.

## Full audio reference and independent ASR

Export all16 audio cases independently from HF grids and the previously frozen
voice conditions. This does not read native generated grids as expected output.

```bash
export PYTHONPATH="$PWD/repos/MOSS-Speech:$PWD/repos/MOSS-Speech/Matcha-TTS:$PWD/sglang-omni"
.venv-p0/bin/python sglang-omni/scripts/moss_speech/p6/reference_audio.py \
  --manifest artifacts/p5/seedtts/manifest.json \
  --reference artifacts/p5/reference_3880/report.json \
  --voice-dir artifacts/p5/ci_3883 --codec-path models/MOSS-Speech-Codec \
  --out-dir artifacts/p6/audio-reference-new

export PYTHONPATH="$PWD/sglang-omni"
.venv-omni/bin/python sglang-omni/scripts/moss_speech/p6/prepare_quality.py \
  --manifest artifacts/p5/seedtts/manifest.json \
  --reference artifacts/p5/reference_3880/report.json \
  --native-dir artifacts/p6/http-new \
  --audio-reference artifacts/p6/audio-reference-new \
  --voice-dir artifacts/p5/ci_3883 --out-dir artifacts/p6/quality-input-new

.venv-omni/bin/python sglang-omni/scripts/moss_speech/p5/evaluate.py \
  --manifest artifacts/p5/seedtts/manifest.json \
  --reference artifacts/p5/reference_3880/report.json \
  --native-dir artifacts/p6/quality-input-new \
  --asr-checkpoint models/eval/small.pt --out-dir artifacts/p6/evaluation-new
```

The quality staging report records the full parent report SHA and extra request
count; all32 canonical IDs must be present exactly once. Its16 WAV pairs are
same-profile streaming reference/native audio. The HF report's timing still
measures HF encode/AR, excluding codec; do not present it as streaming-reference
TTFA. The unchanged P5 evaluator preserves language normalization, denominator,
ASR version, truncation and threshold semantics. Compare P6 with the archived
P5 quality report descriptively; different codec profiles need not share WAVs.

The CPU workflow uses the existing independent CPU-only environment, adds the
P6 suites, and does not require SGLang, CUDA or weights. Local GPU evidence is
not a claim that hosted GPU CI or performance/quality calibration has occurred.


## Long-stream allocator and final aggregation

Run the memory probe inside an isolated Slurm GPU allocation after creating the
full independent audio reference above. The CUDA allocator environment must be
set before Python imports torch. It warms up once and compares two 512-code
requests per chunk, measures actual expandable segments, and verifies live
memory returns to the warm baseline. Avoid long runtime paths: the generated
stage Unix socket must fit the platform's 107-byte path limit.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-omni/bin/python sglang-omni/scripts/moss_speech/p6/probe_memory.py \
  --codec-path models/MOSS-Speech-Codec \
  --state artifacts/p5/ci_3883/zh_00_s2s.pt \
  --reference artifacts/p6/audio-reference-new/zh_00_s2s.waves.pt \
  --out-dir artifacts/p6/memory-new
```

The following CPU postprocessor verifies the archived accepted jobs. The source
amendment argument acknowledges only the recorded package capability declaration
changed after3903; all inference/config/HTTP/lifecycle source hashes remain exact.
Omit this argument for a new GPU run produced from the current source. It does
not waive numerical, lifecycle or quality evidence. Raw reports must remain at
the paths below (they are local artifacts, not included in the Git delivery).

```bash
PYTHONPATH="$PWD/sglang-omni" .venv-omni/bin/python \
  sglang-omni/scripts/moss_speech/p6/summarize.py \
  --manifest artifacts/p5/seedtts/manifest.json \
  --reference artifacts/p5/reference_3880/report.json \
  --native artifacts/p6/ci_3903/report.json \
  --comparison artifacts/p6/quality_input_3904/comparison.json \
  --evaluation artifacts/p6/evaluate_3904/report.json \
  --offline artifacts/p6/o_3900/report.json \
  --baseline artifacts/p5/ci_3883/report.json \
  --codec artifacts/p6/codec_compare_3891/report.json \
  --memory artifacts/p6/memory_3902/report.json \
  --source-amendments sglang-omni/docs/design/moss_speech/p6/source_amendments.json \
  --out artifacts/p6/rechecked-summary.json
```
