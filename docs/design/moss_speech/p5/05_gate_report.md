# P5 delivery gate report (2026-09-08)

P5 engineering qualification is complete. The supported deployment remains the
P4 A80080GB profile; real4090 testing rejects24GB support for the tested FP32
codec workload. All32 fixed reference/native requests and4001 complete grid rows
match, with16 byte-identical WAV pairs. Quality measurements are reported without
claiming a quality SLA. P6 has not started.

| Gate | Result and evidence |
|---|---|
| G1 quality evidence | Pass: reference3880, native3883 and independent ASR3884; all cases accounted for, including two length stops; source/audio/report hashes verified |
| G2 hardware outcome | Pass as an explicit unsupported24GB outcome:4090 job3879 completed25 requests and failed7 with FP32 vocoder OOM; all four children exited; no supported24GB YAML published |
| G3 engineering | Pass: fresh asset-free CPU89, full model/lifecycle/local-golden156, common HTTP111; final local A800 CI entry3883 passes and records54 source hashes; static checks pass |
| G4 documentation/provenance | Pass for integration code and evidence delivery: cookbook, support table, reproduction, source/license audit and SHA index; unresolved external rights remain release constraints |

**Upstream/weight redistribution readiness remains false.** No weights, source
audio, dataset annotations or raw output transcripts are included in this
integration delivery. Code and compact evidence are committed to the existing
feature branch for the already authorized fork push. Hosted GitHub CI has not
been observed running; only the local equivalent CPU entry and Slurm GPU entry
were executed. Quality/performance thresholds remain disabled and uncalibrated.
The repository's newer default CUDA13 stack has not been qualified here.

## Delivered implementation

The only production-code change in P5 isolates the native request dataclass in
sglang-omni/sglang_omni/models/moss_speech/request_data.py. HTTP contract imports
no longer load SGLang; the historical type import in request_builders remains
available lazily. Its fields, initialization and reset behavior are unchanged.
The full156-test suite and native3883 cover this move. No shared framework code,
model arithmetic, P0–P4 expected assets, sampler or numerical tolerance changed.

New tools prepare pinned assets, render explicit local configs, generate an HF
reference, run native HTTP, score with independent Whisper and aggregate evidence.
The CI preset uses the chat endpoint and refuses enabled thresholds without
calibration metadata. The CPU workflow requires neither SGLang nor checkpoints.
GPU assets are explicit; absent assets fail instead of skipping the lane.

Source/profile issue: P5's first reference3871 exposed HF's implicit
logits_to_keep=1, while the frozen P3 observer exercised full-head0. Failed3872
is retained. All40 hidden layers/final norms matched; three inactive-audio step0
rows diverged because of the head GEMM shape. Explicit0 diagnostics3878 restored
raw equality; final3880/3883 preserve P3's profile and strict full-grid gate.
Protocol AppendixA documents this versioned correction; no tie exemption applies.

## Measured scope and limitations

- 32 requests from8 source rows,4 unique source recordings,24 unique request bodies.
- Reference/native agreement is exact; Chinese T2S CER is31/80 (38.75%) in both.
- Two Chinese S2S requests hit512 rows; they share an input and remain in metrics.
- All16 native WAVs are finite, nonempty, mono24kHz; no human MOS was performed.
- Native serial E2E p50: T2T1.096s, T2S5.132s, S2T1.133s, S2S28.287s.
- T2S/S2S HTTP RTF p50:1.236/0.847. These are not streaming latency measures.
- Native serving NVML sampled peak27.645GiB. P4 boundary peak29.188GiB remains the
  more demanding measured baseline; this phase does not reduce its capacity limit.
- 4090 AR1/KV1024 could start only after an experimental static-budget adjustment;
  FP32 vocoder later exhausted physical memory. Seven failures out of32 prevent
  publishing a24GB preset. No low-precision/offload change was silently applied.

Details: sglang-omni/docs/design/moss_speech/p5/04_quality.md and
sglang-omni/docs/design/moss_speech/p5/03_hardware.md. Supported capacity remains
prompt512/new512/context1024, AR running4/waiting20, KV4096, TP1, BF16 AR/KV,
FP32 codec, eager torch_native. Radix, graph, compile, quantization, uploaded
voices and streaming are disabled. Larger contexts and other hardware require
independent qualification. Numeric lengths count dual-channel grid positions.

## Reproduce

Run from the workspace root containing sglang-omni/. Follow the environment and
asset instructions in sglang-omni/docs/cookbook/moss_speech.md and
sglang-omni/docs/design/moss_speech/p5/02_dependencies_and_rights.md first.
The local package receipt is not a fresh-resolvable GPU lock. Reference uses the
locked MOSS-Speech checkout1ea408a and its Matcha subtree with transformers4.57.1;
native uses the verified torch2.9.1+cu128 / transformers5.12.1 / SGLang0.5.16 stack.
Prepare SeedTTS and the official Whisper checkpoint online. Each command below
that runs a model must execute inside a Slurm GPU allocation, offline. Use new
output directories; drivers refuse to overwrite existing experiment directories.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/remote-home1/xrluan/.cache/huggingface
export TMPDIR=/dev/shm OMP_NUM_THREADS=1
PYTHONPATH="$PWD/repos/MOSS-Speech:$PWD/repos/MOSS-Speech/Matcha-TTS:$PWD/sglang-omni/scripts/moss_speech/p0:$PWD/sglang-omni" \
  .venv-p0/bin/python sglang-omni/scripts/moss_speech/p5/reference.py \
  --manifest artifacts/p5/seedtts/manifest.json \
  --model-path models/MOSS-Speech --codec-path models/MOSS-Speech-Codec \
  --out-dir artifacts/p5/reference

export MOSS_SPEECH_MODEL_DIR="$PWD/models/MOSS-Speech"
export MOSS_SPEECH_CODEC_DIR="$PWD/models/MOSS-Speech-Codec"
export MOSS_SPEECH_VOICE_WAV="$PWD/repos/MOSS-Speech/assets/prompt-cn.wav"
export MOSS_SPEECH_MANIFEST="$PWD/artifacts/p5/seedtts/manifest.json"
export MOSS_SPEECH_REFERENCE="$PWD/artifacts/p5/reference/report.json"
export MOSS_SPEECH_CI_OUTPUT="$PWD/artifacts/p5/ci-new"
PATH="$PWD/.venv-omni/bin:$PATH" \
  bash sglang-omni/scripts/moss_speech/ci/run_gpu.sh

# Run only after the native driver has exited and released its GPU processes.
PYTHONPATH="$PWD/sglang-omni" .venv-omni/bin/python \
  sglang-omni/scripts/moss_speech/p5/evaluate.py \
  --manifest artifacts/p5/seedtts/manifest.json \
  --reference artifacts/p5/reference/report.json --native-dir artifacts/p5/ci-new \
  --asr-checkpoint models/eval/small.pt --out-dir artifacts/p5/evaluation-new
```

The locally executed cluster wrappers are scripts/p5_reference.sbatch,
scripts/p5_ci.sbatch and scripts/p5_evaluate.sbatch. They are workspace-local
cluster configuration; the portable Python/shell drivers above are in Git.
Final job IDs are3880/3883/3884. Earlier corrected3881/3882 also passed, but3881
started before embedded source fingerprints were added; final3883 is authoritative.

CPU entry: sglang-omni/scripts/moss_speech/ci/run_cpu.sh,89 passed in the fresh
CPU-only environment. Full local regression command (156 passed):

```bash
TMPDIR=/dev/shm OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MOSS_SPEECH_MODEL_DIR="$PWD/models/MOSS-Speech" \
MOSS_SPEECH_FIXTURES_DIR="$PWD/sglang-omni/tests/fixtures/moss_speech" \
MOSS_P1_ALIGNMENT_DIR="$PWD/artifacts/p1/alignment" \
  .venv-omni/bin/python -m pytest sglang-omni/tests/unit_test/moss_speech \
  sglang-omni/tests/unit_test/serve/test_chat_lifecycle.py \
  sglang-omni/tests/unit_test/serve/test_explicit_generation_params.py -q
```

The156 suite overlaps the89 suite; counts are separate executions, not245 unique
tests. The common HTTP111-test suite separately passes after the TorchCodec0.8.1
compatibility repair. This does not add M4A to MOSS's soundfile input contract.

## Evidence and retained failures

Compact, text-free index: sglang-omni/docs/design/moss_speech/p5/gate_summary.json.
Raw reports/WAVs/states remain under artifacts/p5/; SHA256 permits verification
without copying model assets or dataset text into Git. The aggregate checks
raw grids, coverage, report linkage, WAV bytes, CPU no-skip status and the source
fingerprints. It was challenged with failed/missing/grid/source/metric/WAV/skip
mutations: seven invalid cases rejected plus one valid case accepted.

Final source and static evidence are included in the index. Summarize.py was
added after3883 started and is covered by the separate release-source inventory
and aggregation validation, not falsely included in its54 startup hashes.
Slurm accounting records all jobs3870–3884, including the cancelled dependent
ASR3874. Final generator/scorer jobs exit0; native3883 records four actual child
PIDs, all exit0, no pending completion futures, and no active Slurm jobs remain.

Retained negative evidence: baseline4090 startup3870, mistaken placement-only
budget3873, corrected0.80 but insufficient budget3875, real vocoder OOM3879,
implicit-profile reference3871/failed native3872, diagnostic3876–3878, and early
CPU collection/ABI/config assertion failures. Corrections do not erase these
runs or turn cross-A800/4090 differences into same-hardware parity claims.

P5 delivery does not resolve checkpoint/HF-code redistribution terms, certify
full SeedTTS scores, calibrate hosted quality/performance thresholds, or launch
P6. Future work can use these measurements to scope streaming, performance,
quality investigation and any24GB redesign explicitly.
