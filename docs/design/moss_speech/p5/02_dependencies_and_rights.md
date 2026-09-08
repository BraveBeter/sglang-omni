# P5 dependency and provenance audit

Audit date: 2026-09-08. This file records observed metadata and compatibility;
it does not supply missing permissions or extend a code license to weights.

## Checkpoints and vendored sources

| Source | Observed revision / declaration | Delivery consequence |
|---|---|---|
| [MOSS-Speech](https://huggingface.co/OpenMOSS-Team/MOSS-Speech) | cff025bb41d8459d59abac0b5e44aba7f659ec9e; no license card field or LICENSE file | Same as P0; checkpoint redistribution terms remain unresolved |
| [MOSS-Speech-Codec](https://huggingface.co/OpenMOSS-Team/MOSS-Speech-Codec) | eeec733e4e1dea7da444d332d8e1621ef257414c; no license card field or LICENSE file | Same as P0; do not bundle codec weights |
| [MOSS-Speech code](https://github.com/OpenMOSS/MOSS-Speech#license) | README explicitly describes the repository's code as Apache-2.0; locked reference feat/docs@1ea408a | Does not explicitly license the separate HF checkpoints |
| CosyVoice / Matcha subset | Existing vendored Apache-2.0 / MIT notices | Keep source hashes, notices and modifications with the code |
| HF codec Python subset | No license in the locked HF snapshot; existing LICENSE.note records provenance | Confirm applicability of upstream code terms before asserting upstream redistribution readiness |
| SeedTTS Arrow | Pinned 27f4c1adee83b5b29b7c4b375f6b976324bda308; external dataset | Stage locally; do not add source audio or dataset text to this repository |

API receipts: `artifacts/p5/audit/ar.json`, `artifacts/p5/audit/codec.json`,
`artifacts/p5/audit/seedtts.json`. The old fnlp model URLs now resolve to
OpenMOSS-Team, while the checkpoint SHAs are unchanged. No weights are updated.
The small quality subset remains external and is referenced by manifest hashes.

**Upstream/redistribution readiness is false until the unresolved checkpoint
and HF-code terms are confirmed by their rights holders.** This phase may
publish integration code/measurement records to the already authorized fork;
it does not package checkpoints, contact maintainers, or claim permissions.
P0's historical assumption that code licensing also covers weights is not a
release decision.

## Runtime compatibility

The verified GPU environment is Python3.10.12, torch/torchaudio2.9.1+cu128,
transformers5.12.1, SGLang0.5.16. AR/KV are BF16, codec FP32. Package/version
receipt: `artifacts/p5/environment.json`. This is the existing compatibility
profile used by P1–P4; the repository's newer default torch2.11/torchaudio2.11/
TorchCodec0.11.1/CUDA13 stack has not been qualified for MOSS-Speech here.
Installing the repository's default dependencies is not evidence of parity on
that different stack. Re-run the functional and numerical gates before claiming
another GPU environment is supported.

MOSS codec's actual dependency closure includes **diffusers0.37.0** (vendored
Matcha activation/attention/LoRA imports), in addition to torch, torchaudio,
transformers, numpy, scipy, librosa, einops, onnxruntime, soundfile and
safetensors. The existing `moss-speech` optional extra supplies its model extras;
diffusers already appears in the base project dependencies. The P1 prose
closure omitted diffusers; the fresh P5 CPU install exposed and corrected that
inventory omission without changing the decoder implementation.

### Repair of the existing M4A environment failure

Installed TorchCodec0.10.0+cu128 was incompatible with this torch2.9.1 stack.
The [official TorchCodec0.8 compatibility table](https://raw.githubusercontent.com/meta-pytorch/torchcodec/v0.8.1/README.md)
pairs0.8 with torch2.9. A separate target-directory installation of0.8.1 first
passed the exact failing M4A test; the same CPU decoding wheel then replaced the
incompatible local package, without changing torch or shared HTTP code:

```bash
TMPDIR=/dev/shm uv pip install --python .venv-omni/bin/python --no-deps torchcodec==0.8.1
TMPDIR=/dev/shm .venv-omni/bin/python -m pytest \
  sglang-omni/tests/unit_test/serve/test_openai_api.py -q
```

Result:111 passed,0 skipped. Evidence: `artifacts/p5/torchcodec_probe.log`,
`artifacts/p5/torchcodec_environment_install.log`, `artifacts/p5/cpu_public_http.log`.
P4's original failure report is retained. This repairs the common endpoint's
M4A test; MOSS's own soundfile-based input decoder still accepts only formats it
can decode and may reject M4A. Do not turn this environment fix into a new
MOSS input-format capability claim.

## Fresh CPU CI environment

Unlike the GPU runtime receipt, this lane was installed into a new environment
and runs without SGLang or model checkpoints. The entry point is
`sglang-omni/scripts/moss_speech/ci/run_cpu.sh`; dependencies are
`sglang-omni/scripts/moss_speech/ci/requirements-cpu.txt`.

```bash
uv venv --python python3.10 .venv-p5-cpu
uv pip install --python .venv-p5-cpu/bin/python torch==2.9.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-p5-cpu/bin/python \
  -r sglang-omni/scripts/moss_speech/ci/requirements-cpu.txt
PATH="$PWD/.venv-p5-cpu/bin:$PATH" TMPDIR=/dev/shm \
  bash sglang-omni/scripts/moss_speech/ci/run_cpu.sh
```

Final release runs:89 passed,0 skipped in the fresh environment
(artifacts/p5/cpu_ci_release.log). Full model/lifecycle/local-golden regression
separately passes156 tests (artifacts/p5/cpu_release.log). Earlier88/155 runs
remain archived before the added engine-budget regression case.
The CPU lane covers chat normalization/errors, lifecycle, codec CPU contracts,
parity metric failure detection, fixed data integrity and CI policy. It does not
claim native GPU execution or replace the full SGLang-layer/golden suite.
The native dataclass is now lazily imported, with its historical import path
preserved, so HTTP validation does not import the GPU backend.

GPU CI uses `sglang-omni/scripts/moss_speech/ci/run_gpu.sh` with explicit local
model/codec/voice/manifest/reference/output variables; missing assets fail
immediately. The same preset owns timeouts and chat endpoint. Quality/performance
thresholds stay disabled until real hosted CI hardware is independently
calibrated. Local Slurm A800 execution and hosted CI execution are separate facts.
