# P0-02 Dependency Inventory & Reference Environment

Date: 2026-09-04. Decision record for T0.3.

## 1. Import-chain findings (static analysis + runtime verification)

| Import | Required by | Status |
|---|---|---|
| `gradio` | `utils/interface.py` **top-level** + monkey-patch `gr.processing_utils._check_allowed` (line 1–2); `gr.Warning/Error/Info` used in UI paths | installed in reference env (D5 = install, zero porting risk) |
| `matcha.models.components.{flow_matching,decoder,transformer}`, `matcha.hifigan.models` | `cosyvoice/flow/*`, `cosyvoice/hifigan/hifigan.py` | vendored Matcha-TTS @ bd4d90d on PYTHONPATH |
| `cosyvoice.flow.flow.CausalMaskedDiffWithXvec` | HF processor remote code (dynamic class lookup) | GitHub `feat/docs` repo on PYTHONPATH |
| `conformer`, `diffusers`, `einops`, `lightning`, `omegaconf`, `hydra`, `hyperpyyaml` | matcha / cosyvoice stack | installed |
| `soundfile`, `librosa`, `torchaudio`, `onnxruntime` (campplus) | interface / codec | installed |
| `gdown`, `matplotlib`, `wget`, `pyarrow`, `pyworld` | transitive matcha imports | installed |
| `pkg_resources` | matcha stack (via setuptools) | requires `setuptools<81` (81+ dropped pkg_resources) |

Blocker resolution (per review B2/B3): both were closed inside T0.3 — matcha vendored & importable; gradio installed. Verified: `import matcha, cosyvoice` OK.

## 2. transformers compatibility (decision D2)

- `.venv-omni` has transformers **5.12.1**: `AutoConfig` loads, but `modeling_moss_speech.py` fails — `cannot import name 'isin_mps_friendly' from 'transformers.pytorch_utils'` (removed in v5). Remote code is 4.x-era.
- **Decision: option B** — dedicated reference venv `.venv-p0` (Python 3.10.12) with **transformers 4.57.1** (same version as the cluster's other validated env), torch 2.9.1+cu128, torchaudio 2.9.1. `.venv-omni` remains untouched for sglang-omni work.
- P3 parity fixtures will be exported from `.venv-p0` (reference side).

## 3. Environment snapshots

- Full freeze: `artifacts/p0/venv-p0-freeze.txt` (143 packages).
- Key pins: `torch==2.9.1+cu128`, `torchaudio==2.9.1`, `transformers==4.57.1`, `accelerate`, `gradio 5.x`, `setuptools<81`.

## 4. Runtime requirements

- Reference runs need CUDA (processor loads flow weights to `self.device` via `torch.load`; login node fails with `torch.cuda.is_available() is False`) — all reference execution happens on Slurm compute nodes.
- Reference run recipe:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/remote-home1/xrluan/.cache/huggingface
export PYTHONPATH=/remote-home1/xrluan/SGLang_experiments/repos/MOSS-Speech:/remote-home1/xrluan/SGLang_experiments/repos/MOSS-Speech/Matcha-TTS
.venv-p0/bin/python <driver> --model-path models/MOSS-Speech --codec-path models/MOSS-Speech-Codec ...
```

## 5. Residual risks (non-blocking for P0)

- Matcha-TTS brings a wide transitive surface (lightning, matplotlib, pyworld); for the future sglang-omni `components/` port we will vendor only the used modules (`models/components/*`, `hifigan/models.py`) and re-audit imports (P1 adapter work).
- License of HF weight repos unspecified (see 01_version_lock.md §2).
