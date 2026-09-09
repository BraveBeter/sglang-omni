# P0-01 Version Lock Record

Date: 2026-09-04 (Asia/Shanghai). Artifact paths are relative to the workspace root.

## 1. Locked sources

| Source | Role | Revision | Ref |
|---|---|---|---|
| `fnlp/MOSS-Speech` (HF) | AR model + remote code + tokenizer | snapshot `cff025bb41d8459d59abac0b5e44aba7f659ec9e` | materialized at `models/MOSS-Speech` |
| `fnlp/MOSS-Speech-Codec` (HF) | codec weights + remote code | snapshot `eeec733e4e1dea7da444d332d8e1621ef257414c` | materialized at `models/MOSS-Speech-Codec` |
| `OpenMOSS/MOSS-Speech` (GitHub) | reference driver (`utils/interface.py`, `gradio_demo.py`), assets, `cosyvoice/` decoder package | branch `feat/docs` @ `1ea408a10d07b9fdc7a27bce19d1211b62067784` ("docs: update logo") — local branch `reference` | cloned at `repos/MOSS-Speech` |
| `OpenMOSS/MOSS-Speech` (GitHub) `main` | pins the Matcha-TTS submodule | `main` @ `3a1001ae37b731ae49a98311139fb54c49230fd9`; submodule `Matcha-TTS` pinned @ `bd4d90d93214b37f7a159cf205ae85762c2c10aa` | — |
| `shivammehta25/Matcha-TTS` (GitHub) | decoder components (`matcha.models.components.*`, `matcha.hifigan.models`) imported by `cosyvoice/` | `bd4d90d93214b37f7a159cf205ae85762c2c10aa` ("Update README.md") | vendored at `repos/MOSS-Speech/Matcha-TTS` (`.git` kept) |

Key facts discovered during locking:

- The GitHub repo `main` branch contains **no runnable code** (README/LICENSE/submodule only). All reference code lives on the **`feat/docs` branch**, which does **not** pin the Matcha-TTS submodule; the submodule pin (`bd4d90d`) exists only on `main`. We lock BOTH: code = `feat/docs@1ea408a`, matcha = `bd4d90d` (from `main`'s gitlink, matching user-verified pin).
- The HF remote processor hard-depends on the GitHub `cosyvoice` package: it resolves class `cosyvoice.flow.flow.CausalMaskedDiffWithXvec` dynamically. The codec HF repo ships flow **weights** (`flow/flow.pt`, `flow-chunk-5.pt`, `flow-chunk-25.pt`, `hift.pt`, `campplus.onnx`) but the flow **code** comes from the GitHub repo. → Reference runs require `PYTHONPATH=repos/MOSS-Speech:repos/MOSS-Speech/Matcha-TTS`.
- Reference code filename bugs (recorded, not fixed in reference): `MIMOInterface.default_decoder_audio_prompt_path = "./assets/prompt_cn.wav"` (underscore; actual file is `prompt-cn.wav`) and gradio default `value=".assets/prompt_cn.wav"` (leading dot). Headless driver passes the real path explicitly.

## 2. Licenses

| Component | License | Note |
|---|---|---|
| GitHub `OpenMOSS/MOSS-Speech` | Apache-2.0 | LICENSE archived at `repos/MOSS-Speech/LICENSE` |
| Matcha-TTS @ bd4d90d | MIT | LICENSE archived at `repos/MOSS-Speech/Matcha-TTS/LICENSE` |
| HF `fnlp/MOSS-Speech` weights | **NONE FOUND** | no LICENSE file, no `license:` field in README card |
| HF `fnlp/MOSS-Speech-Codec` weights | **NONE FOUND** | no LICENSE/README in repo |

> Open item (carried to MODEL_CONTRACT "Known limitations"): weight redistribution license is unspecified on both HF repos. GitHub code is Apache-2.0; we assume weights follow, but this must be confirmed with MOSS team before any upstream release packaging.

## 3. Integrity hashes

- Weight shards + codec flow weights: `artifacts/p0/weights_sha256.txt` (full sha256).
- Remote/driver code + assets: `artifacts/p0/code_sha256.txt`.
- Selected short hashes:

| File | sha256 (first 16) |
|---|---|
| models/MOSS-Speech/configuration_moss_speech.py | 3f2dd9b0fbe18a8a |
| models/MOSS-Speech/modeling_moss_speech.py | 00dc56ec8e3fdccc |
| models/MOSS-Speech/processing_moss_speech.py | 2cbea069bcd0e4ed |
| models/MOSS-Speech/config.json | e86bb40c4aaaba4a |
| models/MOSS-Speech-Codec/modeling_moss_speech_codec.py | 26467789d521ae48 |
| models/MOSS-Speech-Codec/config.json | 7fcfa2e182024287 |
| repos/MOSS-Speech/utils/interface.py | d7fce96bdde413a8 |
| repos/MOSS-Speech/gradio_demo.py | a8e0d1b4f54fbc8c |
| repos/MOSS-Speech/assets/prompt-cn.wav | cde3e98fdf0c90d4 |
| repos/MOSS-Speech/assets/prompt-en.wav | 5c1541599a64decd |

- Model shard sizes: 4.6G + 4.7G + 4.7G + 3.1G ≈ 17.1 GiB bf16 (matches plan §2.1 "≈16.9GiB").
- Codec repo total ≈ 7.5 GiB (includes flow/, hift, campplus.onnx, model.safetensors).

## 4. Offline materialization

- HF cache: the operator-provided `HF_HOME` cache, downloaded via `HF_ENDPOINT=https://hf-mirror.com` (direct huggingface.co unreachable from login node; GitHub reachable via SSH).
- `models/` dirs are dereferenced copies (no symlinks) for offline compute nodes.
- Compute-node runs must set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; optionally set `HF_HOME` to your prepared cache.

## 5. Reproduction commands

```bash
# login node (internet): weights
HF_ENDPOINT=https://hf-mirror.com hf download fnlp/MOSS-Speech
HF_ENDPOINT=https://hf-mirror.com hf download fnlp/MOSS-Speech-Codec
# login node: reference code
git clone git@github.com:OpenMOSS/MOSS-Speech.git repos/MOSS-Speech
git -C repos/MOSS-Speech checkout -b reference origin/feat/docs   # 1ea408a
git clone git@github.com:shivammehta25/Matcha-TTS.git /tmp/matcha
git -C /tmp/matcha checkout bd4d90d93214b37f7a159cf205ae85762c2c10aa
cp -r /tmp/matcha repos/MOSS-Speech/Matcha-TTS
```
