# Vendored Sources & Diff Manifest (components/)

All vendored code serves the MOSS-Speech codec inference closure only. Each
subtree keeps its upstream license; numerics of kept methods are unchanged
unless a deviation is listed. Locked sources:

| Source | Revision |
|---|---|
| OpenMOSS-Team/MOSS-Speech (model, tokenizer, HF configuration/processor) | `cff025bb41d8459d59abac0b5e44aba7f659ec9e` |
| OpenMOSS-Team/MOSS-Speech-Codec (HF codec code and weights) | `eeec733e4e1dea7da444d332d8e1621ef257414c` |
| OpenMOSS/MOSS-Speech (GitHub reference driver and CosyVoice subset) | `1ea408a10d07b9fdc7a27bce19d1211b62067784` |
| shivammehta25/Matcha-TTS | `bd4d90d93214b37f7a159cf205ae85762c2c10aa` |

The locked HF snapshots do not contain a license file. The GitHub source's
Apache-2.0 declaration does not establish terms for the separate checkpoints
or automatically resolve the HF codec source terms. Applicability remains
unconfirmed; do not infer redistribution permission. No weights are bundled.
Preserve the Apache-2.0 / MIT notices with the corresponding vendored sources.

## matcha_components/ ← shivammehta25/Matcha-TTS @ bd4d90d (MIT)

| file | source | deviations |
|---|---|---|
| `flow_matching_base.py` | `matcha/models/components/flow_matching.py` | kept `BASECFM.__init__/forward/solve_euler` only; removed `compute_loss`, `ConformerWrapper`, `Decoder`, and imports of `conformer`/`diffusers`/`lightning` |
| `decoder.py` | `matcha/models/components/decoder.py` | removed `ConformerWrapper`, `Decoder`, `conformer` import (unused by kept classes); `get_activation` (diffusers) kept — used by `ResnetBlock1D` |
| `transformer.py` | `matcha/models/components/transformer.py` | import rewiring only (diffusers kept: `Attention`, `LoRACompatibleLinear`, …) |
| `audio.py` | `matcha/utils/audio.py` | import rewiring only; `mel_spectrogram` unchanged (n_fft 1920 / hop 480 / 80 mel / center=False per codec yaml) |

## cosyvoice_flow/ ← OpenMOSS/MOSS-Speech `feat/docs` @ 1ea408a (Apache-2.0)

| file | source | deviations |
|---|---|---|
| `flow.py` | `cosyvoice/flow/flow.py` | `omegaconf.DictConfig` defaults replaced by plain dicts (attribute access only; defaults unused by explicit construction); import rewiring |
| `flow_matching.py` | `cosyvoice/flow/flow_matching.py` | import rewiring (BASECFM → matcha_components) |
| `decoder.py` | `cosyvoice/flow/decoder.py` | import rewiring |
| `length_regulator.py` | `cosyvoice/flow/length_regulator.py` | none |
| `hifigan_generator.py`, `hifigan_f0_predictor.py` | `cosyvoice/hifigan/*` | import rewiring |
| `transformer/*.py` | `cosyvoice/transformer/*` | import rewiring (package-local) |
| `class_utils.py` | `cosyvoice/utils/class_utils.py` | trimmed to activation/subsample/positional/attention registries used by `UpsampleConformerEncoder`; llm/flow/hifigan/cli registries and imports removed |
| `utils_common.py` | `cosyvoice/utils/common.py` | kept `set_all_random_seed`, `mask_to_bias`, `get_padding`, `init_weights` only |
| `utils_mask.py` | `cosyvoice/utils/mask.py` | none |

## hf_codec/ ← OpenMOSS-Team/MOSS-Speech-Codec snapshot eeec733e (terms unconfirmed)

| file | source | deviations |
|---|---|---|
| `configuration.py` | `configuration_moss_speech_codec.py` | none |
| `whisper.py` | `modeling_whisper.py` | import rewiring (`WhisperVQConfig` from local utils); `EncoderDecoderCache.from_legacy_cache(None)` guarded — removed in transformers 5.x, no-op for the None path; return-format selection reads `return_dict` directly, retaining the legacy TorchScript tuple fallback |
| `utils.py` | `utils.py` (2285 lines) | kept `WhisperVQConfig`, `_resample_buffer`, `extract_speech_token` only; `torchaudio.load` → soundfile loader (float32 (C,T) + sr, same semantics) |
| `modeling.py` | `modeling_moss_speech_codec.py` | ① `AudioDecoder.__init__` builds the flow/HiFT stack explicitly from the frozen yaml parameter set instead of `hyperpyyaml` (removes `__set_seed*` load-time side effects and training-only `compute_fbank`/`compute_f0`); ② construction wrapped in `_scoped_global_rng` (the flow ctor's `set_all_random_seed(0)` + fixed `rand_noise` buffer preserved verbatim for parity, caller RNG restored); ③ `torchaudio.load` → soundfile; ④ added `decode_from_conditioning` + `compute_voice_conditioning` tensor-native paths (delegated to by the reference-compatible `decode(prompt_speech=path)`); ⑤ `from_pretrained` restricted to local materialized directories (offline discipline); ⑥ `device` parameter added to `MossSpeechCodec.__init__`; ⑦ correct `sess_options` keyword so the intended ONNX single-thread setting takes effect |

## Non-vendored components

- `codec_adapter.py`, `voice.py` — original code (P1); wrapping semantics documented in their docstrings.
