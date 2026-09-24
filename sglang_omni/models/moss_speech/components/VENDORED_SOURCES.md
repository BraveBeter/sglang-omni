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
| `flow.py` | `cosyvoice/flow/flow.py` | `omegaconf.DictConfig` defaults replaced by plain dicts (attribute access only; defaults unused by explicit construction); import rewiring; opt-in shape-based inference masks avoid host length synchronization |
| `flow_matching.py` | `cosyvoice/flow/flow_matching.py` | import rewiring (BASECFM → matcha_components); opt-in reuse of fixed device noise, CFG conditions, and per-solve attention bias; release unused intermediate solutions |
| `decoder.py` | `cosyvoice/flow/decoder.py` | import rewiring; opt-in per-solve mask/head expansion reuse, keyed by resolution, dtype, device and chunk profile |
| `length_regulator.py` | `cosyvoice/flow/length_regulator.py` | none |
| `hifigan_generator.py`, `hifigan_f0_predictor.py` | `cosyvoice/hifigan/*` | import rewiring |
| `transformer/*.py` | `cosyvoice/transformer/*` | import rewiring (package-local) |
| `class_utils.py` | `cosyvoice/utils/class_utils.py` | trimmed to activation/subsample/positional/attention registries used by `UpsampleConformerEncoder`; llm/flow/hifigan/cli registries, unused `get_model_type` factory, and associated imports removed |
| `utils_common.py` | `cosyvoice/utils/common.py` | kept `set_all_random_seed`, `mask_to_bias`, `get_padding`, `init_weights` only; unused TensorRT `TrtContextWrapper` removed |
| `utils_mask.py` | `cosyvoice/utils/mask.py` | none |

## hf_codec/ ← OpenMOSS-Team/MOSS-Speech-Codec snapshot eeec733e (terms unconfirmed)

| file | source | deviations |
|---|---|---|
| `configuration.py` | `configuration_moss_speech_codec.py` | none |
| `whisper.py` | `modeling_whisper.py` | import rewiring (`WhisperVQConfig` from local utils); `EncoderDecoderCache.from_legacy_cache(None)` guarded — removed in transformers 5.x, no-op for the None path; return-format selection reads `return_dict` directly, retaining the legacy TorchScript tuple fallback |
| `utils.py` | `utils.py` (2285 lines) | kept `WhisperVQConfig`, `_resample_buffer`, `extract_speech_token` only; `torchaudio.load` → soundfile loader (float32 (C,T) + sr, same semantics) |
| `modeling.py` | `modeling_moss_speech_codec.py` | ① `AudioDecoder.__init__` builds the flow/HiFT stack explicitly from the frozen yaml parameter set instead of `hyperpyyaml` (removes `__set_seed*` load-time side effects and training-only `compute_fbank`/`compute_f0`); ② construction wrapped in `_scoped_global_rng` (the flow ctor's `set_all_random_seed(0)` + fixed `rand_noise` buffer preserved verbatim for parity, caller RNG restored); ③ `torchaudio.load` → soundfile; ④ added `decode_from_conditioning` + `compute_voice_conditioning` tensor-native paths (delegated to by the reference-compatible `decode(prompt_speech=path)`); ⑤ `from_pretrained` restricted to local materialized directories (offline discipline); ⑥ `device` parameter added to `MossSpeechCodec.__init__`; ⑦ correct `sess_options` keyword so the intended ONNX single-thread setting takes effect; ⑧ optional codec optimizations, shared HiFT weights and omission of the unused speaker session in serving adapters |

## Non-vendored components

- `codec_adapter.py`, `voice.py` — original code (P1); wrapping semantics documented in their docstrings.

The `codec_optimizations` serving option defaults to disabled. It retains FP32,
the original diffusion steps, masks, flow weights and final full-context pass.
Its mask cache is local to one ODE solve; it does not cache historical flow
activations between chunks. Streaming overlap retains the reference CPU
float64 window arithmetic and stores final samples in FP32.

The separate `codec_memory_optimizations` option reuses the private encoder
attention score allocation and releases positional score temporaries before
masking and softmax. A read-only relative-shift view replaces padding/copying
for contiguous square self-attention; other layouts retain the copy path.
Matrix multiplication and attention shapes, dtypes, and add/divide order remain
unchanged. Storage reuse applies only to evaluation with gradients disabled;
default and training paths are retained.

The memory option defaults to disabled and is independent of codec preparation.
It reduces peak codec allocations and can increase latency.

## Local CFM processor derived from Diffusers 0.37.0 (Apache-2.0)

`cosyvoice_flow/cfm_attention.py` specializes the projection and output sequence
of `AttnProcessor2_0` in Hugging Face Diffusers 0.37.0,
`src/diffusers/models/attention_processor.py` (Copyright 2025 The HuggingFace
Team). Its Apache-2.0 notice is preserved in the local file. The installed
Diffusers processor remains the fallback; this does not replace the shared
Diffusers implementation.

With `codec_optimizations`, a private all-valid mask proof is passed through
`flow.py`, `flow_matching.py`, and `decoder.py`. On the specifically qualified
A800/FP32 runtime and discrete streaming attention lengths, two aligned query
segments omit keys that the original causal mask suppresses. QKV projections,
mask values, backend preferences, output strides, diffusion steps and final
full-context decoding remain unchanged. Unknown hardware/builds, shapes,
layouts, masks, training/custom processors or selected backends use the original
path. The qualification is empirical bitwise regression of the supported
shapes and tested inputs, rather than a claim that arbitrary SDPA subdomains
are automatically bitwise equivalent.
