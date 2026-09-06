# Vendored Sources & Diff Manifest (components/)

All vendored code serves the MOSS-Speech codec inference closure only. Each
subtree keeps its upstream license; numerics of kept methods are unchanged
unless a deviation is listed. Locked revisions: see
`docs/design/moss_speech/p0/01_version_lock.md`.

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

## hf_codec/ ← fnlp/MOSS-Speech-Codec snapshot eeec733e (license: no file upstream — Apache-2.0 assumed pending MOSS confirmation)

| file | source | deviations |
|---|---|---|
| `configuration.py` | `configuration_moss_speech_codec.py` | none |
| `whisper.py` | `modeling_whisper.py` | import rewiring (`WhisperVQConfig` from local utils); `EncoderDecoderCache.from_legacy_cache(None)` guarded — removed in transformers 5.x, no-op for the None path |
| `utils.py` | `utils.py` (2285 lines) | kept `WhisperVQConfig`, `_resample_buffer`, `extract_speech_token` only; `torchaudio.load` → soundfile loader (float32 (C,T) + sr, same semantics) |
| `modeling.py` | `modeling_moss_speech_codec.py` | ① `AudioDecoder.__init__` builds the flow/HiFT stack explicitly from the frozen yaml parameter set instead of `hyperpyyaml` (removes `__set_seed*` load-time side effects and training-only `compute_fbank`/`compute_f0`); ② construction wrapped in `_scoped_global_rng` (the flow ctor's `set_all_random_seed(0)` + fixed `rand_noise` buffer preserved verbatim for parity, caller RNG restored); ③ `torchaudio.load` → soundfile; ④ added `decode_from_conditioning` + `compute_voice_conditioning` tensor-native paths (delegated to by the reference-compatible `decode(prompt_speech=path)`); ⑤ `from_pretrained` restricted to local materialized directories (offline discipline); ⑥ `device` parameter added to `MossSpeechCodec.__init__` |

## Non-vendored components

- `codec_adapter.py`, `voice.py` — original code (P1); wrapping semantics documented in their docstrings.
