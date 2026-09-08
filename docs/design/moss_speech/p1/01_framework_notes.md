# P1-01 Framework, Dependency & Experiment-Design Audit (T1.1)

Date: 2026-09-06. Login-node analysis over the locked revisions (P0 `01_version_lock.md`).

## 1. Precedent comparison (component organization)

| model | codec/vocoder organization | reusable for MOSS-Speech | not applicable because |
|---|---|---|---|
| `moss_tts` | fully native: vendored `audio_tokenizer.py`, custom `vocoder.py` + CUDA kernels, `streaming_vocoder.py` | engineering blueprint for "vendor close to reference then optimize" | different codec (16 codebooks/delay pattern), kernels are MOSS-TTS-specific |
| `fun_cosyvoice3` | **closest kin**: extracts `flow`+`hift` from an installed `cosyvoice` package, `_CosyVoice3ReferenceEncodeHook(KeyedReferenceEncodeHook)`, `ReferenceEncodeService`, batched `decode_batch`, `SimpleScheduler` preprocessing | hook pattern, flow/hift extraction idea, decode-batch call shape | requires pip `cosyvoice` package; pypi variant lacks our causal classes (`CausalMaskedDiffWithXvec`), `.venv-omni` has no cosyvoice → we vendor instead |
| `qwen3_omni` | chat-native, `terminal_stages=[decode, code2wav]` dual-terminal precedent | terminal-routing reference for P2 | codec is qwen3-tts specific |
| `moss_tts_local` | local-transformer + `state_pool.py` + streaming vocoder | state-pool/abort-cleanup patterns | local transformer unrelated |

Decision: vendor the **HF `AudioDecoder`/`MossSpeechCodec` + cosyvoice/matcha inference closure** (closest to locked reference semantics; fun_cosyvoice3-style reimplementation deferred until a proven need for batched decode).

## 2. Scheduler contracts (verified in code)

- `SimpleScheduler.__init__`: `max_concurrency > 1` XOR `batch_compute_fn` (raises `ValueError`) — `simple_scheduler.py:64-68`.
- Concurrency path spawns N worker coroutines dispatching `compute_fn` via `asyncio.to_thread` → **compute_fn must be re-entrant** (`simple_scheduler.py:58-63`).
- Terminology fixed for T1.4: **queue concurrency** (arrival/backlog), **execution concurrency** (`max_concurrency` workers), **internal batch** (`encode(batch_size=…)` chunking inside one call), **cross-request batching** (`batch_compute_fn`). Only the last one coalesces requests.
- `ThreadedSimpleScheduler` exists (`threaded_simple_scheduler.py`) for sync-first stages.
- Abort: `abort_callback` wired on every scheduler touching shared state; idempotent by design (tts_model_integration.md "Scheduler contracts").

## 3. Dependency closure audit (inference only)

### 3.1 What the reference actually loads

`MossSpeechCodec` (HF eeec733e) = `WhisperVQEncoder` (causal whisper + RVQ codebook) + `AudioDecoder`; `AudioDecoder.__init__` runs `load_hyperpyyaml(flow/config.yaml)` which dynamically constructs:

| yaml target | class | inference-needed? |
|---|---|---|
| `flow` | `cosyvoice.flow.flow.CausalMaskedDiffWithXvec` (encoder=`UpsampleConformerEncoder`, decoder=`CausalConditionalCFM`→`CausalConditionalDecoder`) | yes |
| `hift` | `cosyvoice.hifigan.generator.HiFTGenerator` + `ConvRNNF0Predictor` | yes |
| `feat_extractor` | `Matcha-TTS.matcha.utils.audio.mel_spectrogram` (n_fft 1920, hop 480, 80 mel, center=False) | yes |
| `compute_fbank`/`compute_f0` | `cosyvoice.dataset.processor.*` | **training only — strip** |
| `__set_seed1-4` | `!apply random/np/torch/cuda seed [1986]` | **load-time side effect — strip** |

### 3.2 RNG semantics (critical finding)

- `CausalConditionalCFM.__init__` calls `set_all_random_seed(0)` then builds a **fixed noise buffer** `self.rand_noise = torch.randn([1, 80, 50*300])`; decode uses `z = self.rand_noise[:, :, :T]` (`flow_matching.py:200-224`).
- Therefore **decode is a pure function of (codes, voice conditioning, weights)** — call-time seeding is inert in the reference. P0's "same-seed determinism" is structural; P0 never claimed per-request decode seeds.
- Consequences: (a) adapter must reproduce the identical `rand_noise` buffer for bit-parity; (b) per-request decode RNG isolation is unnecessary in V1 — no seed parameter exposed as a capability; (c) the `set_all_random_seed(0)` **global-RNG pollution happens at model construction** → adapter wraps decoder construction in a save/restore RNG scope; (d) AR sampling isolation (AGENT.md §4.2) remains a P3/P4 concern, untouched by codec.
- `AudioDecoder` holds per-uuid state (`mel_overlap_dict`, `hift_cache_dict` defaultdicts) → `cleanup(request_id)` must pop its uuid keys; decode lock serializes access in V1.

### 3.3 Vendor scope & external deps (final)

```
components/
  hf_codec/     ← models/MOSS-Speech-Codec @ eeec733e (trimmed):
      configuration.py          (MossSpeechCodecConfig)
      modeling.py               (MossSpeechCodec + AudioDecoder; hyperpyyaml → explicit python config)
      whisper.py                (WhisperVQEncoder/Layer/Config, 322 lines)
      utils.py                  (extract_speech_token + WhisperVQConfig + minimal deps only)
  cosyvoice_flow/ ← repos/MOSS-Speech feat/docs @ 1ea408a (subset):
      flow.py, flow_matching.py, decoder.py, length_regulator.py,
      upsample_encoder.py (+ transformer/{attention,convolution,encoder_layer,positionwise_feed_forward,subsampling?}.py as imported),
      hift_generator.py, hift_f0_predictor.py, utils_common.py (mask/bias helpers)
  matcha_components/ ← Matcha-TTS @ bd4d90d (subset):
      flow_matching_base.py     (BASECFM: init/forward/solve_euler only — heavy imports stripped),
      decoder.py, transformer.py (Block1D/ResnetBlock1D/BasicTransformerBlock/…),
      audio.py                  (mel_spectrogram)
```

External deps of the closure: **torch, torchaudio (kaldi fbank, Resample), transformers (WhisperFeatureExtractor, PreTrainedModel, ACT2FN, cache_utils), numpy, scipy (get_window), librosa (mel filter), einops, onnxruntime (campplus), soundfile (input normalization), safetensors**.

Explicitly NOT required after vendoring: lightning, matplotlib, pyworld, gdown, wget, pyarrow, hydra, hyperpyyaml, omegaconf*, conformer (pypi). (*`omegaconf.DictConfig` only provides attribute access for cfm params → replaced by a local frozen namespace; `.venv-omni` has omegaconf anyway, but removing it shrinks the closure.)

Version-compat risk to verify in T1.2 smoke: `modeling_whisper.py` against **transformers 5.12.1** (`.venv-omni`) — `EncoderDecoderCache`, `ACT2FN`, `PreTrainedModel` import surface; any fix goes into the vendored copy with a documented diff, never into `.venv-omni`.

Licenses: HF codec code (no file — treated per GitHub repo Apache-2.0; recorded as P0 open item), cosyvoice @ 1ea408a (Apache-2.0), Matcha-TTS @ bd4d90d (MIT). Each vendored subtree carries its LICENSE + source SHA.

### 3.4 Input normalization boundary

Reference `extract_speech_token` uses `torchaudio.load` for path inputs (torchcodec trap, P0 §7.5). Adapter normalizes ALL inputs (path / `(wav, sr)` / tensor) via soundfile + `torchaudio.transforms.Resample` to 16 kHz mono at the boundary, then calls the closure with tensors — path handling stays out of the vendored code.

## 4. Voice conditioning & ReferenceEncodeService (reuse decision)

**Reuse: yes.** `VoiceConditioning` = multi-tensor artifact (prompt codes `[T]` int64, prompt mel `[1, 80, 4T]` fp32, xvector `[192]` fp32, plus sr/dtype metadata) → implement a `KeyedReferenceEncodeHook` subclass (pattern per `_CosyVoice3ReferenceEncodeHook`):

- key parts: `model_id="fnlp/MOSS-Speech-Codec"`, `model_revision=eeec733e…`, `encoder_id="moss_speech_voice"`, `encoder_config_hash` (vendored-closure content hash), `artifact_kind="moss_speech_voice"`, `input_key` = blake2b of reference audio bytes (CPU materialization for metadata only).
- `store_artifact` → dict of detached CPU tensors; `load_artifact` → clone to target device/dtype (caller-owned).
- single-flight + byte-budget + no-poison-on-failure come free from the service.
- ownership: default voice precomputed at startup, owned by the decoder-side component; per-request ad-hoc voices (future) go through the service. `cleanup(request_id)` never evicts the shared default voice.

Enrollment reuses the encoder only; mel/xvector computed by dedicated feature paths (mel via `matcha_components.audio`, xvector via campplus+fbank) — **no decoder load for voice enrollment** (G1 requirement).

## 5. Experiment protocols (frozen before measurement)

### T1.3 calibration & acceptance

- Reference calibration set: t2s_cn/s2s_cn codes (P0 fixtures), prompt-cn/en voices, plus synthetic edge cases (1-code, empty, eosp-truncated, out-of-range). Run reference decode ×3 repeats (seed variety irrelevant post-§3.2, but protocol keeps 2 seeds to prove inertness), encode ×2 repeats on the 5 P0 audio inputs.
- Frozen thresholds (recorded in `02_alignment.md` BEFORE adapter runs): **encode = per-sample code equality, exact**; **decode fp32 = waveform blake2b equality** (hash inequality → FAIL, no tolerance invented; root-cause analysis mandatory); condition tensors (prompt codes/mel/xvector) compared exactly (ints) / allclose 1e-6 (floats). Batch encode accepted only if per-code equal to per-sample runs at every tested `batch_size ∈ {1,4,128}` including mixed lengths.
- Isolation checks: interleaved requests (different voices), cancel/retry, duplicate cleanup — outputs must equal the solo-run hashes; global RNG state sampled before/after each adapter call must be bit-identical.

### T1.4 A/B protocol

- Workload manifest: fixed seeds/voices; text-only and speech requests mixed 50/50 (text skips encoder); speech clips short (~3 s) and long (~27 s); decode codes {100, 200, 500}; arrival = fixed schedule (Poisson later in P4); ≥3 warmup, 20 completed requests, 3 rounds per config; wall-clock timing at stage boundaries, CUDA events for device time, `torch.cuda.synchronize` kept OUT of concurrent hot paths (per-request timing via events only).
- Layout A: preprocessing+encoder one process (SimpleScheduler colocated); Layout B: separate encoder process (its own SimpleScheduler) with the actual Stage transport/payload mechanism used by the framework (queue + pickle payload), same single GPU, one encoder instance each, independent decoder terminal in both. Both run first without AR, then with an **independent reference-AR load process** (bf16, `.venv-p0`, P0 canonical inputs, AR-only weights — no reference codec).
- Metrics per layout: throughput, p50/p95 per stage, overlap windows with AR, per-process + total GPU memory (weights/allocated/reserved peaks, load transients), IPC bytes/latency.

## 6. Minimal viable configuration (V1 baseline)

- Single A800, AR bf16 (P3/P4), **codec fp32**, encoder colocated (pending A/B), decoder = `audio_vocoder` terminal on `SimpleScheduler`, `max_concurrency=1`, `batch_compute_fn=None`, encode internal `batch_size=128` (evidence T1.3/T1.4), default voice precomputed once. Optional (non-blocking): bf16/fp16 codec, higher execution concurrency, chunk5/25 probes.

## 7. Framework-gap watch list

No gap requiring a framework PR found in T1.1. Watch items: (a) `SimpleScheduler` re-entrancy note if decode ever goes concurrent (thread-safety of vendored `AudioDecoder` state); (b) chat-side `modalities` routing precedent exists (qwen3_omni) — P2 concern; (c) `ReferenceEncodeService` fits as-is.


## P5 dependency inventory correction (2026-09-08)

The inference closure in section3.3 also requires diffusers (verified0.37.0):
the vendored Matcha decoder/transformer imports activation, attention and LoRA
helpers from it. It is already a base project dependency. A new CPU environment
exposed the omission in this prose inventory; the standalone CPU requirements
now include it. Decoder implementation and P1 evidence are unchanged.
