# P0-06 Codec Contract Verification (T0.7)

Date: 2026-09-04. Raw results: `artifacts/p0/codec/codec_contract.json` (A800 run, log `sbatch_suite_3628.log`).

## Encoder (audio → codes)

| metric | measured | contract expectation | status |
|---|---|---|---|
| frame rate | 343 codes / 27.43 s = **12.505 Hz**; 1 s tensor input → 13 codes | 12.5 Hz | ✅ |
| code range | [248, 16352] ⊂ [0, 16384) | `quantize_vocab_size=16384` | ✅ |
| input sr | resampled to 16 kHz internally (whisper mel, 128 bins) | 16 kHz | ✅ |
| batch encode | `[cn, en]` → lengths [343, 453] (per-item lengths preserved; no cross-item padding loss) | — | ✅ |
| api | `encode(paths | (wav,sr) | tensors, batch_size=128) -> List[List[int]]` | — | ✅ |

## Decoder (codes → audio)

| metric | measured | expectation | status |
|---|---|---|---|
| output sr / channels | 24000 Hz, mono | 24 kHz mono | ✅ |
| hop | 200 codes → duration s.t. **80.0 ms/token** (=1/12.5 Hz) | — | ✅ |
| determinism | same seed + same prompt → identical waveform blake2b | flow ODE seeded | ✅ |
| voice conditioning | same codes, prompt-cn vs prompt-en → **different waveforms** | xvector + prompt tokens + prompt mel | ✅ |
| conditioning inputs | prompt codec codes + prompt 80-mel@24k (truncated to 4×codes) + campplus 192-d xvector@16k | CosyVoice2-style | ✅ |
| prompt resample chain | load@orig_sr → 24k → mel; 24k → 16k → xvector (correct for 44.1 kHz assets) | — | ✅ |

## Streaming surface (recorded only; V1 non-streaming)

- `AudioDecoder` methods: `offline_inference`, `stream_inference`, `streaming_inference` (signatures in codec_contract.json).
- Shipped flow weights: `flow.pt`, `flow-chunk-5.pt`, `flow-chunk-25.pt`, `hift.pt` (+ `campplus.onnx`).
- Scratch config: `chunk_size: 5` tokens, `pre_lookahead_len: 3`, `token_mel_ratio: 4`, `num_decoding_left_chunks: -1`; config yaml seeds RNG at 1986.
- `decode(..., finalize=True, uuid=os.urandom)` — per-call uuid keys chunk state; P1 adapter must map uuid/finalize semantics.

## Port notes for P1 adapter

1. `decode()` requires a `prompt_speech` **file path** (API, not tensor) — adapter should accept waveform/tensor and wrap.
2. decode hardcodes `device=cuda` default with CPU fallback.
3. encode is a pure encoder pass (no state) — safe to colocate in preprocessing; decode holds flow+hift+campplus (~3 GiB+ VRAM, peak 37 GiB measured alongside AR model on one 80G card).
4. Batch decode NOT implemented upstream (single-sample loop); P1 must decide per-request decode vs micro-batch.

## License / deps

- Matcha-TTS @ bd4d90d: MIT (vendored, `repos/MOSS-Speech/Matcha-TTS/LICENSE`).
- Matcha brings transitive imports (lightning, conformer, diffusers, matplotlib, pyworld, gdown, wget, pyarrow, hydra, omegaconf, hyperpyyaml, setuptools<81 for pkg_resources) — for the sglang-omni port, vendor only `matcha/models/components/*` + `matcha/hifigan/models.py` and re-audit (target: torch + diffusers + einops only).
