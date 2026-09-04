# MOSS-Speech Model Contract

> Status: P0 draft (evidence-verified against locked revisions; see `p0/01_version_lock.md`).
> Evidence keys: `[CFG]` config.json · `[MOD]` modeling_moss_speech.py · `[PROC]` processing_moss_speech.py · `[INT]` GitHub utils/interface.py · `[CDC]` modeling_moss_speech_codec.py · `[GPU]` measured on A800-80G · plan §2.1 row numbers in parentheses.

## 1. Identity

| Field | Value | Evidence |
|---|---|---|
| HF repo | `fnlp/MOSS-Speech` (4 shards, ≈17.1 GiB bf16) | [CFG] |
| architecture / model_type | `MossSpeechForCausalLM` / `moss_speech` | [CFG] |
| Remote code | configuration/modeling/processing_moss_speech.py, written for transformers **4.57.0.dev0** | [CFG] |
| Reference driver | GitHub `OpenMOSS/MOSS-Speech` branch `feat/docs` @ `1ea408a` (`gradio_demo.py`, `utils/interface.py`) | [INT] |
| Codec | `fnlp/MOSS-Speech-Codec` (`WhisperVQEncoder` arch; code from GitHub `cosyvoice/` + Matcha-TTS @ `bd4d90d`) | [CDC] |

## 2. Architecture

- 32 shared layers + 4-layer `text_block` tail + 4-layer `audio_block` tail = **40 layer instances** (plan §2.1-3 ✅). `num_hidden_layers=36` in config is NOT the KV-relevant count; the base model instantiates `shared_block(32)` + two `MossSpeechTransformerBlock(4)` separately. [MOD L472–505]
- Hidden 4096, GQA 32 q-heads / 8 kv-heads, head_dim 128, intermediate 12288, RMSNorm eps 1e-6, RoPE theta 1e6, max_position 40960 (SFT context 10240 per tech report), attention_bias=False, QK-RMSNorm per head (Qwen3 pattern), supports eager/sdpa/flash. [CFG, MOD L231–311]
- **Three KV caches**: `past_key_values_dict = {"shared": 32-layer cache, "text": 4-layer, "audio": 4-layer}`; per-token KV = 40 layers × 2 × (8×128) × 2 B = **160 KB/token bf16** (4K ctx ≈ 0.62 GiB, 10K ≈ 1.53 GiB). [MOD L592–615]
- Dual embedding tables: `embed_tokens` (151680×4096) + `audio_embed` (16512×4096); per-position selection: text embedding where text-channel ≠ modality_pad(151667), audio embedding elsewhere. [MOD L556–570]
- Dual lm_heads: `text_lm_head` (→151680), `audio_lm_head` (→16512); both weights **present in checkpoint** (446 keys, no missing/unexpected) → `_tied_weights_keys` listing is a save-time hint, not an actual tie (`tie_word_embeddings=false`). [MOD L993, weight index]
- Tail routing: shared trunk output feeds BOTH tails every forward; each tail has its own final norm (`text_norm`/`audio_norm`); `modalities=["text","audio"]` hardcoded in `ForCausalLM.forward`. [MOD L1046–1051]

## 3. Token & grid conventions

- Grid `(B, L, 2)` channels-last is the processor/generate convention; `ForCausalLM.forward` transposes to `(B, 2, L)` internally. Labels docstring also uses `(B,2,L)`. Both orientations appear; port must be explicit. [PROC collate L220, MOD L1041–1043]
- Key token ids [CFG]: `sosp=151646` (text channel), `eosp=16384` (audio channel), `modality_pad=151667` (text channel placeholder inside audio segments), `audio_pad=512` (audio channel placeholder inside text segments), eos `<|im_end|>=151645`, text vocab 151680, audio vocab 16512.
- Audio segment layout (per turn) [PROC L120–140]: text channel `[sosp, modality_pad × (n−1), modality_pad]`, audio channel `[audio_pad, codes…, eosp]`, equal length n+2... precisely: `len = n_codes + 2` both channels, position-aligned.
- Chat template: processor hand-builds `<|im_start|>{role}\n{content}<|im_end|>\n` segments; the repo's `chat_template.jinja` (Qwen lineage) is NOT used by the processor. Assistant prefix for audio output: `<|im_start|>assistant\n<|object_ref_start|>`; for text output: `<|im_start|>assistant\n`. [PROC L221–243]
- System prompt (official defaults) [INT]: audio → "You are a helpful voice assistant. Answer the user's questions with spoken responses."; text → "You are a helpful assistant. Answer the user's questions with text."; processor has its own fallback prompts (different wording) used only when no system turn present. [PROC L262–266]
- Batch collate: LEFT padding; text channel padded with `pad_token_id`, audio channel with `audio_pad`; attention prefix zeros. [PROC L200–218]

## 4. Input/output schema

- Input conversation: list of turns; `content` is str (text turn) or `{"path": <wav path>, "type": "audio/wav"|"filepath"}` (audio turn); roles user/assistant/system; multi-turn may freely mix text/audio turns (processor encodes each audio turn via codec). Single-turn content single-type. [PROC L142–190, INT]
- `output_modalities`: `"text"` or `"audio"` only (both → error in interface; processor raises NotImplementedError otherwise). [INT L62–65, PROC L240]
- Output: `MossSpeechResponse{generated_text, audio (T,) tensor, sampling_rate=24000}`; grid returned by `generate` is prompt-stripped (`output_only=True` default). [PROC L347–394, MOD L919–939]
- Decode: text channel detokenized with `skip_special_tokens` + `<|empty|>`→"." / `<|end_empty|>`→":" replacements; audio codes = audio-channel tokens between first sosp/eosp; batch decode NOT implemented (B=1 only). [PROC L360–393]

## 5. Generation FSM, sampling, stopping

- FSM [MOD L858–870]: start modality inferred from last prompt position (text-ch==modality_pad → audio; audio-ch==audio_pad → text); text mode + last text token==sosp → switch audio; audio mode + last audio token==eosp → switch text; stop via stopping criteria on **text channel only** (`input_ids[:, :, 0]`). [MOD L912]
- Every step: forward computes both tails; both channels sampled; when in audio mode, text-channel token is overwritten with modality_pad (151667); in text mode the audio channel is still sampled-and-appended but ignored on input (asymmetric — corrections to plan §2.1 "非活跃通道出 pad": only the text channel is padded). [MOD L723–770, L897–901]
- Audio logits mask: `audio_logits[:, 16385:] = −inf` (hard), and eosp suppressed while `generating_length < min_new_tokens`. [MOD L735–739]
- Per-channel logits processors supported via `generation_config.layers` (per-channel temperature/top_p/top_k/repetition_penalty) — unused by the reference interface, which applies global params to both channels. [MOD L701–721]
- Reference sampling defaults [INT]: temp 0.6, top_p 0.95, top_k 20, repetition_penalty 1.1, max_new 500, min_new 0, `do_sample=True` hardcoded.
- Stoppers: `MIMOStopper(pad_token_id)` and `MIMOStopper(<|im_end|>)`, evaluated on the text channel's last token; B=0 semantics (checks batch element 0). [INT L22–29]
- KV across modalities: text tail writes KV at every step including audio-segment positions; after eosp, text generation attends over those positions (verified: all three caches share seq length; empirically the model emits `<|im_end|>` immediately after eosp — no content-bearing text tail in any observed run; see `p0/05_trace_and_kv.md`). No single-tail shortcut.

## 6. Codec contract

- **Encoder** [CDC]: causal Whisper-large-v3-style VQ encoder (d_model 1280, 32+32 layers, causal conv + causal attention, pooling kernel 4 → 12.5 Hz), input resampled to 16 kHz, output single codebook ids in [0, 16384), `quantize_vocab_size=16384`; `encode()` accepts file paths / (wav, sr) / tensors, internal batch 128, returns `List[List[int]]`. Frame rate 12.5 Hz confirmed at runtime (`input frame rate=12.5`).
- **Decoder** [CDC]: CosyVoice2-style flow-matching (`CausalMaskedDiffWithXvec` from GitHub `cosyvoice/`, Matcha-TTS components) + HiFi-GAN (`hift.pt`); output 24 kHz mono; `decode(codes (B,1,T), prompt_speech=path)` conditions on THREE prompt artifacts: prompt codec codes + prompt 80-mel (24 kHz, truncated to 4×tokens) + campplus xvector (16 kHz, 192-dim). Prompt resample chain 44.1k→24k→16k is correct.
- Streaming surface: `AudioDecoder.{offline_inference, stream_inference, streaming_inference}`; flow weights ship as `flow.pt` + `flow-chunk-5.pt` + `flow-chunk-25.pt`; scratch config: chunk_size 5 tokens, pre_lookahead_len 3, token_mel_ratio 4, mel_overlap. (V1 non-streaming only; details for P1/V1.1.)
- Runtime measured values: see `p0/06_codec_contract.md`. Measured VRAM (colocated, A800): model+codec 35.7 GiB allocated / 39.0 GiB reserved → 24G cards infeasible without codec downcast or split placement (`p0/05_trace_and_kv.md`).

## 7. Known limitations & official-code quirks (port must NOT inherit silently)

1. `MIMOInterface.default_decoder_audio_prompt_path = "./assets/prompt_cn.wav"` — wrong filename (actual asset: `prompt-cn.wav`); gradio default `value=".assets/prompt_cn.wav"` adds a leading-dot typo. Reference drivers must pass the prompt explicitly.
2. Batch decode not implemented; reference is B=1 throughout.
3. `channels: 2` in model config = token-grid channels, NOT audio waveform channels (waveform is mono 24 kHz out / 16 kHz in).
4. Remote code targets transformers 4.57.0.dev0; on 4.57.1 `generate()` drops `streamer=None` kwarg → `_sample(streamer)` TypeError; reference runs need a no-op streamer (driver-level, semantics-neutral).
5. torchaudio 2.9 `load` routes to TorchCodec; cluster-compatible wheels unavailable → soundfile shim in reference driver (IO-only).
6. HF weight repos carry no LICENSE file (code: Apache-2.0 GitHub / MIT Matcha); redistribution terms unconfirmed — open item.
7. Audio loss in `ForCausalLM.forward` passes `vocab_size=151680` for the 16512-wide audio head (training-only bug, irrelevant to inference).
8. `interface.py` top-level `import gradio` + monkey-patch of `gr.processing_utils._check_allowed` (UI convenience; headless use requires gradio installed or patch stripped).

## 8. HF → SGLang weight mapping (draft)

| HF namespace (446 keys) | Count | SGLang target (flattened) |
|---|---|---|
| `model.embed_tokens.weight` | 1 | `embed_tokens` (text vocab 151680) |
| `model.audio_embed.weight` | 1 | second embedding (audio vocab 16512) |
| `model.shared_block.layers.{0..31}.*` | 32×10 | trunk layers 0–31 (q/k/v/o_proj, q/k_norm, 2×ln, mlp gate/up/down) |
| `model.text_block.layers.{0..3}.*` | 4×10 | text tail layers 32–35 |
| `model.audio_block.layers.{0..3}.*` | 4×10 | audio tail layers 36–39 |
| `model.text_norm.weight` / `model.audio_norm.weight` | 1+1 | per-branch final norms |
| `text_lm_head.weight` / `audio_lm_head.weight` | 1+1 | dual lm heads (explicit, not tied) |

Note: tail blocks both instantiate with `start_idx=0` (they read `layer_types[0:4]`); weight names are block-scoped, so mapping is unambiguous.

## 9. Version lock

See `p0/01_version_lock.md` (HF snapshots `cff025bb…` / `eeec733e…`, GitHub `feat/docs@1ea408a`, Matcha `bd4d90d`, hashes, licenses).
