# P0-07 Collision Check & Design Issue Draft (T0.9)

## 1. PR #1902 (MOSS-TTS-Nano) overlap analysis

Upstream: `sgl-project/sglang-omni` PR #1902 "feat(tts): add MOSS-TTS-Nano support" (open, author z2z23n0).

- Scope: MOSS-TTS-Nano, **one-way TTS** via OpenAI `/v1/audio/speech`; pipeline `preprocessing -> tts_engine -> vocoder`; 16 audio codebooks; MOSS-Audio-Tokenizer-Nano; 48 kHz output; reference-free + voice-clone.
- PR-specific files: `sglang_omni/models/moss_tts_nano/` (12 files), `tests/unit_test/moss_tts_nano/`, `examples/configs/moss_tts_nano.yaml`, `docs/cookbook/moss_tts_nano.md`. (The 552-file diff vs `origin/main` is dominated by upstream commits merged into the PR branch; not PR-owned.)
- Our touch points: `sglang_omni/models/moss_speech/`, `tests/unit_test/moss_speech/`, `examples/configs/moss_speech*.yaml`, `docs/cookbook/moss_speech.md`, benchmarks wiring.

**Intersection: none at file level.** Both add model packages through the standard `EntryClass` auto-discovery. Only adjacent (additive) shared edits are possible: `sglang_omni/cli/serve.py` frozensets (if `--decode-mode` flags are needed) and `benchmarks/` wiring — string additions, independently reviewable, no semantic conflict.

Conceptual boundary (for the design issue): MOSS-Speech is a **speech-to-speech chat model** (four I/O modes) served via `/v1/chat/completions` with audio input routing and `modalities`-based output routing; MOSS-TTS-Nano is a **text-to-speech** model on the dedicated TTS endpoint. Different checkpoints, different token topology (dual-channel grid vs 16 codebooks), different endpoints. We do not borrow or modify `moss_tts_nano`; our engineering blueprint for multi-channel mechanics remains `models/moss_tts` (plan §4 P3).

## 2. Design issue draft (to post with PR 1; English, upstream-facing)

> **Title: [Model] MOSS-Speech: four-mode speech-to-speech chat model (S2S/S2T/T2S/T2T)**
>
> **Motivation.** MOSS-Speech (fnlp/MOSS-Speech) is a speech-to-speech dialogue model with a shared 32-layer trunk plus 4+4 text/audio tails, a `(B, L, 2)` dual-channel token grid (text vocab 151680, audio vocab 16512, single 12.5 Hz codebook), and a CosyVoice2-style flow-matching codec. We adapt it as a chat-native model: `/v1/chat/completions` with audio input and `modalities=["text"|"audio"]` output routing, non-streaming first.
>
> **Scope.**
> - V1: four modes, non-streaming, TP=1, bf16, eager; output-modality-isolated batching; RadixCache/CUDA Graph/quantization off.
> - Model-specific logic fully under `sglang_omni/models/moss_speech/` (config, stages, request builders, payload types, engine builder, SGLang model class incl. ported `MossSpeechConfig`, model runner with dual-channel FSM/sampler, codec adapter components).
> - KV accounted as 40 layer instances (32 shared + 4 text + 4 audio); three-cache topology verified against the reference (see contract §5).
> - No framework changes planned; if a gap is proven (e.g., conditional terminal routing), a separate minimal framework PR will be proposed.
>
> **Out of scope / non-overlap.** Not a TTS model: no `/v1/audio/speech` endpoint, does not touch MOSS-TTS-Nano (#1902) or `moss_tts_local`. Streaming speech (SSE audio delta) follows as a separate V1.1 PR; performance capabilities (cache/graph/TP/quant) as separate V2 PRs.
>
> **Evidence.** `MODEL_CONTRACT.md` + golden fixtures (greedy, bit-exact reproducible on A800) + four-mode reference baseline in `docs/design/moss_speech/p0/`.
>
> **Risks.** (condensed from plan §10) branch-tail KV semantics; codec streaming dependencies (Matcha vendoring, MIT); endpoint default-value pollution of sampling; 24 GB feasibility unproven until measured.

## 3. Framework-gap candidates (watch list; none confirmed in P0)

1. Conditional terminal routing text vs vocoder for one model (qwen3-omni `terminal_stages` precedent) — decide in P2.
2. Colocated codec-in-preprocessing concurrency limits — P1 placement decision.
3. `serve.py` decode-mode frozensets — additive only.
