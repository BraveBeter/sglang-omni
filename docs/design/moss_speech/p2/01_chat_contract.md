# P2-01 Chat Contract, Defaults & Framework Interface Audit (T2.1)

Date: 2026-09-07. Login-node audit against `feat/moss-speech` base (`0f0c2b8` + P1 commits). All source references verified in code.

## 1. Verified framework interfaces (with source locations)

| interface | fact | source |
|---|---|---|
| registry discovery | iterates `sglang_omni.models.*` subpackages, imports `<pkg>.config` and reads **`config.EntryClass`**; matches `architecture` (+optional `architecture_aliases`) against the HF config; duplicate registrations raise | `models/registry.py:35-80` |
| routing | `route_fn` imported by string, called as **`fn(request_id, output)`**, must return exactly one target from `next` (validated; empty → error); terminal stages reject `route_fn` | `pipeline/stage_workers.py:648-657`, `config/schema.py:540-542` |
| topology | `StageConfig(name, process, factory, factory_args, next, route_fn, terminal, stream_to, …)`; process grouping by `process` name; `MultiProcessPipelineRunner` (419) drives real processes/queues | `config/schema.py`, `pipeline/mp_runner.py` |
| chat lowering | `_build_chat_generate_request`: `ChatCompletionRequest → GenerateRequest`; sampling fill-ins **temperature→1.0, top_p→1.0, top_k→-1, min_p→0.0, rep→1.0**; `modalities or ["text"]`; `audios`/`images`/`videos`/`audio` config → `metadata`; `stop` normalized; `seed` passthrough; stage_sampling passthrough | `serve/openai_api.py:904-1005` |
| explicit params | `_explicit_generation_params` records only `{max_new_tokens, temperature, top_p, top_k, repetition_penalty}` present in `model_fields_set` | `serve/openai_api.py:887-898` |
| audio content parts | `Message.content: str | list[Any]` passes parts through untouched; model-side consumption precedent for `input_audio`/`audio_url` parts exists (ming_omni preprocessor); no framework-level audio decode for chat | `serve/protocol.py:170-173`, `models/ming_omni/components/preprocessor.py:729` |
| audios[] | list of path/URL strings in `metadata["audios"]` (no per-turn binding imposed by framework — model responsibility) | `openai_api.py:951-1000` |
| GPU startup serialization | `gpu_startup_lock(logical_gpu_id)` file lock wraps factory construction on a GPU; **factories must not block inside it** | `utils/gpu_memory.py:272-296`, `pipeline/stage_workers.py:30` |
| AR infra boundary | `create_sglang_infrastructure` constructs ModelWorker + memory pools + initializes backend — no public "stop at model resolution" switch; a factory exception is a startup failure | P1 notes + `scheduling/bootstrap.py` |

## 2. Framework gaps found (candidate minimal patches)

**GAP-1 (confirmed): length-parameter explicit marking is broken.**
`_explicit_generation_params` checks `max_new_tokens`, but `ChatCompletionRequest`
exposes `max_tokens`/`max_completion_tokens` (`effective_max_tokens`, protocol.py:55-103).
Consequence: a user-specified length is indistinguishable from the endpoint
default. **Fix**: additive change to the field tuple (+ alias mapping) in
`serve/openai_api.py` + regression tests. Independent framework patch; model
work proceeds on the patched tree and records the dependency; Gate stays open
until the patch lands upstream or in our branch as a separate commit.

No other gaps: routing/terminal/registry/audio-parts mechanisms cover P2 needs.

## 3. Three-layer schema (frozen for T2.3)

### 3.1 HTTP layer (ChatCompletionRequest — framework-owned, unchanged)

- `messages: [{role, content}]`, role ∈ {system, user, assistant}; content = `str | list[parts]`; audio part = `{"type":"input_audio","input_audio":{"data":<base64>,"format":...}}`.
- `modalities`: missing/null/[] → `["text"]` (framework behavior, P2 沿用); accepted values exactly `["text"]` or `["audio"]`; both/duplicates/unknown → 400 at model validation.
- `audios: list[str]` (paths/URLs) supported with **strict binding rule** (3.2).
- V1 rejected capabilities: `stream=true`, image/video parts, request-level voice (`audio.voice`), `stage_sampling`/`stage_params`/talker_* params → 400 with "not supported in V1".
- Sampling: explicit fields marked via GAP-1 patch; endpoint fill-ins 1.0/1.0/-1/1.0 vs **model defaults 0.6/0.95/20/1.1** applied only to truly-unspecified fields.

### 3.2 Canonical request (model-owned dataclass, `payload_types.py`)

- turns: `[{role, kind: text|audio, text?|audio?(waveform,sr,source_key)}]`; per-turn single modality (mixed content in one turn → 400).
- audio binding: `input_audio` parts bind to their own turn (preferred form,
  required for mixed multi-turn). `audios[]` accepted only when the message
  list contains exactly one audio-needing turn per provided path, in order;
  ambiguity/count mismatch/duplicate provision (part + audios[]) → 400.
- audio representation: `(CPU float32 waveform (T,), original_sr)`; decode via
  soundfile from bytes/base64/file bytes; reject undecodable/non-finite/empty;
  duration cap (V1: 30 s/turn, 120 s/request — configurable in YAML), size cap
  50 MB, sample-rate range 8k–48k. URL sources fetched by the existing HTTP
  client path with timeout (no new download stack).
- `output_modality: text|audio`; `explicit_params: set[str]`;
  `effective_seed` (§5). P3 V1 rejects custom `stop` strings and nonzero `min_p` before codec/GPU work; generation stops on the frozen text-channel stop IDs or the requested length limit.

### 3.3 Canonical model input (grid — built in T2.3 processor)

- P0 contract §3 exactly: hand-built segments `<|im_start|>{role}\n…<|im_end|>\n`;
  audio turn = text-ch `[sosp, modality_pad×(n+1)]` + audio-ch `[audio_pad, codes…, eosp]`;
  assistant prefix text/audio; left-padding collate; grid `(B, L, 2)` at
  processor/generate boundary. The repo's `chat_template.jinja` is NOT used
  (reference does not use it). `(B,2,L)` conversion stays in the P3 runner.

## 4. Resampling responsibilities (P1-aligned, frozen)

- user-audio codes: orig-sr → 16 kHz (adapter encode with `(wav, orig_sr)`).
- voice prompt codes: orig-sr → 16 kHz; voice mel: orig-sr → 24 kHz; xvector:
  orig-sr → 24 kHz → 16 kHz. Decoded voice reference for enrollment is kept at
  original_sr and handed to the P1 adapter (no pre-resampling in stages).
- T2.4 smoke includes an in-memory-vs-P1-file numeric comparison for one voice.

## 5. Seed policy (corrects the P1 draft wording)

- `seed` explicit in HTTP → `effective_seed = seed` (never derived from request_id).
- absent → `effective_seed = blake2b(request_id)[:8]` (stable, not Python `hash()`).
- vocoder consumes `effective_seed` directly (save/set/restore inside the
  serial decode scope); AR derivation rule recorded for P3/P4. Same request_id
  retry reuses the stored effective seed.
- P1's "seed per request-id" phrasing is superseded by "per-request effective
  seed"; P1 raw experiment records are not rewritten (correction noted here).

## 6. Validation matrix & error placement

| check | trigger point (CPU, before any GPU) | error |
|---|---|---|
| roles/turn structure/single-modality per turn | model normalize (`request_builders`) | 400 `invalid_request` |
| modalities both/duplicate/unknown | model normalize | 400 |
| V1-unsupported capability (stream/images/video/voice/stage overrides) | model normalize | 400 `not_supported` |
| audio decode/finite/empty/size/duration/sr-range | preprocessing normalize | 400 `invalid_audio` |
| audios[] binding ambiguity | preprocessing normalize | 400 |
| max_tokens alias conflict/invalid values | HTTP layer (existing) + model re-check | 400 |
| post-encode context length vs limit | preprocessing after encode, **before AR admission** | 400 `context_too_long` (CPU pre-check uses conservative upper bound: chars→tokens + 12.5 Hz × duration) |
| model/checkpoint/codec asset missing | launch-time (config load) | startup failure |

GPU-resident stage start is not "touching GPU for a request"; the spy-based
no-GPU-call assertion applies to rejected requests only.

## 7. Differences vs P1 handoff drafts (recorded, not silent)

1. Voice delivery: per-request payload field (preprocessing→AR→vocoder, AR
   preserves) replaces P1's private startup-queue one-shot; vocoder may cache
   device copies keyed by content key. Startup-queue variant would hold the
   gpu_startup_lock — forbidden.
2. Seed: effective-seed policy above.
3. AR boundary: no `create_sglang_infrastructure` probing; formal factory
   raises a tagged `NotImplementedError` (P3 pointer) after arg/hf_config
   checks; test-only AR stub used for the multi-process smoke.


## P3 takeover correction (2026-09-08)

The formal AR factory now builds the native engine. The CPU-only preflight remains in `sglang-omni/sglang_omni/models/moss_speech/stages.py` (`validate_ar_preconditions`); it is not a serving-readiness test. Native acceptance and the historical P2 stub regression are reported separately in `sglang-omni/docs/design/moss_speech/p3/03_gate_report.md`.

Explicit sampling parameters retain their exact values. Absent parameters use the model defaults (0.6/0.95/20/1.1); explicit temperature=0 selects greedy. Nonfinite values and unsupported min_p/custom stop are rejected. The single-request wire grid is `(L,2)`, mask `(L,)`, and output grid `(L_new,2)`; the processor batch API remains `(B,L,2)`. The generated audio EOSP ends decoding only on an audio-mode row, not on an ignored audio value in text mode.
