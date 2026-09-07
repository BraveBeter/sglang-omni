# P2-02 Skeleton Notes (T2.2)

## Files

- `__init__.py`: `CAPABILITIES` (all False, V1) — import-light, no CUDA/codec at discovery.
- `config.py`: `MossSpeechPipelineConfig` → `EntryClass`; 4 stages:
  `preprocessing`(process=preproc, gpu=0, next=ar_engine) →
  `ar_engine`(process=ar, gpu=0, next=[text_decode, audio_vocoder], route_fn=`request_builders.resolve_output_terminal`) →
  `text_decode`(process=text_out, CPU, terminal) +
  `audio_vocoder`(process=vocoder, gpu=0, terminal).
  Three GPU stages in separate processes per P1 D-3 (no process_local_edges).
- `hf_config.py`: locked-revision port of `MossSpeechConfig`; deviations:
  transformers-5 compat shims for `rope_config_validation` /
  `layer_type_validation` (checkpoint uses defaults → identical outcomes);
  registration via `AutoConfig.register(..., exist_ok=True)` (5.x removed
  `_CONFIG_MAPPING`); `num_hidden_layers=36` preserved verbatim — 40-layer KV
  accounting stays a P3 memory-pool concern (config not falsified).
- `payload_types.py`: `MossSpeechState(DeclarativeStateBase)` with wire codecs
  for all cross-process fields (turns/codes/grid/voice tensors/effective_seed/
  explicit_params); device tensors only at consumption points.

## Registry timing

`registry` imports `models.moss_speech.config` → package `__init__` runs
first (CAPABILITIES only). `hf_config` registration happens when
`stages.py`/factories import it (and in the launch_boundary script before
any checkpoint parse). Verified: fresh subprocess discovers
`MossSpeechPipelineConfig` with no CUDA/codec modules loaded
(`test_model_package.py::test_registry_discovers_entry_class_in_fresh_process`).

## StageConfig field notes (framework `config/schema.py`)

- `process`: arbitrary grouping key → one worker process per distinct name.
- `gpu`: logical GPU index for placement; startup serialized by
  `gpu_startup_lock` — factories must not block inside it (voice travels in
  request payload, not startup messages; P2 contract §7.1).
- `route_fn`: `fn(request_id, output)` → exactly one of `next` (validated).
- terminal stages reject `route_fn`.

## Test evidence

`tests/unit_test/moss_speech/test_model_package.py` (3 passed):
1. CAPABILITIES six-flag all-False assertions.
2. Fresh-process discovery via `PIPELINE_CONFIG_REGISTRY`.
3. `AutoConfig.from_pretrained(<locked dir>, trust_remote_code=False)` →
   `MossSpeechConfig` with all P0-recorded fields (36 layers, dual vocab,
   token ids, `channels=2`, `audio_pad=512`).

Set `MOSS_SPEECH_MODEL_DIR` to run the checkpoint-parse check.
