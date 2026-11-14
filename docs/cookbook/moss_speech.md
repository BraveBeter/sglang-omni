# MOSS-Speech Flow Concurrency

MOSS-Speech supports non-streaming text or speech input with text or speech output. Speech output uses the request's reference audio. Its default vocoder decodes one request at a time.

## Enable Two Flow Slots

Set the vocoder factory option when launching your existing MOSS-Speech configuration:

```bash
sgl-omni serve --config moss_speech.yaml \
  --audio_vocoder.factory.flow_slots 2
```

The equivalent configuration patch is:

```yaml
stages:
  audio_vocoder:
    factory:
      flow_slots: 2
```

The default is `1`; supported values are `1` and `2`. Two slots require CUDA. This option does not change the AR engine's request capacity; set `ar_engine.engine.max_running_requests` separately for your workload.

Each slot runs a complete B1 Flow inference on a private CUDA stream with its own model and mutable caches. The second Flow copy consumes additional device memory. Requests retain their original FP32 Flow, ten denoising steps, conditioning, and sample count; this option does not combine requests into a model batch or introduce Codec CUDA Graphs.

Reference conditioning and HiFT remain serialized on one owner stream. HiFT uses the request seed, with Python, NumPy, and Torch RNG states restored after each call. CUDA events establish the handoff between the owner stream and the Flow streams. The vocoder freezes deterministic cuDNN execution while its workers are active and restores the preceding policy after draining them at shutdown.

The existing scheduler discards results from aborted requests; an already running Flow finishes before its slot is reused. Compute failures also release the slot. Text output bypasses audio decoding.

## Validate the Codec

The accelerator regression test compares concurrent PCM against the serial decoder using different reference voices, lengths, and seeds, then checks failed-slot reuse, RNG preservation, and shutdown during an active request:

```bash
MOSS_SPEECH_CODEC_PATH="$CODEC_PATH" python -m pytest -q \
  tests/unit_test/moss_speech/test_concurrent_codec.py
```
