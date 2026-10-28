# MOSS-Speech

This pipeline implements native dual-channel generation for text-to-text, text-to-speech, speech-to-text, and speech-to-speech requests. Text and audio terminals handle streaming and non-streaming requests. Audio output uses one configured reference voice; per-request voice cloning is not supported.

The implementation uses eager execution, a request-owned sampling generator, and the MOSS-Speech codec's WhisperVQ encoder, flow decoder, and HiFT vocoder. The checkpoint has 32 shared layers and four layers in each modality tail: its HF configuration describes 36 sequential layers, while paged attention needs 40 distinct KV slots.

## Configuration

Download the MOSS-Speech and MOSS-Speech-Codec checkpoints and a reference WAV, then edit the asset paths in the example configuration. Assets are local directories; the AR tokenizer and configuration load without remote Python code.

Install this model's additional dependency with uv pip install -e '.[moss-speech]'. The base runtime supplies ONNX Runtime and Diffusers; a second ONNX Runtime distribution is unnecessary.

Activate the installed environment before launch so standard SGLang CUDA kernel compilation can find its ninja executable. These runtime kernels may compile on the first request.

```bash
sgl-omni serve --config examples/configs/moss_speech.yaml
```

The default placement assigns preprocessing, AR, and audio decoding to GPU 0 in separate processes. The two-GPU example assigns AR to GPU 0 and the codec stages to GPU 1:

```bash
sgl-omni serve --config examples/configs/moss_speech_dual_gpu.yaml
```

These examples configure stage placement. Use the real-checkpoint smoke below to validate startup, output contracts, and recovery on the target GPUs. Measure deployment capacity and performance separately with fixed requests and output budgets.

Set deployment values through typed stage configuration, for example preprocessing.factory.voice_wav, preprocessing.factory.codec_path, audio_vocoder.factory.codec_path, and ar_engine.engine.max_running_requests. The streaming codec uses chunk_size=25, declared once in the vocoder factory configuration. Optional optimization flags from the experimental branch are not part of this pipeline.

## Validation status

This migration is a work in progress. Structural and CPU checks do not establish GPU correctness, audio quality, latency, throughput, or a release-ready package. Third-party source provenance and distribution contents must be checked before publication.

## Output and comparison contract

Each generated row contains a text token and an audio token. A modality-pad text token selects the audio channel; text-channel end-of-text and message-end tokens terminate generation. EOSP closes the audio segment and returns the finite-state machine to text mode; EOSP in an inactive channel does not terminate the request.

Audio decoding keeps every valid active-channel code before the first EOSP. A 512-row audio segment containing EOSP has 511 valid codes; a length-limited segment with 512 valid codes retains all 512. Completion usage counts generated grid rows, including delimiters. Compare both the raw grid and extracted codes, and keep the reported stop or length reason. Do not drop a final valid code to make another decoder agree.

The streaming codec emits 1,920 samples per retained code at 24 kHz. Its final hop flushes held overlap; the sum of emitted chunk samples must equal the final sample ledger. Streaming results contain usage and ledger metadata and do not replay audio already sent in deltas. The complete-utterance and chunk-trained flow checkpoints are different decoder profiles, so waveform equality between those profiles is not assumed.

## Tests

The unit suite is self-contained and generates tiny model and codec weights locally. Its checked-in processor fixture contains complete input codes and frozen grid provenance; it does not require downloaded model caches or workspace audit files.

```bash
SGLANG_USE_CPU_ENGINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests/unit_test/moss_speech -m 'not accelerator'
```

Set the official CPU platform flag before importing SGLang. GPU cases are selected with -m accelerator and require a supported allocated GPU; collecting or skipping them does not establish a GPU validation result. Full checkpoint and tokenizer parity, four-mode service execution, quality, and performance remain separate integration checks.

The real-checkpoint service smoke consumes a resolved single- or two-GPU configuration and pinned English/Chinese speech inputs. Prepare its assets on a machine with network access, then run inside an allocated GPU environment:

```bash
python .github/scripts/prepare_moss_speech_ci.py \
  --output-directory /path/to/moss-smoke --gpu-count 1
export MOSS_SPEECH_CI_CONFIG=/path/to/moss-smoke/moss_speech.yaml
export MOSS_SPEECH_CI_AUDIO_EN=/path/to/moss-smoke/input-en.wav
export MOSS_SPEECH_CI_AUDIO_ZH=/path/to/moss-smoke/input-cn.wav
export MOSS_SPEECH_CI_OUTPUT_DIR=/path/to/moss-smoke/results
python -m pytest tests/test_model/test_moss_speech_ci.py -m accelerator -v -s -x
```

The prepared speech inputs retain the first 20 seconds of each pinned reference recording, with source hashes and selected frame ranges recorded separately. The configured synthesis voice uses the complete Chinese reference WAV. The fixture rejects invalid or overlong input recordings before starting a server.

The smoke covers both languages, all four input/output modes, streaming and non-streaming, natural text EOS, audio length truncation, disconnect recovery, and invalid-request recovery. It retains HTTP responses, generated WAVs, usage, timing observations, asset hashes, and server logs. A first PCM event can contain silence; non-streaming TTFA and TTFT are unobserved. These observations have no calibrated performance or quality thresholds, and the chat API does not expose the native token grid for exact retained-code comparison.

The central Omni workflow selects this dedicated smoke with the run-omni-ci and run-moss-speech labels. It does not add MOSS-Speech to the random model selection. Local checks and workflow wiring do not establish a successful remote CI run.
