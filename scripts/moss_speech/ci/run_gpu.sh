#!/usr/bin/env bash
# Run inside an allocated GPU job with all externally staged assets offline.
set -euo pipefail
: "${MOSS_SPEECH_MODEL_DIR:?Set the local AR checkpoint directory}"
: "${MOSS_SPEECH_CODEC_DIR:?Set the local codec checkpoint directory}"
: "${MOSS_SPEECH_VOICE_WAV:?Set the configured default voice WAV}"
: "${MOSS_SPEECH_MANIFEST:?Set the frozen qualification manifest}"
: "${MOSS_SPEECH_REFERENCE:?Set the independent HF reference report}"
: "${MOSS_SPEECH_CI_OUTPUT:?Set a fresh output directory}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
# Keep user-supplied relative asset paths relative to the invocation directory.
python "$repo_root/scripts/moss_speech/p5/make_config.py" \
  --model-path "$MOSS_SPEECH_MODEL_DIR" --codec-path "$MOSS_SPEECH_CODEC_DIR" \
  --voice-wav "$MOSS_SPEECH_VOICE_WAV" --runtime-dir "${MOSS_SPEECH_CI_OUTPUT}_runtime" \
  --output "${MOSS_SPEECH_CI_OUTPUT}.yaml"
python "$repo_root/scripts/moss_speech/p5/native.py" \
  --manifest "$MOSS_SPEECH_MANIFEST" --reference "$MOSS_SPEECH_REFERENCE" \
  --config "${MOSS_SPEECH_CI_OUTPUT}.yaml" --out-dir "$MOSS_SPEECH_CI_OUTPUT"
