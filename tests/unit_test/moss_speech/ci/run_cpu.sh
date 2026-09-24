#!/usr/bin/env bash
# Asset-free contracts. Native model-layer and local golden tests run separately.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q \
  tests/unit_test/moss_speech/test_preprocessing_lifetime.py \
  tests/unit_test/serve/test_chat_preflight.py \
  tests/unit_test/moss_speech/test_request_contract.py \
  tests/unit_test/moss_speech/test_http_contract.py \
  tests/unit_test/moss_speech/test_speech_endpoints.py \
  tests/unit_test/serve/test_speech_errors.py \
  tests/unit_test/moss_speech/test_codec_adapter_cpu.py \
  tests/unit_test/moss_speech/test_parity_metrics_cpu.py \
  tests/unit_test/moss_speech/test_streaming_codec.py \
  tests/unit_test/moss_speech/test_streaming_pipeline.py \
  tests/unit_test/moss_speech/test_voice_transport.py \
  tests/unit_test/moss_speech/test_small_first_config.py \
  tests/unit_test/moss_speech/test_cfm_prefix_cache.py \
  tests/unit_test/moss_speech/test_cfm_prefix_manifest.py \
  tests/unit_test/moss_speech/test_cfm_prefix_bank.py \
  tests/unit_test/moss_speech/test_cfm_prefix_lifetime.py \
  tests/unit_test/moss_speech/test_stream_decode_fairness.py \
  tests/unit_test/moss_speech/test_startup_schedule.py \
  tests/unit_test/moss_speech/test_codec_launch_profile.py \
  tests/unit_test/scheduling/test_bounded_streaming_vocoder.py \
  tests/unit_test/moss_speech/test_cfm_segmented_attention.py \
  tests/unit_test/moss_speech/test_cli_launch.py \
  tests/unit_test/pipeline/test_config_discovery.py \
  tests/unit_test/serve/test_chat_stream_errors.py \
  tests/unit_test/serve/test_chat_lifecycle.py \
  tests/unit_test/serve/test_explicit_generation_params.py
