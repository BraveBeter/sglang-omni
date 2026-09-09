#!/usr/bin/env bash
# Asset-free contracts. Native model-layer and local golden tests run separately.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q \
  tests/unit_test/moss_speech/test_request_contract.py \
  tests/unit_test/moss_speech/test_http_contract.py \
  tests/unit_test/moss_speech/test_codec_adapter_cpu.py \
  tests/unit_test/moss_speech/test_parity_metrics_cpu.py \
  tests/unit_test/moss_speech/test_p5_qualification.py \
  tests/unit_test/moss_speech/test_streaming_codec.py \
  tests/unit_test/moss_speech/test_streaming_pipeline.py \
  tests/unit_test/moss_speech/test_cli_launch.py \
  tests/unit_test/pipeline/test_config_discovery.py \
  tests/unit_test/serve/test_chat_stream_errors.py \
  tests/unit_test/serve/test_chat_lifecycle.py \
  tests/unit_test/serve/test_explicit_generation_params.py
