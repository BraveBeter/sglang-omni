# SPDX-License-Identifier: Apache-2.0
"""Joint complete-PCM and lifetime proof for shared HiFT with live bias."""

import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from typing import TypedDict
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from sglang_omni.models.moss_speech.components.codec_adapter import (
    MossSpeechCodecAdapter,
)
from sglang_omni.models.moss_speech.components.streaming_codec import (
    CodecStreamSession,
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.components.voice import VoiceConditioning
from sglang_omni.models.moss_speech.stages import create_audio_vocoder_executor
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import scoped_global_rng


class CumulativeCodecReference(TypedDict):
    report_sha256: str
    tensors_sha256: str
    pcm_atol: float
    pcm_rtol: float


class CumulativeCodecProof(TypedDict):
    optimizations: list[str]
    reference: CumulativeCodecReference
    joint_pcm_within_frozen_tolerance: bool
    code_ledger_exact: bool
    request_state_isolation_exact: bool
    rng_isolation_exact: bool
    cleanup_exact: bool
    executions: dict[str, int]
    comparison_scope: str
    tensor_artifact: str
    tensor_artifact_sha256: str
    stream_code_count: int
    stream_sample_count: int


@torch.inference_mode()
def assert_cumulative_codec(
    codec_path: str,
    reference_json: str,
    reference_tensors: str,
    optimizations: list[str],
    vocoder_options: dict[str, bool],
) -> CumulativeCodecProof:
    """Preserve fixed HF offline PCM and independently owned Torch stream behavior."""
    selected = set(optimizations) & {"codec-bias-broadcast", "codec-shared-hift"}
    assert selected == {"codec-bias-broadcast", "codec-shared-hift"}
    assert vocoder_options == {"broadcast_attention_bias": True}
    reference_path = Path(reference_json)
    tensors_path = Path(reference_tensors)
    reference = json.loads(reference_path.read_text())
    assert reference["status"] == "passed"
    assert (
        hashlib.sha256(tensors_path.read_bytes()).hexdigest()
        == reference["tensor_artifact_sha256"]
    )
    oracle = load_file(str(tensors_path), device="cpu")
    voice = VoiceConditioning(
        prompt_token=oracle["voice.prompt_token"].clone(),
        prompt_feat=oracle["voice.prompt_feat"].clone(),
        embedding=oracle["voice.embedding"].clone(),
        meta={},
    )
    codes = oracle["codes511.codes"][0].tolist()
    assert len(codes) == 511 and all(
        type(code) is int and 0 <= code < 16384 for code in codes
    )
    device = torch.device("cuda", torch.cuda.current_device())
    caller_random_state = (
        pickle.dumps(random.getstate(), protocol=5),
        pickle.dumps(np.random.get_state(), protocol=5),
        torch.get_rng_state().clone(),
        torch.cuda.get_rng_state(device).clone(),
    )
    shared = create_audio_vocoder_executor(
        "unused",
        chunk_size=25,
        codec_path=codec_path,
        device=device.type,
        gpu_id=device.index,
        broadcast_attention_bias=True,
    )
    independent: MossSpeechStreamingCodec | None = None
    artifacts: dict[str, torch.Tensor] = {}

    def decode_offline(
        adapter: MossSpeechCodecAdapter, code_values: list[int]
    ) -> torch.Tensor:
        with scoped_global_rng():
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            return adapter.decode(code_values, voice, request_id="offline")[1]

    def decode_stream(codec: MossSpeechStreamingCodec, request_id: str) -> torch.Tensor:
        codec.begin(request_id, voice.to_dict(), seed=42)
        session = codec.sessions[request_id]
        waveform = torch.cat(
            codec.push(request_id, codes[:36])
            + codec.push(request_id, codes[36:], final=True)
        )
        assert session.codes == codes and session.cursor == len(codes)
        assert session.emitted_samples == len(codes) * codec.samples_per_code
        assert waveform.numel() == len(codes) * codec.samples_per_code
        assert request_id not in codec.sessions
        return waveform

    try:
        independent = MossSpeechStreamingCodec(
            codec_path,
            chunk_size=25,
            device=device,
            broadcast_attention_bias=False,
        )
        decoder = shared.offline_codec.decoder
        assert decoder is not None
        assert shared.codec.hift is decoder.hift
        assert shared.offline_codec.decode_lock is shared.codec.lock
        assert shared.codec.hift is not independent.hift
        assert len(shared.codec.bias_broadcast_processors) == 56
        for name in reference["case_order"]:
            code_values = oracle[name + ".codes"][0].tolist()
            expected = oracle[name + ".pcm"].squeeze()
            actual = decode_offline(shared.offline_codec, code_values)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            artifacts[name + ".actual_offline_pcm"] = actual.clone().contiguous()
            artifacts[name + ".hf_pcm"] = expected.clone().contiguous()
        expected_stream = decode_stream(independent, "reference")
        artifacts["stream.reference_pcm"] = expected_stream.clone().contiguous()
        held_sessions: list[CodecStreamSession] = []
        first_chunks: list[list[torch.Tensor]] = []
        with patch.object(
            shared.codec.hift, "inference", wraps=shared.codec.hift.inference
        ) as hift_calls:
            for request_id in ("live-a", "live-b"):
                shared.codec.begin(request_id, voice.to_dict(), seed=42)
                held_sessions.append(shared.codec.sessions[request_id])
                first_chunks.append(shared.codec.push(request_id, codes[:36]))
            assert all(first_chunks) and set(shared.codec.sessions) == {
                "live-a",
                "live-b",
            }
            interleaved_offline = decode_offline(
                shared.offline_codec, oracle["codes32.codes"][0].tolist()
            )
            torch.testing.assert_close(
                interleaved_offline,
                oracle["codes32.pcm"].squeeze(),
                atol=1e-5,
                rtol=1e-5,
            )
            assert set(shared.codec.sessions) == {"live-a", "live-b"}
            artifacts["interleaved.offline_pcm"] = (
                interleaved_offline.clone().contiguous()
            )
            with patch.object(
                decoder.flow,
                "inference",
                Mock(side_effect=RuntimeError("Injected offline failure")),
            ):
                with pytest.raises(RuntimeError, match="Injected offline failure"):
                    decode_offline(shared.offline_codec, codes)
            assert not shared.offline_codec.sessions
            assert set(shared.codec.sessions) == {"live-a", "live-b"}
            shared.codec.begin("failed", voice.to_dict(), seed=42)
            with patch.object(
                shared.codec.flow,
                "inference",
                Mock(side_effect=RuntimeError("Injected stream failure")),
            ):
                with pytest.raises(RuntimeError, match="Injected stream failure"):
                    shared.codec.push("failed", codes, final=True)
            assert set(shared.codec.sessions) == {"live-a", "live-b"}
            for request_id, first, session in zip(
                ("live-a", "live-b"), first_chunks, held_sessions, strict=True
            ):
                actual = torch.cat(
                    first + shared.codec.push(request_id, codes[36:], final=True)
                )
                assert (
                    actual.dtype == expected_stream.dtype
                    and actual.shape == expected_stream.shape
                )
                torch.testing.assert_close(
                    actual, expected_stream, atol=1e-5, rtol=1e-5
                )
                assert session.codes == codes and session.cursor == len(codes)
                assert (
                    session.emitted_samples
                    == actual.numel()
                    == len(codes) * shared.codec.samples_per_code
                )
                artifacts[request_id + ".pcm"] = actual.clone().contiguous()
            shared_execution_count = hift_calls.call_count
        assert not shared.codec.sessions and not shared.offline_codec.sessions
        processors = shared.codec.bias_broadcast_processors
        assert all(
            processor.broadcast_calls > 0 and processor.last_head_stride == 0
            for processor in processors
        )
        bias_execution_count = sum(
            processor.broadcast_calls for processor in processors
        )
        shared.codec.close()
        actual_offline = decode_offline(shared.offline_codec, codes)
        torch.testing.assert_close(
            actual_offline, oracle["codes511.pcm"].squeeze(), atol=1e-5, rtol=1e-5
        )
        shared.offline_codec.close()
        actual_stream = decode_stream(shared.codec, "after-offline-close")
        torch.testing.assert_close(actual_stream, expected_stream, atol=1e-5, rtol=1e-5)
        assert not shared.codec.sessions and not shared.offline_codec.sessions
        assert pickle.dumps(random.getstate(), protocol=5) == caller_random_state[0]
        assert pickle.dumps(np.random.get_state(), protocol=5) == caller_random_state[1]
        assert torch.equal(torch.get_rng_state(), caller_random_state[2])
        assert torch.equal(torch.cuda.get_rng_state(device), caller_random_state[3])
        output_dir = Path(os.environ["MOSS_BIAS_BROADCAST_REPORT"]).parent
        artifacts["stream.codes"] = oracle["codes511.codes"].clone().contiguous()
        for name, value in voice.to_dict().items():
            artifacts["voice." + name] = value.clone().contiguous()
        artifact_path = output_dir / "cumulative-codec.safetensors"
        save_file(artifacts, str(artifact_path))
        return {
            "optimizations": sorted(selected),
            "reference": {
                "report_sha256": hashlib.sha256(
                    reference_path.read_bytes()
                ).hexdigest(),
                "tensors_sha256": hashlib.sha256(tensors_path.read_bytes()).hexdigest(),
                "pcm_atol": 1e-5,
                "pcm_rtol": 1e-5,
            },
            "joint_pcm_within_frozen_tolerance": True,
            "code_ledger_exact": True,
            "request_state_isolation_exact": True,
            "rng_isolation_exact": True,
            "cleanup_exact": True,
            "executions": {
                "codec-bias-broadcast": bias_execution_count,
                "codec-shared-hift": shared_execution_count,
            },
            "comparison_scope": "All four full offline cases against frozen independent HF PCM; complete511-code streaming against independently owned unoptimized Torch, with the same pinned voice and code ledger.",
            "tensor_artifact": str(artifact_path),
            "tensor_artifact_sha256": hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest(),
            "stream_code_count": len(codes),
            "stream_sample_count": expected_stream.numel(),
        }
    finally:
        shared.stop()
        if independent is not None:
            independent.close()
        else:
            pass
