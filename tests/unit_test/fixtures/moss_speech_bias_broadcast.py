# SPDX-License-Identifier: Apache-2.0
"""Trained complete-Codec qualification for live attention bias broadcasting."""

import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict

import pytest
import torch
from safetensors.torch import load_file

from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)


class BiasBroadcastStreamCase(TypedDict):
    code_count: int
    pcm_sha256: str
    intermediate_hashes: list[str]


class BiasBroadcastStreamingReport(TypedDict):
    status: Literal["passed"]
    code_profile_sha256: str
    golden_tensors_sha256: str
    flow_sha256: str
    hift_sha256: str
    installed_processors: int
    broadcast_calls: int
    fallback_calls: int
    request_isolation: bool
    failure_isolation: bool
    cases: list[BiasBroadcastStreamCase]


@torch.inference_mode()
def qualify_bias_broadcast_streaming(
    codec_path: Path,
    code_profile_path: Path,
    golden_tensors_path: Path,
    output_dir: Path,
) -> BiasBroadcastStreamingReport:
    """Compare trained production paths and preserve explicit input/output hashes."""
    profile = json.loads(code_profile_path.read_text())
    base_codes = profile["base_en_codes"]
    assert len(base_codes) >= 128 and all(type(code) is int for code in base_codes)
    golden = load_file(str(golden_tensors_path), device="cpu")
    voice = {
        name: golden[f"voice.{name}"]
        for name in ("prompt_token", "prompt_feat", "embedding")
    }
    codecs = {
        name: MossSpeechStreamingCodec(
            codec_path, chunk_size=25, device="cuda:0", broadcast_attention_bias=enabled
        )
        for name, enabled in (("base", False), ("broadcast", True))
    }
    assert not codecs["base"].bias_broadcast_processors
    assert len(codecs["broadcast"].bias_broadcast_processors) == 56

    def tensor_digest(tensor: torch.Tensor) -> str:
        return hashlib.sha256(
            tensor.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()

    parameters_before = {
        name: {
            parameter_name: (id(parameter), tensor_digest(parameter))
            for parameter_name, parameter in codec.flow.named_parameters()
        }
        for name, codec in codecs.items()
    }
    references: dict[
        int, tuple[str, list[dict[str, int | bool]], str, str, list[str]]
    ] = {}
    cases: list[BiasBroadcastStreamCase] = []

    def run_stream(
        codec: MossSpeechStreamingCodec, code_count: int
    ) -> tuple[str, list[dict[str, int | bool]], str, str, list[str]]:
        intermediate_hashes: list[str] = []
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        original_flow = codec.flow.inference
        original_hift = codec.hift.inference

        def observe_estimator(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            intermediate_hashes.append(f"estimator:{tensor_digest(output)}")

        def observe_flow(
            token: torch.Tensor,
            token_len: torch.Tensor,
            prompt_token: torch.Tensor,
            prompt_token_len: torch.Tensor,
            prompt_feat: torch.Tensor,
            prompt_feat_len: torch.Tensor,
            embedding: torch.Tensor,
            streaming: bool,
            finalize: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = original_flow(
                token,
                token_len,
                prompt_token,
                prompt_token_len,
                prompt_feat,
                prompt_feat_len,
                embedding,
                streaming,
                finalize,
            )
            intermediate_hashes.extend(
                f"flow:{tensor_digest(tensor)}" for tensor in output
            )
            return output

        def observe_hift(
            *, speech_feat: torch.Tensor, cache_source: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = original_hift(speech_feat=speech_feat, cache_source=cache_source)
            intermediate_hashes.extend(
                f"hift:{tensor_digest(tensor)}" for tensor in output
            )
            return output

        observer = codec.flow.decoder.estimator.register_forward_hook(observe_estimator)
        codec.flow.inference = observe_flow
        codec.hift.inference = observe_hift
        codec.begin("proof", voice, seed=42)
        session = codec.sessions["proof"]
        pcm: list[torch.Tensor] = []
        try:
            for code in base_codes[:code_count]:
                pcm.extend(codec.push("proof", [code]))
            pcm.extend(codec.push("proof", [], final=True))
            waveform = torch.cat(pcm)
            assert waveform.numel() == code_count * codec.samples_per_code
            assert torch.isfinite(waveform).all() and not codec.sessions
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
            assert session.device_rng is not None
            return (
                tensor_digest(waveform),
                session.ledger.copy(),
                tensor_digest(session.cpu_rng),
                tensor_digest(session.device_rng),
                intermediate_hashes,
            )
        finally:
            observer.remove()
            codec.flow.inference = original_flow
            codec.hift.inference = original_hift
            codec.cleanup("proof")

    try:
        for code_count in (36, 128):
            reference = run_stream(codecs["base"], code_count)
            references[code_count] = reference
            for _ in range(2):
                assert run_stream(codecs["broadcast"], code_count) == reference
                assert run_stream(codecs["base"], code_count) == reference
            cases.append(
                {
                    "code_count": code_count,
                    "pcm_sha256": reference[0],
                    "intermediate_hashes": reference[4],
                }
            )
        codec = codecs["broadcast"]
        codec.begin("aborted", voice, seed=42)
        codec.begin("survivor", voice, seed=42)
        codec.push("aborted", base_codes[:36])
        codec.cleanup("aborted")
        assert "survivor" in codec.sessions
        survivor_pcm = codec.push("survivor", base_codes[:36])
        survivor_pcm.extend(codec.push("survivor", [], final=True))
        held_waveform = torch.cat(survivor_pcm)
        assert tensor_digest(held_waveform) == references[36][0]
        codec.begin("failed", voice, seed=42)
        codec.begin("survivor", voice, seed=42)
        original_hift = codec.hift.inference

        def fail_hift(
            *, speech_feat: torch.Tensor, cache_source: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            raise RuntimeError("Injected HiFT failure")

        codec.hift.inference = fail_hift
        try:
            with pytest.raises(RuntimeError, match="Injected HiFT failure"):
                codec.push("failed", base_codes[:36])
        finally:
            codec.hift.inference = original_hift
        assert "failed" not in codec.sessions and "survivor" in codec.sessions
        survivor_pcm = codec.push("survivor", base_codes[:36])
        survivor_pcm.extend(codec.push("survivor", [], final=True))
        assert tensor_digest(torch.cat(survivor_pcm)) == references[36][0]
        assert tensor_digest(held_waveform) == references[36][0]
        codec.begin("closed", voice, seed=42)
        codec.close()
        codec.close()
        assert not codec.sessions
        assert parameters_before == {
            name: {
                parameter_name: (id(parameter), tensor_digest(parameter))
                for parameter_name, parameter in codec.flow.named_parameters()
            }
            for name, codec in codecs.items()
        }
        processors = codec.bias_broadcast_processors
        assert all(
            processor.broadcast_calls > 0 and processor.last_head_stride == 0
            for processor in processors
        )
        report: BiasBroadcastStreamingReport = {
            "status": "passed",
            "code_profile_sha256": hashlib.sha256(
                code_profile_path.read_bytes()
            ).hexdigest(),
            "golden_tensors_sha256": hashlib.sha256(
                golden_tensors_path.read_bytes()
            ).hexdigest(),
            "flow_sha256": hashlib.sha256(
                (codec_path / "flow/flow-chunk-25.pt").read_bytes()
            ).hexdigest(),
            "hift_sha256": hashlib.sha256(
                (codec_path / "flow/hift.pt").read_bytes()
            ).hexdigest(),
            "installed_processors": len(processors),
            "broadcast_calls": sum(
                processor.broadcast_calls for processor in processors
            ),
            "fallback_calls": sum(processor.fallback_calls for processor in processors),
            "request_isolation": True,
            "failure_isolation": True,
            "cases": cases,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "bias-broadcast-streaming.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report
    finally:
        for codec in codecs.values():
            codec.close()
