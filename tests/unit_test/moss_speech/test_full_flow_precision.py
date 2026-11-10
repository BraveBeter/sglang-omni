# SPDX-License-Identifier: Apache-2.0
"""Native full-Flow BF16 solver, exact restoration and inference-owner lifetime."""

import threading
from contextlib import ExitStack

import pytest
import torch

from sglang_omni.models.moss_speech.components.full_flow_precision import (
    FullFlowBF16Scope,
)
from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.config import MossSpeechVocoderArgs
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import scoped_global_rng
from tests.unit_test.fixtures.moss_speech_codec import cfm_solver_inputs, tiny_full_flow


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
@pytest.mark.parametrize("streaming", [False, True])
@torch.no_grad()
def test_full_flow_bf16_native_solver_and_exact_owner_restoration(
    device: str, streaming: bool
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    with scoped_global_rng():
        flow = tiny_full_flow(torch.device(device))
        original = {
            name: parameter.detach().cpu().clone()
            for name, parameter in flow.named_parameters()
        }
        identities = {
            name: id(parameter) for name, parameter in flow.named_parameters()
        }
        inputs = cfm_solver_inputs(torch.device(device), mel_length=8)
        reference = flow.decoder.solve_euler(*inputs, streaming=streaming)
        codec = MossSpeechStreamingCodec.__new__(MossSpeechStreamingCodec)
        codec.flow = flow
        codec.flow_precision = "full_flow_bf16"
        codec.precision_context = ExitStack()
        codec.precision_scopes = ()
        codec.sessions = {}
        codec.lock = threading.Lock()
        codec.cfm_graph = None
        first: torch.Tensor | None = None
        random_state = torch.get_rng_state().clone()
        for _ in range(2):
            codec.activate_precision()
            scope = codec.precision_scopes[0]
            assert isinstance(scope, FullFlowBF16Scope)
            assert all(
                parameter.dtype is torch.bfloat16 for parameter in flow.parameters()
            )
            assert all(
                state.backup.device.type == "cpu"
                and state.backup.dtype is torch.float32
                for state in scope.parameters
            )
            assert scope.fp32_cpu_backup_bytes == 2 * scope.bf16_parameter_bytes > 0
            with pytest.raises(ValueError, match="idle unmodified FP32"):
                FullFlowBF16Scope(flow).__enter__()
            _, _, mu, mask, speaker_embeddings, conditioning = [
                value.bfloat16() for value in inputs
            ]
            actual = flow.decoder(
                mu,
                mask,
                10,
                spks=speaker_embeddings,
                cond=conditioning,
                streaming=streaming,
            )[0]
            repeated = flow.decoder(
                mu,
                mask,
                10,
                spks=speaker_embeddings,
                cond=conditioning,
                streaming=streaming,
            )[0]
            assert actual.dtype is torch.float32 and torch.isfinite(actual).all()
            assert torch.equal(actual, repeated)
            if first is not None:
                assert torch.equal(actual, first)
            else:
                first = actual
            assert flow.decoder.solver_precision_steps > 0
            assert flow.decoder.solver_state_dtypes == [torch.bfloat16] * 10
            assert (
                flow.decoder.estimator.time_embeddings.compute_dtype is torch.bfloat16
            )
            if device == "cuda":
                assert all(
                    processor.fallback_calls == 0 and processor.bf16_sdpa_calls > 0
                    for processor in scope.processors
                )
            else:
                pass
            codec.close()
            assert flow.inference_dtype is None and flow.decoder.solver_dtype is None
            assert flow.decoder.estimator.time_embeddings.compute_dtype is torch.float32
            assert torch.equal(
                reference, flow.decoder.solve_euler(*inputs, streaming=streaming)
            )
            for name, parameter in flow.named_parameters():
                assert (
                    id(parameter) == identities[name]
                    and parameter.dtype is torch.float32
                )
                assert torch.equal(parameter.detach().cpu(), original[name])
        assert torch.equal(random_state, torch.get_rng_state())


def test_full_flow_bf16_failed_entry_restores_all_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow = tiny_full_flow(torch.device("cpu"))
    original = {name: parameter.clone() for name, parameter in flow.named_parameters()}
    native = torch.Tensor.bfloat16
    calls = 0

    def interrupted(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("full precision entry failed")
        else:
            return native(tensor)

    monkeypatch.setattr(torch.Tensor, "bfloat16", interrupted)
    scope = FullFlowBF16Scope(flow)
    with pytest.raises(MemoryError, match="full precision entry failed"):
        scope.__enter__()
    assert not scope.active and not scope.parameters and not scope.processors
    assert flow.inference_dtype is None and flow.decoder.solver_dtype is None
    for name, parameter in flow.named_parameters():
        assert parameter.dtype is torch.float32 and torch.equal(
            parameter, original[name]
        )


def test_full_flow_bf16_rejects_training_and_missing_live_bias() -> None:
    flow = tiny_full_flow(torch.device("cpu"))
    flow.train()
    with pytest.raises(ValueError, match="idle unmodified FP32"):
        FullFlowBF16Scope(flow).__enter__()
    with pytest.raises(ValueError, match="live bias broadcasting"):
        MossSpeechVocoderArgs(flow_precision="full_flow_bf16")
    with pytest.raises(ValueError, match="excludes the FP32 CFM graph"):
        MossSpeechVocoderArgs(
            flow_precision="full_flow_bf16",
            broadcast_attention_bias=True,
            cfm_graph_mel_length=1500,
        )
