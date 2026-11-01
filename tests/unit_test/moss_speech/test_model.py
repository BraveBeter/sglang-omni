# SPDX-License-Identifier: Apache-2.0
"""Native weight-loading, numerical, and dual-tail attention contracts."""

import os
from pathlib import Path

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.scheduler import GenerationBatchResult
from transformers import Qwen3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_speech.hf_config import (
    MossSpeechConfig,
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.model_layers import (
    MossSpeechRMSNorm,
    reference_rope_frequencies,
    request_linear,
)
from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.models.moss_speech.request_builders import (
    make_moss_speech_scheduler_adapters,
)
from sglang_omni.models.moss_speech.request_data import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.request_lifecycle import MossSpeechRequestLifecycle
from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.fixtures.moss_speech_model import (
    RowCommitRunner,
    checkpoint_weights,
    decode,
    native_runtime,
    prefill,
    tiny_model,
)

NATIVE_CPU = pytest.mark.skipif(
    os.environ.get("SGLANG_USE_CPU_ENGINE") != "1",
    reason="Native CPU construction requires SGLANG_USE_CPU_ENGINE=1 before imports",
)


@NATIVE_CPU
def test_actual_worker_allocates_both_modality_tails_without_rewriting_hf_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = MossSpeechConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=256,
        audio_vocab_size=64,
        max_position_embeddings=16,
        architectures=["MossSpeechForCausalLM"],
    )
    configuration.save_pretrained(tmp_path)
    monkeypatch.setitem(
        CONFIG_MAPPING._extra_content, "moss_speech", MossSpeechConfig
    )  # noqa: leading-underscore  # Transformers registry spelling.
    ensure_moss_speech_config_registered()
    worker_configuration = ModelConfig(
        str(tmp_path), trust_remote_code=False, dtype="bfloat16"
    )
    ModelWorker.apply_arch_override(worker_configuration, "MossSpeechForCausalLM")
    assert worker_configuration.num_attention_layers == 40
    assert worker_configuration.num_hidden_layers == 36
    assert worker_configuration.hf_config.num_hidden_layers == 36
    with native_runtime(torch.device("cpu")):
        model = MossSpeechSGLangModel(configuration)
        weights = checkpoint_weights(configuration)
        assert len(weights) == 446
        model.load_weights(weights)
        assert model.attention_layer_ids() == list(range(40))


@NATIVE_CPU
def test_checkpoint_sentinels_reach_every_independent_parameter() -> None:
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu"))
        weights = dict(checkpoint_weights(model.config))
        model.load_weights(weights.items())
        for parameter, source in (
            (model.embed_tokens.weight, "model.embed_tokens.weight"),
            (model.audio_embed.weight, "model.audio_embed.weight"),
            (model.text_norm.weight, "model.text_norm.weight"),
            (model.audio_norm.weight, "model.audio_norm.weight"),
            (model.text_lm_head.weight, "text_lm_head.weight"),
            (model.audio_lm_head.weight, "audio_lm_head.weight"),
        ):
            torch.testing.assert_close(parameter, weights[source], rtol=0, atol=0)
        for layers, source_block in (
            (model.layers, "shared_block"),
            (model.text_block, "text_block"),
            (model.audio_block, "audio_block"),
        ):
            for layer_index, layer in enumerate(layers):
                prefix = f"model.{source_block}.layers.{layer_index}."
                expected_qkv = torch.cat(
                    [
                        weights[prefix + f"self_attn.{projection}_proj.weight"]
                        for projection in ("q", "k", "v")
                    ]
                )
                expected_gate_up = torch.cat(
                    [
                        weights[prefix + f"mlp.{projection}_proj.weight"]
                        for projection in ("gate", "up")
                    ]
                )
                torch.testing.assert_close(
                    layer.self_attn.qkv_proj.weight, expected_qkv, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    layer.mlp.gate_up_proj.weight, expected_gate_up, rtol=0, atol=0
                )
                for parameter, leaf in (
                    (layer.self_attn.o_proj.weight, "self_attn.o_proj.weight"),
                    (layer.self_attn.q_norm.weight, "self_attn.q_norm.weight"),
                    (layer.self_attn.k_norm.weight, "self_attn.k_norm.weight"),
                    (layer.mlp.down_proj.weight, "mlp.down_proj.weight"),
                    (layer.input_layernorm.weight, "input_layernorm.weight"),
                    (
                        layer.post_attention_layernorm.weight,
                        "post_attention_layernorm.weight",
                    ),
                ):
                    torch.testing.assert_close(
                        parameter, weights[prefix + leaf], rtol=0, atol=0
                    )


@NATIVE_CPU
@pytest.mark.parametrize(
    "defect,message",
    [
        ("missing", "cover"),
        ("unexpected", "Unexpected"),
        ("duplicate", "Duplicate"),
        ("broadcast", "shape"),
        ("shape", "shape"),
    ],
)
def test_incomplete_duplicate_and_malformed_checkpoints_fail(
    defect: str, message: str
) -> None:
    with native_runtime(torch.device("cpu")):
        model = tiny_model(torch.device("cpu"))
        weights = checkpoint_weights(model.config)
        if defect == "missing":
            weights.pop()
        elif defect == "unexpected":
            weights.append(("model.lm_head.weight", torch.zeros(1)))
        elif defect == "duplicate":
            weights.append(weights[0])
        elif defect == "broadcast":
            source_name, tensor = weights[6]
            weights[6] = (source_name, tensor[:1])
        else:
            source_name, tensor = weights[0]
            weights[0] = (
                source_name,
                torch.zeros(tuple(length + 3 for length in tensor.shape)),
            )
        with pytest.raises(RuntimeError, match=message):
            model.load_weights(weights)


@NATIVE_CPU
@pytest.mark.parametrize("channel", ["text", "audio"])
def test_equal_embedding_and_head_weights_are_rejected(channel: str) -> None:
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu"))
        model.load_weights(checkpoint_weights(model.config))
        model.assert_heads_independent()
        embedding, head = (
            (model.embed_tokens, model.text_lm_head)
            if channel == "text"
            else (model.audio_embed, model.audio_lm_head)
        )
        head.weight.copy_(embedding.weight)
        with pytest.raises(RuntimeError, match="independent"):
            model.assert_heads_independent()


def test_bf16_normalization_rounds_residual_and_normalized_value() -> None:
    """Combining FP32 residual addition or weight multiplication changes checkpoint rounding."""
    generator = torch.Generator().manual_seed(921)
    hidden_states = torch.randn(3, 128, generator=generator).bfloat16()
    residual = torch.randn(3, 128, generator=generator).bfloat16()
    original_hidden = hidden_states.clone()
    with torch.random.fork_rng(devices=[]):
        normalization = MossSpeechRMSNorm(128, eps=1e-6).bfloat16()
    with torch.no_grad():
        normalization.weight.copy_(torch.randn(128, generator=generator).bfloat16())
    rounded_residual = hidden_states + residual
    reference = Qwen3RMSNorm(128, eps=1e-6).bfloat16()
    with torch.no_grad():
        reference.weight.copy_(normalization.weight)
    expected = reference(rounded_residual)
    actual, actual_residual = normalization(hidden_states, residual)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_residual, rounded_residual)
    assert torch.equal(hidden_states, original_hidden)


def test_projection_preserves_each_request_gemm_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(78)
    hidden_states = torch.randn(7, 128, generator=generator).bfloat16()
    weight = torch.randn(32, 128, generator=generator).bfloat16()
    linear = torch.nn.functional.linear
    expected = torch.cat(
        [linear(hidden_states[:2], weight), linear(hidden_states[2:], weight)]
    )
    token_shapes: list[tuple[int, ...]] = []

    def record_linear(
        request_hidden: torch.Tensor, request_weight: torch.Tensor
    ) -> torch.Tensor:
        token_shapes.append(tuple(request_hidden.shape))
        return linear(request_hidden, request_weight)

    monkeypatch.setattr(torch.nn.functional, "linear", record_linear)
    actual = request_linear(hidden_states, weight, [2, 5])
    assert token_shapes == [(2, 128), (5, 128)]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("head_dim,theta", [(8, 10000.0), (128, 1_000_000.0)])
def test_rope_frequencies_keep_cpu_float32_reference_values(
    head_dim: int, theta: float
) -> None:
    frequencies = reference_rope_frequencies(head_dim, theta)
    assert frequencies.device.type == "cpu"
    assert frequencies.dtype == torch.float32
    with torch.device("cpu"):
        reference = Qwen3RotaryEmbedding(
            Qwen3Config(head_dim=head_dim, rope_theta=theta)
        )
    torch.testing.assert_close(frequencies, reference.inv_freq, rtol=0, atol=0)


def assert_native_request_isolation(device: torch.device) -> None:
    with native_runtime(device), torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(183)
        model = tiny_model(device)
        prompt_embeddings = [
            torch.randn((length, 128), generator=generator, device=device)
            for length in (2, 3)
        ]
        feedback_embeddings = torch.randn((2, 128), generator=generator, device=device)
        packed_prefill, packed_backend = prefill(
            model, prompt_embeddings, ["first", "second"]
        )
        packed_decode = decode(
            model, packed_backend, [2, 3], feedback_embeddings, ["first", "second"]
        )
        for request_index, prompt in enumerate(prompt_embeddings):
            isolated_prefill, isolated_backend = prefill(model, [prompt], ["isolated"])
            isolated_decode = decode(
                model,
                isolated_backend,
                [len(prompt)],
                feedback_embeddings[request_index : request_index + 1],
                ["isolated"],
            )
            for packed, isolated in (
                (packed_prefill.text_logits, isolated_prefill.text_logits),
                (packed_prefill.audio_logits, isolated_prefill.audio_logits),
                (packed_decode.text_logits, isolated_decode.text_logits),
                (packed_decode.audio_logits, isolated_decode.audio_logits),
            ):
                assert packed is not None and isolated is not None
                torch.testing.assert_close(
                    packed[request_index], isolated[0], rtol=0, atol=0
                )
        first_tail = model.config.num_shared_layers
        second_tail = first_tail + model.config.num_modality_layers
        text_keys = packed_backend.token_to_kv_pool.get_key_buffer(first_tail)
        audio_keys = packed_backend.token_to_kv_pool.get_key_buffer(second_tail)
        assert text_keys.data_ptr() != audio_keys.data_ptr()
        assert torch.count_nonzero(text_keys[1:8]) > 0
        assert torch.count_nonzero(audio_keys[1:8]) > 0
        assert not torch.equal(text_keys[1:8], audio_keys[1:8])


@NATIVE_CPU
def test_native_cpu_prefill_decode_keep_requests_and_tail_kv_independent() -> None:
    assert_native_request_isolation(torch.device("cpu"))


@pytest.mark.accelerator
def test_native_cuda_prefill_decode_keep_requests_and_tail_kv_independent() -> None:
    require_cuda()
    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        pytest.skip("Run the accelerator lane without the CPU engine override")
    else:
        assert_native_request_isolation(torch.device("cuda", 0))


@pytest.mark.accelerator
def test_mixed_output_cuda_batch_preserves_complete_heads_and_kv(
    caplog: pytest.LogCaptureFixture,
) -> None:
    require_cuda()
    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        pytest.skip("Run the accelerator lane without the CPU engine override")
    else:
        pass
    device = torch.device("cuda", 0)
    with native_runtime(device), torch.no_grad():
        model = tiny_model(device, audio_vocab_size=1024)
        assert all(parameter.device == device for parameter in model.parameters())
        runner = RowCommitRunner(model)
        lifecycle = MossSpeechRequestLifecycle(model, mixed_modalities=True)
        build, _ = make_moss_speech_scheduler_adapters(lifecycle=lifecycle)
        requests = []
        for request_id, output_modality, rows in (
            ("text", "text", [[12, 512], [13, 512]]),
            ("audio", "audio", [[151667, 7], [151667, 8], [151667, 9]]),
        ):
            state = MossSpeechState(
                input_grid=rows,
                output_modality=output_modality,
                max_new_tokens=4,
                temperature=0,
                top_k=1,
                repetition_penalty=1,
            )
            with torch.device("cpu"):
                request = build(
                    StagePayload(request_id, OmniRequest(inputs=None), state.to_dict())
                )
            assert isinstance(request, MossSpeechSGLangRequestData)
            requests.append(request)
        assert lifecycle.snapshot() == {"requests": 2, "active": 2, "pending": 0}
        prompts = [
            runner.rows_to_embeddings(request.prompt_rows) for request in requests
        ]
        feedback = runner.rows_to_embeddings(torch.tensor([[14, 512], [151667, 10]]))
        packed_prefill, packed_backend = prefill(model, prompts, ["text", "audio"])
        packed_decode = decode(
            model, packed_backend, [2, 3], feedback, ["text", "audio"]
        )
        for index, prompt in enumerate(prompts):
            isolated_prefill, isolated_backend = prefill(model, [prompt], ["isolated"])
            isolated_decode = decode(
                model,
                isolated_backend,
                [len(prompt)],
                feedback[index : index + 1],
                ["isolated"],
            )
            for packed, isolated in (
                (packed_prefill.text_logits, isolated_prefill.text_logits),
                (packed_prefill.audio_logits, isolated_prefill.audio_logits),
                (packed_decode.text_logits, isolated_decode.text_logits),
                (packed_decode.audio_logits, isolated_decode.audio_logits),
            ):
                assert packed is not None and isolated is not None
                torch.testing.assert_close(packed[index], isolated[0], rtol=0, atol=0)
            packed_slots = packed_backend.req_to_token_pool.req_to_token[
                index + 1, : len(prompt) + 1
            ].long()
            isolated_slots = isolated_backend.req_to_token_pool.req_to_token[
                1, : len(prompt) + 1
            ].long()
            for layer_id in model.attention_layer_ids():
                for packed_buffer, isolated_buffer in (
                    (
                        packed_backend.token_to_kv_pool.get_key_buffer(layer_id),
                        isolated_backend.token_to_kv_pool.get_key_buffer(layer_id),
                    ),
                    (
                        packed_backend.token_to_kv_pool.get_value_buffer(layer_id),
                        isolated_backend.token_to_kv_pool.get_value_buffer(layer_id),
                    ),
                ):
                    torch.testing.assert_close(
                        packed_buffer[packed_slots],
                        isolated_buffer[isolated_slots],
                        rtol=0,
                        atol=0,
                    )
        runner.report_mixed_batch = True
        scheduled = [
            SchedulerRequest(request_id=request.req.rid, data=request)
            for request in requests
        ]
        with (
            torch.device("cpu"),
            caplog.at_level(
                "INFO", logger="sglang_omni.models.moss_speech.model_runner"
            ),
        ):
            runner.collect_step(GenerationBatchResult(), scheduled, phase="decode")
            decode(model, packed_backend, [2, 3], feedback, ["text", "audio"])
            runner.collect_step(GenerationBatchResult(), scheduled, phase="decode")
        assert (
            sum(
                "executed mixed-output AR batch" in record.message
                for record in caplog.records
            )
            == 1
        )
        assert not runner.report_mixed_batch
        assert all(len(request.output_rows) == 2 for request in requests)
        for request_id in ("text", "audio"):
            lifecycle.release(request_id)
        assert lifecycle.snapshot() == {"requests": 0, "active": 0, "pending": 0}
