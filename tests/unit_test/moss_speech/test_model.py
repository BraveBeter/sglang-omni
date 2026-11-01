# SPDX-License-Identifier: Apache-2.0
"""Native weight-loading, numerical, and dual-tail attention contracts."""

import os
import threading
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.utils import DynamicGradMode
from torch.nn.functional import scaled_dot_product_attention
from transformers import Qwen3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_speech.attention import MossSpeechTorchNativeBackend
from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
from sglang_omni.models.moss_speech.hf_config import (
    MossSpeechConfig,
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.model_layers import (
    MossSpeechRMSNorm,
    reference_rope_frequencies,
    request_linear,
)
from sglang_omni.models.moss_speech.native_norm import NativeNorm
from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.models.moss_speech.projection_graphs import (
    MossSpeechProjectionGraphs,
    PreparedSection,
)
from sglang_omni.models.moss_speech.request_builders import (
    make_moss_speech_scheduler_adapters,
)
from sglang_omni.models.moss_speech.request_data import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.request_lifecycle import MossSpeechRequestLifecycle
from sglang_omni.models.moss_speech.sglang_model import (
    MossSpeechModelOutput,
    MossSpeechSGLangModel,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.fixtures.moss_speech_model import (
    RowCommitRunner,
    assert_ar_combination,
    assert_full_decode_graph,
    assert_full_prefill_graph,
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


def assert_native_request_isolation(
    device: torch.device, *, use_host_attention_metadata: bool
) -> None:
    with native_runtime(device), torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(183)
        model = tiny_model(device)
        prompt_embeddings = [
            torch.randn((length, 128), generator=generator, device=device)
            for length in (2, 3)
        ]
        feedback_embeddings = torch.randn((2, 128), generator=generator, device=device)
        packed_prefill, packed_backend = prefill(
            model,
            prompt_embeddings,
            ["first", "second"],
            use_host_attention_metadata=use_host_attention_metadata,
        )
        packed_decode = decode(
            model, packed_backend, [2, 3], feedback_embeddings, ["first", "second"]
        )
        for request_index, prompt in enumerate(prompt_embeddings):
            isolated_prefill, isolated_backend = prefill(
                model,
                [prompt],
                ["isolated"],
                use_host_attention_metadata=use_host_attention_metadata,
            )
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
@pytest.mark.parametrize("use_host_attention_metadata", [False, True])
def test_native_cpu_prefill_decode_keep_requests_and_tail_kv_independent(
    use_host_attention_metadata: bool,
) -> None:
    assert_native_request_isolation(
        torch.device("cpu"), use_host_attention_metadata=use_host_attention_metadata
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("use_host_attention_metadata", [False, True])
def test_native_cuda_prefill_decode_keep_requests_and_tail_kv_independent(
    use_host_attention_metadata: bool,
) -> None:
    require_cuda()
    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        pytest.skip("Run the accelerator lane without the CPU engine override")
    else:
        assert_native_request_isolation(
            torch.device("cuda", 0),
            use_host_attention_metadata=use_host_attention_metadata,
        )


@NATIVE_CPU
@pytest.mark.parametrize("with_residual", [False, True])
def test_projection_sections_preserve_independent_hf_math(with_residual: bool) -> None:
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu"))
        layer = model.layers[0]
        generator = torch.Generator().manual_seed(731)
        hidden_states = torch.randn(3, 128, generator=generator).bfloat16()
        residual = (
            torch.randn(3, 128, generator=generator).bfloat16()
            if with_residual
            else None
        )
        token_lengths = [1, 2]
        positions = torch.tensor([2, 6, 9])

        def normalize(
            values: torch.Tensor, normalization: MossSpeechRMSNorm
        ) -> torch.Tensor:
            reference = Qwen3RMSNorm(
                normalization.weight.numel(), eps=normalization.variance_epsilon
            ).bfloat16()
            reference.weight.copy_(normalization.weight)
            return torch.cat(
                [reference(chunk) for chunk in values.split(token_lengths)], dim=0
            )

        def linear(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    torch.nn.functional.linear(chunk, weight)
                    for chunk in values.split(token_lengths)
                ],
                dim=0,
            )

        expected_residual = (
            hidden_states if residual is None else hidden_states + residual
        )
        normalized = normalize(expected_residual, layer.input_layernorm)
        attention = layer.self_attn
        query_weight, key_weight, value_weight = attention.qkv_proj.weight.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=0
        )
        query = normalize(
            linear(normalized, query_weight).reshape(
                -1, attention.num_heads, attention.head_dim
            ),
            attention.q_norm,
        )
        key = normalize(
            linear(normalized, key_weight).reshape(
                -1, attention.num_kv_heads, attention.head_dim
            ),
            attention.k_norm,
        )
        reference_rope = Qwen3RotaryEmbedding(
            Qwen3Config(head_dim=attention.head_dim, rope_theta=attention.rope_theta)
        )
        cosine, sine = reference_rope(hidden_states, positions.unsqueeze(0))
        rotated = []
        for values in (query, key):
            left, right = values.chunk(2, dim=-1)
            rotated.append(
                (
                    values * cosine[0, :, None, :]
                    + torch.cat((-right, left), dim=-1) * sine[0, :, None, :]
                ).flatten(1)
            )
        prepared = layer.prepare_projection(
            hidden_states, residual, positions, token_lengths
        )
        for actual, expected in zip(
            prepared,
            (*rotated, linear(normalized, value_weight), expected_residual),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        attention_output = torch.randn(
            3, attention.q_size, generator=generator
        ).bfloat16()
        rounded_residual = (
            linear(attention_output, attention.o_proj.weight) + expected_residual
        )
        normalized = normalize(rounded_residual, layer.post_attention_layernorm)
        gate_weight, up_weight = layer.mlp.gate_up_proj.weight.chunk(2, dim=0)
        expected_hidden = linear(
            torch.nn.functional.silu(linear(normalized, gate_weight))
            * linear(normalized, up_weight),
            layer.mlp.down_proj.weight,
        )
        actual_hidden, actual_residual = layer.finish_projection(
            attention_output, expected_residual, token_lengths
        )
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
        torch.testing.assert_close(actual_residual, rounded_residual, rtol=0, atol=0)


@NATIVE_CPU
def test_projection_startup_rejects_cpu_before_cuda_allocation() -> None:
    with native_runtime(torch.device("cpu")):
        model = tiny_model(torch.device("cpu"))
        with patch.object(torch.cuda, "CUDAGraph") as graph:
            with pytest.raises(ValueError, match="CUDA"):
                MossSpeechProjectionGraphs(
                    [*model.layers, *model.text_block, *model.audio_block], batch_size=1
                )
            graph.assert_not_called()
        assert model.projection_graphs is None


@NATIVE_CPU
@pytest.mark.parametrize("failure", ["stream", "reset"])
def test_projection_cleanup_attempts_all_graphs_and_detaches_owner(
    failure: str,
) -> None:
    with native_runtime(torch.device("cpu")):
        model = tiny_model(torch.device("cpu"))
        layers = [*model.layers, *model.text_block, *model.audio_block]
        owner = MossSpeechProjectionGraphs.__new__(MossSpeechProjectionGraphs)
        owner.lock = threading.RLock()
        owner.owner_stream = Mock(spec=["synchronize"])
        owner.capture_stream = Mock(spec=["synchronize"])
        owner.layers = tuple(layers)
        owner.closed = False
        buffers = torch.ones(1)
        sections = [
            PreparedSection(
                graph=Mock(spec=["reset"]),
                inputs=(buffers, buffers, None),
                outputs=(buffers, buffers),
            )
            for _ in range(2 * len(layers))
        ]
        owner.sections = {
            (layer, section): prepared
            for (layer, section), prepared in zip(
                [
                    (layer, section)
                    for layer in layers
                    for section in ("prepare", "finish")
                ],
                sections,
            )
        }
        owner.inverse_frequencies = {layer: buffers for layer in layers}
        first_error = RuntimeError("First projection cleanup failure")
        if failure == "stream":
            owner.owner_stream.synchronize.side_effect = first_error
        else:
            sections[0].graph.reset.side_effect = first_error
        sections[-1].graph.reset.side_effect = RuntimeError("Later reset failure")
        with patch(
            "sglang_omni.models.moss_speech.engine_builder.MossSpeechProjectionGraphs",
            autospec=True,
            return_value=owner,
        ) as prepare:
            disabled = MossSpeechEngineBuilder(streaming=True)
            disabled.setup_model_resources(
                model, Mock(), generation_cuda_graph_enabled=False
            )
            prepare.assert_not_called()
            builder = MossSpeechEngineBuilder(
                streaming=True, projection_graph_batch_size=2
            )
            builder.setup_model_resources(
                model, Mock(), generation_cuda_graph_enabled=False
            )
            prepare.assert_called_once_with(layers, batch_size=2)
        shutdown = builder.extra_scheduler_callbacks()["shutdown_callback"]
        assert shutdown is not None
        with pytest.raises(RuntimeError) as caught:
            shutdown()
        assert caught.value is first_error
        owner.owner_stream.synchronize.assert_called_once_with()
        owner.capture_stream.synchronize.assert_called_once_with()
        for prepared in sections:
            prepared.graph.reset.assert_called_once_with()
            assert prepared.inputs is None and prepared.outputs is None
        assert not owner.sections and not owner.inverse_frequencies
        assert model.projection_graphs is None
        assert builder.projection_graphs is None and builder.projection_model is None
        shutdown()


def projection_decode(
    model: MossSpeechSGLangModel,
    backend: TorchNativeAttnBackend,
    request_indices: list[int],
    token_lengths: list[int],
    token_slots: list[int],
    feedback_embeddings: torch.Tensor,
    request_ids: list[str],
) -> MossSpeechModelOutput:
    device = feedback_embeddings.device
    for request_index, token_length, token_slot in zip(
        request_indices, token_lengths, token_slots
    ):
        backend.req_to_token_pool.req_to_token[request_index, token_length] = token_slot
    model.decode_input_embedding.weight[: len(request_indices)].copy_(
        feedback_embeddings
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=len(request_indices),
        input_ids=torch.arange(len(request_indices), device=device),
        req_pool_indices=torch.tensor(request_indices, device=device),
        seq_lens=torch.tensor(
            [length + 1 for length in token_lengths], dtype=torch.int32, device=device
        ),
        out_cache_loc=torch.tensor(token_slots, device=device),
        seq_lens_sum=sum(token_lengths) + len(request_indices),
        positions=torch.tensor(token_lengths, device=device),
        rids=request_ids,
    )
    backend.init_forward_metadata(batch)
    with forward_context(ForwardContext(attn_backend=backend)):
        return model(batch.input_ids, batch.positions, batch)


def assert_projection_outputs_and_kv(
    actual: MossSpeechModelOutput,
    expected: MossSpeechModelOutput,
    model: MossSpeechSGLangModel,
    backend: TorchNativeAttnBackend,
    reference_backend: TorchNativeAttnBackend,
    written_slots: set[int],
) -> None:
    for actual_logits, expected_logits in (
        (actual.text_logits, expected.text_logits),
        (actual.audio_logits, expected.audio_logits),
        (actual.hidden_states, expected.hidden_states),
    ):
        assert actual_logits is not None and expected_logits is not None
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    used_slots = torch.tensor(
        sorted(written_slots), device=model.embed_tokens.weight.device
    )
    for layer_id in model.attention_layer_ids():
        for actual_cache, expected_cache in (
            (
                backend.token_to_kv_pool.get_key_buffer(layer_id),
                reference_backend.token_to_kv_pool.get_key_buffer(layer_id),
            ),
            (
                backend.token_to_kv_pool.get_value_buffer(layer_id),
                reference_backend.token_to_kv_pool.get_value_buffer(layer_id),
            ),
        ):
            torch.testing.assert_close(
                actual_cache[used_slots], expected_cache[used_slots], rtol=0, atol=0
            )


@pytest.mark.accelerator
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_projection_cuda_replays_full_model_and_all_kv_exactly(batch_size: int) -> None:
    require_cuda()
    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        pytest.skip("Run the accelerator lane without the CPU engine override")
    else:
        pass
    device = torch.device("cuda", 0)
    with native_runtime(device):
        model = tiny_model(device, decode_batch_capacity=4).eval()
        reference = tiny_model(device, decode_batch_capacity=4).eval()
        reference.load_state_dict(model.state_dict())
        generator = torch.Generator(device=device).manual_seed(731)
        request_count = max(batch_size, 2)
        prompts = [
            torch.randn(2 + index % 2, 128, generator=generator, device=device)
            for index in range(request_count)
        ]
        request_ids = [f"request-{index}" for index in range(1, request_count + 1)]
        layers = [*model.layers, *model.text_block, *model.audio_block]
        with torch.no_grad():
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(device).clone()
            owner = MossSpeechProjectionGraphs(layers, batch_size=batch_size)
            model.projection_graphs = owner
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
            try:
                actual_prefill, backend = prefill(model, prompts, request_ids)
                expected_prefill, reference_backend = prefill(
                    reference, prompts, request_ids
                )
                assert owner.replays == 0
                assert actual_prefill.text_logits is not None
                assert expected_prefill.text_logits is not None
                assert actual_prefill.audio_logits is not None
                assert expected_prefill.audio_logits is not None
                torch.testing.assert_close(
                    actual_prefill.text_logits,
                    expected_prefill.text_logits,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual_prefill.audio_logits,
                    expected_prefill.audio_logits,
                    rtol=0,
                    atol=0,
                )
                token_lengths = {
                    index: len(prompt) for index, prompt in enumerate(prompts, 1)
                }
                next_slot = sum(token_lengths.values()) + 1
                written_slots = set(range(1, next_slot))
                eligible = list(range(1, batch_size + 1))
                other_batch = [1, 2] if batch_size == 1 else eligible[:-1]
                schedules = [eligible, list(reversed(eligible)), other_batch, eligible]
                last_slots: dict[int, int] = {}
                for request_indices in schedules:
                    feedback = torch.randn(
                        len(request_indices), 128, generator=generator, device=device
                    )
                    slots = list(range(next_slot, next_slot + len(request_indices)))
                    next_slot += len(request_indices)
                    lengths = [token_lengths[index] for index in request_indices]
                    identifiers = [request_ids[index - 1] for index in request_indices]
                    previous_replays = owner.replays
                    previous_embedding_version = (
                        model.decode_input_embedding.weight._version  # note (Codex): The external Tensor mutation counter proves staging updates; noqa: leading-underscore.
                    )
                    actual = projection_decode(
                        model,
                        backend,
                        request_indices,
                        lengths,
                        slots,
                        feedback,
                        identifiers,
                    )
                    expected = projection_decode(
                        reference,
                        reference_backend,
                        request_indices,
                        lengths,
                        slots,
                        feedback,
                        identifiers,
                    )
                    assert (
                        model.decode_input_embedding.weight._version  # note (Codex): The external Tensor mutation counter proves staging updates; noqa: leading-underscore.
                        > previous_embedding_version
                    )
                    if len(request_indices) == batch_size:
                        assert owner.replays > previous_replays
                    else:
                        assert owner.replays == previous_replays
                    written_slots.update(slots)
                    assert_projection_outputs_and_kv(
                        actual,
                        expected,
                        model,
                        backend,
                        reference_backend,
                        written_slots,
                    )
                    for index, slot in zip(request_indices, slots):
                        token_lengths[index] += 1
                        last_slots[index] = slot
                reusable_slot = last_slots[1]
                for attention_backend in (backend, reference_backend):
                    attention_backend.req_to_token_pool.req_to_token[1].copy_(
                        attention_backend.req_to_token_pool.req_to_token[2]
                    )
                token_lengths[1] = token_lengths[2]
                slots = [reusable_slot, *range(next_slot, next_slot + batch_size - 1)]
                feedback = torch.randn(
                    batch_size, 128, generator=generator, device=device
                )
                lengths = [token_lengths[index] for index in eligible]
                identifiers = ["replacement", *request_ids[1:batch_size]]
                previous_replays = owner.replays
                actual = projection_decode(
                    model, backend, eligible, lengths, slots, feedback, identifiers
                )
                expected = projection_decode(
                    reference,
                    reference_backend,
                    eligible,
                    lengths,
                    slots,
                    feedback,
                    identifiers,
                )
                assert owner.replays > previous_replays
                written_slots.update(slots)
                assert_projection_outputs_and_kv(
                    actual, expected, model, backend, reference_backend, written_slots
                )
                positions = torch.tensor(lengths, device=device)
                assert owner.can_replay(feedback, positions, is_decode=True)
                for layer, residual in (
                    (model.layers[0], None),
                    (
                        model.layers[1],
                        torch.randn(
                            batch_size, 128, generator=generator, device=device
                        ),
                    ),
                ):
                    retained = owner.prepare(layer, feedback, residual, positions)
                    assert retained is not None
                    saved = tuple(value.clone() for value in retained)
                    fresh = torch.randn(
                        batch_size, 128, generator=generator, device=device
                    )
                    fresh_residual = (
                        None
                        if residual is None
                        else torch.randn(
                            batch_size, 128, generator=generator, device=device
                        )
                    )
                    replayed = owner.prepare(
                        layer, fresh, fresh_residual, positions + 1
                    )
                    assert replayed is not None
                    expected_sections = layer.prepare_projection(
                        fresh, fresh_residual, positions + 1, [1] * batch_size
                    )
                    for value, snapshot, current, reference_value in zip(
                        retained, saved, replayed, expected_sections
                    ):
                        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
                        torch.testing.assert_close(
                            current, reference_value, rtol=0, atol=0
                        )
                    finished = owner.finish(layer, feedback, retained[-1])
                    assert finished is not None
                    finished_snapshot = tuple(value.clone() for value in finished)
                    fresh_finished = owner.finish(layer, fresh, replayed[-1])
                    assert fresh_finished is not None
                    expected_finished = layer.finish_projection(
                        fresh, replayed[-1], [1] * batch_size
                    )
                    for value, snapshot, current, reference_value in zip(
                        finished, finished_snapshot, fresh_finished, expected_finished
                    ):
                        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
                        torch.testing.assert_close(
                            current, reference_value, rtol=0, atol=0
                        )
                with torch.enable_grad():
                    assert not owner.can_replay(feedback, positions, is_decode=True)
                model.layers[0].self_attn.qkv_proj.weight.add_(0)
                assert not owner.can_replay(feedback, positions, is_decode=True)
            finally:
                owner.close()
                owner.close()
            assert not owner.can_replay(feedback, positions, is_decode=True)


def assert_host_attention_matches_native(
    device: torch.device, *, prompt_lengths: tuple[int, int] = (2, 3)
) -> None:
    with native_runtime(device), torch.no_grad():
        model = tiny_model(device)
        assert all(parameter.device == device for parameter in model.parameters())
        generator = torch.Generator(device=device).manual_seed(691)
        prompts = [
            torch.randn((length, 128), generator=generator, device=device)
            for length in prompt_lengths
        ]
        feedback = torch.randn((2, 128), generator=generator, device=device)
        # note (Codex): Metadata allocation must stay on CPU even when the model is on CUDA.
        with torch.device("cpu"):
            baseline_output, baseline = prefill(
                model, prompts, ["first", "second"], request_slots=[3, 1]
            )
            candidate_output, candidate = prefill(
                model,
                prompts,
                ["first", "second"],
                request_slots=[3, 1],
                use_host_attention_metadata=True,
            )
            assert isinstance(candidate, MossSpeechTorchNativeBackend)
            for baseline_logits, candidate_logits in (
                (baseline_output.text_logits, candidate_output.text_logits),
                (baseline_output.audio_logits, candidate_output.audio_logits),
            ):
                torch.testing.assert_close(
                    baseline_logits, candidate_logits, rtol=0, atol=0
                )
            for decode_lengths, slots, request_ids in (
                ([prompt_lengths[1], prompt_lengths[0]], [1, 3], ["second", "first"]),
                ([prompt_lengths[0] + 1], [3], ["first"]),
            ):
                baseline_output = decode(
                    model,
                    baseline,
                    decode_lengths,
                    feedback[: len(slots)],
                    request_ids,
                    request_slots=slots,
                )
                candidate_output = decode(
                    model,
                    candidate,
                    decode_lengths,
                    feedback[: len(slots)],
                    request_ids,
                    request_slots=slots,
                )
                for baseline_logits, candidate_logits in (
                    (baseline_output.text_logits, candidate_output.text_logits),
                    (baseline_output.audio_logits, candidate_output.audio_logits),
                ):
                    torch.testing.assert_close(
                        baseline_logits, candidate_logits, rtol=0, atol=0
                    )
                for layer_index in range(model.total_attention_layers):
                    for baseline_cache, candidate_cache in (
                        (
                            baseline.token_to_kv_pool.get_key_buffer(layer_index),
                            candidate.token_to_kv_pool.get_key_buffer(layer_index),
                        ),
                        (
                            baseline.token_to_kv_pool.get_value_buffer(layer_index),
                            candidate.token_to_kv_pool.get_value_buffer(layer_index),
                        ),
                    ):
                        torch.testing.assert_close(
                            baseline_cache[1 : sum(prompt_lengths) + 3],
                            candidate_cache[1 : sum(prompt_lengths) + 3],
                            rtol=0,
                            atol=0,
                        )


@NATIVE_CPU
def test_host_metadata_cpu_preserves_model_logits_and_all_tail_kv() -> None:
    assert_host_attention_matches_native(torch.device("cpu"))


@pytest.mark.accelerator
def test_host_metadata_cuda_preserves_model_logits_and_all_tail_kv() -> None:
    require_cuda()
    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        pytest.skip("Run the accelerator lane without the CPU engine override")
    else:
        assert_host_attention_matches_native(torch.device("cuda", 0))


@NATIVE_CPU
@pytest.mark.parametrize("use_cpu_mirrors", [False, True])
def test_host_metadata_prefix_freshness_and_lifetime_use_real_sdpa(
    use_cpu_mirrors: bool,
) -> None:
    device = torch.device("cpu")
    with native_runtime(device), torch.no_grad():
        model = tiny_model(device)
        generator = torch.Generator(device=device).manual_seed(421)
        prompts = [torch.randn((length, 128), generator=generator) for length in (2, 3)]
        _, baseline = prefill(model, prompts, ["first", "second"], request_slots=[3, 1])
        _, candidate = prefill(
            model,
            prompts,
            ["first", "second"],
            request_slots=[3, 1],
            use_host_attention_metadata=True,
        )
        assert isinstance(candidate, MossSpeechTorchNativeBackend)
        layer = RadixAttention(
            num_heads=2, head_dim=64, scaling=64**-0.5, num_kv_heads=1, layer_id=0
        )
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.zeros(3, dtype=torch.long),
            req_pool_indices=torch.tensor([3, 1]),
            req_pool_indices_cpu=torch.tensor([3, 1]),
            seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4, 4], dtype=torch.int64),
            seq_lens_sum=8,
            out_cache_loc=torch.tensor([6, 7, 8]),
            extend_seq_lens=torch.tensor([2, 1], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([2, 3], dtype=torch.int32),
            extend_seq_lens_cpu=[2, 1],
            extend_prefix_lens_cpu=[2, 3],
        )
        if not use_cpu_mirrors:
            batch.seq_lens_cpu = None
            batch.req_pool_indices_cpu = None
            batch.extend_seq_lens_cpu = None
            batch.extend_prefix_lens_cpu = None
        with pytest.raises(RuntimeError, match="this forward batch"):
            candidate.host_batch_view(batch)
        for backend in (baseline, candidate):
            backend.req_to_token_pool.req_to_token[3, 2:4] = torch.tensor([6, 7])
            backend.req_to_token_pool.req_to_token[1, 3] = 8
            backend.init_forward_metadata(batch)
        original_fields = vars(batch).copy()
        if use_cpu_mirrors:
            batch.seq_lens_cpu.zero_()
            batch.req_pool_indices_cpu.zero_()
            batch.extend_seq_lens_cpu[:] = [0, 0]
            batch.extend_prefix_lens_cpu[:] = [0, 0]
        snapshot = candidate.host_metadata
        assert snapshot is not None
        assert snapshot.sequence_lengths.device.type == "cpu"
        assert snapshot.sequence_lengths.dtype == batch.seq_lens.dtype
        assert snapshot.request_slots.device.type == "cpu"
        assert snapshot.extend_lengths.device.type == "cpu"
        assert snapshot.extend_prefix_lengths.device.type == "cpu"
        q = torch.randn((3, 128), generator=generator)
        k = torch.randn((3, 1, 64), generator=generator)
        v = torch.randn((3, 1, 64), generator=generator)
        expected = baseline.forward_extend(q, k, v, layer, batch)
        actual = candidate.forward_extend(q, k, v, layer, batch)
        references = []
        query_offset = 0
        for token_slots, prefix_length, extend_length in (
            ([1, 2, 6, 7], 2, 2),
            ([3, 4, 5, 8], 3, 1),
        ):
            query = (
                q[query_offset : query_offset + extend_length]
                .view(-1, 2, 64)
                .movedim(0, 1)
            )
            keys = candidate.token_to_kv_pool.get_key_buffer(0)[token_slots].movedim(
                0, 1
            )
            values = candidate.token_to_kv_pool.get_value_buffer(0)[
                token_slots
            ].movedim(0, 1)
            mask = (
                torch.arange(4)[None, :]
                <= prefix_length + torch.arange(extend_length)[:, None]
            )
            references.append(
                scaled_dot_product_attention(
                    query[None],
                    keys[None],
                    values[None],
                    attn_mask=mask,
                    enable_gqa=True,
                    scale=layer.scaling,
                )
                .squeeze(0)
                .movedim(0, 1)
                .reshape(extend_length, 128)
            )
            query_offset += extend_length
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, torch.cat(references), rtol=0, atol=0)
        cached_keys = candidate.token_to_kv_pool.get_key_buffer(0)[1:9].clone()
        cached_values = candidate.token_to_kv_pool.get_value_buffer(0)[1:9].clone()
        torch.testing.assert_close(
            candidate.forward_extend(
                q, k + 1, v + 1, layer, batch, save_kv_cache=False
            ),
            actual,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate.token_to_kv_pool.get_key_buffer(0)[1:9],
            cached_keys,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate.token_to_kv_pool.get_value_buffer(0)[1:9],
            cached_values,
            rtol=0,
            atol=0,
        )
        assert all(
            vars(batch)[name] is value for name, value in original_fields.items()
        )
        assert batch.seq_lens.tolist() == [4, 4]
        assert batch.req_pool_indices.tolist() == [3, 1]
        assert batch.extend_seq_lens.tolist() == [2, 1]
        assert batch.extend_prefix_lens.tolist() == [2, 3]
        assert batch.out_cache_loc.tolist() == [6, 7, 8]
        stale_batch = batch
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.zeros(1, dtype=torch.long),
            req_pool_indices=torch.tensor([1]),
            req_pool_indices_cpu=torch.tensor([3]),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            seq_lens_sum=4,
            out_cache_loc=torch.tensor([8]),
        )
        candidate.init_forward_metadata(batch)
        assert candidate.host_metadata.request_slots.tolist() == [1]
        with pytest.raises(RuntimeError, match="this forward batch"):
            candidate.host_batch_view(stale_batch)
        torch.testing.assert_close(
            candidate.forward_decode(
                q[:1], None, None, layer, batch, save_kv_cache=False
            ),
            baseline.forward_decode(
                q[:1], None, None, layer, batch, save_kv_cache=False
            ),
            rtol=0,
            atol=0,
        )
        batch_reference = weakref.ref(batch)
        del batch
        assert batch_reference() is None
        candidate.init_forward_metadata(
            replace(stale_batch, forward_mode=ForwardMode.IDLE)
        )
        assert candidate.host_metadata is None
        with pytest.raises(ValueError, match="ordinary extend or decode"):
            candidate.init_forward_metadata(
                replace(stale_batch, forward_mode=ForwardMode.MIXED)
            )
        assert candidate.host_metadata is None
        recovered_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.zeros(1, dtype=torch.long),
            req_pool_indices=torch.tensor([3]),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            seq_lens_sum=4,
            out_cache_loc=torch.tensor([7]),
        )
        candidate.init_forward_metadata(recovered_batch)
        assert candidate.host_metadata.request_slots.tolist() == [3]
        torch.testing.assert_close(
            candidate.forward_decode(
                q[:1], None, None, layer, recovered_batch, save_kv_cache=False
            ),
            baseline.forward_decode(
                q[:1], None, None, layer, recovered_batch, save_kv_cache=False
            ),
            rtol=0,
            atol=0,
        )


@NATIVE_CPU
def test_host_metadata_cpu_preserves_long_context_model_logits_and_kv() -> None:
    assert_host_attention_matches_native(torch.device("cpu"), prompt_lengths=(1025, 3))


@pytest.mark.accelerator
def test_host_metadata_cuda_preserves_long_context_model_logits_and_kv() -> None:
    require_cuda()
    assert_host_attention_matches_native(
        torch.device("cuda", 0), prompt_lengths=(1025, 3)
    )


def test_native_norm_cpu_and_grad_fallback() -> None:
    native = NativeNorm.__new__(NativeNorm)
    hidden = torch.randn(1, 4096, requires_grad=True)
    assert native.apply(hidden, torch.ones(4096), 1e-6) is None
    with torch.inference_mode():
        assert native.apply(hidden, torch.ones(4096), 1e-6) is None


@pytest.mark.accelerator
def test_native_norm_cuda_rounding_and_fallback() -> None:
    require_cuda()
    library = os.environ.get("MOSS_NATIVE_NORM_LIBRARY")
    if not library:
        pytest.skip("Set MOSS_NATIVE_NORM_LIBRARY to test the optional backend")
    else:
        pass
    native = NativeNorm(library)
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            for shape in ((1, 4096), (1, 32, 128), (1, 8, 128)):
                generator = torch.Generator(device="cuda").manual_seed(20261031)
                hidden = torch.randn(
                    shape, device="cuda", dtype=torch.bfloat16, generator=generator
                )
                weight = torch.randn(
                    shape[-1], device="cuda", dtype=torch.bfloat16, generator=generator
                )
                for epsilon in (1e-6, 1e-5):
                    reference = hidden.float() * torch.rsqrt(
                        hidden.float().square().mean(-1, keepdim=True) + epsilon
                    )
                    reference = weight * reference.to(hidden.dtype)
                    actual = native.apply(hidden, weight, epsilon)
                    assert actual is not None
                    assert actual.data_ptr() != hidden.data_ptr()
                    assert torch.equal(
                        actual.view(torch.uint8), reference.view(torch.uint8)
                    )
                    saved = actual.clone()
                    native.apply(hidden + 1, weight, epsilon)
                    assert torch.equal(actual, saved)
                assert native.apply(hidden.float(), weight, 1e-6) is None
                assert native.apply(hidden.expand(2, *shape[1:]), weight, 1e-6) is None
    assert native.apply(hidden, weight, 1e-6) is None
    assert native.calls == 24
    norm = MossSpeechRMSNorm(4096).to(device="cuda", dtype=torch.bfloat16).eval()
    norm.native_norm = native
    with DynamicGradMode():
        routed = norm(torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16))
        assert routed.shape == (1, 4096) and native.calls == 25
        noncontiguous = torch.randn(1, 8192, device="cuda", dtype=torch.bfloat16)[
            :, ::2
        ]
        assert native.apply(noncontiguous, norm.weight, 1e-6) is None
    hidden = torch.randn(
        1, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    norm(hidden).float().sum().backward()
    assert hidden.grad is not None and norm.weight.grad is not None
    assert native.calls == 25


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


@pytest.mark.accelerator
def test_full_decode_graph_matches_eager_and_owns_outputs() -> None:
    require_cuda()
    with native_runtime(torch.device("cuda", 0)), torch.no_grad():
        model = tiny_model(torch.device("cuda", 0)).eval()
        assert_full_decode_graph(model)


@pytest.mark.accelerator
def test_full_prefill_graph_matches_eager_and_scattered_kv() -> None:
    require_cuda()
    with native_runtime(torch.device("cuda", 0)), torch.no_grad():
        model = tiny_model(torch.device("cuda", 0)).eval()
        assert_full_prefill_graph(model)


@pytest.mark.accelerator
@pytest.mark.parametrize("projection_batch_size", [1, 4])
def test_ar_combination_cuda_preserves_complete_heads_and_kv(
    projection_batch_size: int,
) -> None:
    require_cuda()
    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        pytest.skip("Run the accelerator lane without the CPU engine override")
    else:
        pass
    device = torch.device("cuda", 0)
    with native_runtime(device):
        model = tiny_model(
            device, audio_vocab_size=1024, decode_batch_capacity=4
        ).eval()
        report = assert_ar_combination(
            model,
            [
                "ar-decode-graph",
                "ar-projection-graph",
                "context-prefill-graph",
                "ar-host-metadata",
                "mixed-modalities",
            ],
            projection_batch_size=projection_batch_size,
            decode_graph_lengths=tuple(range(1, 13)) + (512,),
        )
        assert report["complete_heads_exact"] and report["all_kv_exact"]
        assert all(
            result["executions"] > 0 for result in report["optimizations"].values()
        )
        assert (
            model.decode_graph is model.prefill_graph is model.projection_graphs is None
        )


@NATIVE_CPU
def test_ar_cleanup_attempts_every_owner_and_preserves_first_error() -> None:
    with native_runtime(torch.device("cpu")):
        model = tiny_model(torch.device("cpu"))
        builder = MossSpeechEngineBuilder(streaming=True)
        builder.projection_model = model
        owners = [Mock(spec=["close"]) for _ in range(3)]
        first_error = RuntimeError("first AR cleanup failure")
        owners[0].close.side_effect = first_error
        owners[-1].close.side_effect = RuntimeError("later AR cleanup failure")
        for attribute, owner in zip(
            ("decode_graph", "prefill_graph", "projection_graphs"), owners, strict=True
        ):
            setattr(model, attribute, owner)
            setattr(builder, attribute, owner)
        with pytest.raises(RuntimeError) as caught:
            builder.cleanup_build_failure()
        assert caught.value is first_error
        assert (
            model.decode_graph is model.prefill_graph is model.projection_graphs is None
        )
        assert (
            builder.decode_graph
            is builder.prefill_graph
            is builder.projection_graphs
            is None
        )
        assert builder.projection_model is None
        for owner in owners:
            owner.close.assert_called_once_with()
        builder.cleanup_build_failure()
        for owner in owners:
            owner.close.assert_called_once_with()
