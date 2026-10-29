# SPDX-License-Identifier: Apache-2.0
"""Native weight-loading, numerical, and dual-tail attention contracts."""

import os
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from transformers import Qwen3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from sglang_omni.model_runner.model_worker import ModelWorker
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
from sglang_omni.models.moss_speech.projection_graphs import (
    MossSpeechProjectionGraphs,
    PreparedSection,
)
from sglang_omni.models.moss_speech.sglang_model import (
    MossSpeechModelOutput,
    MossSpeechSGLangModel,
)
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.fixtures.moss_speech_model import (
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
