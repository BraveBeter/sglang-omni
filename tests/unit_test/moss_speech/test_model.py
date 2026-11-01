# SPDX-License-Identifier: Apache-2.0
"""Native weight-loading, numerical, and dual-tail attention contracts."""

import os
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from torch.nn.functional import scaled_dot_product_attention
from transformers import Qwen3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_speech.attention import MossSpeechTorchNativeBackend
from sglang_omni.models.moss_speech.hf_config import (
    MossSpeechConfig,
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.model_layers import (
    MossSpeechRMSNorm,
    reference_rope_frequencies,
    request_linear,
)
from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel
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
